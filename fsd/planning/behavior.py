"""Behavior planner — finite state machine over perception context.

States (emitted as strings so callers can switch on them):

    LANE_KEEP        cruise at the lane speed
    LANE_CHANGE_L    executing a left lane change
    LANE_CHANGE_R    executing a right lane change
    FOLLOW_LEAD      pace a lead vehicle (IDM target speed)
    STOP_AT_LIGHT    controlled stop at a red/yellow light
    EMERGENCY_STOP   TTC / free-space violation -> full brake

Priority each tick: EMERGENCY > STOP_AT_LIGHT > lane-change bookkeeping
> FOLLOW_LEAD > LANE_KEEP. The returned target speed is the *minimum* over
all currently active constraints so, e.g., following a lead car toward a
red light takes the lower of the two speeds.

Lead following uses the Intelligent Driver Model:

    s*      = s0 + v*T + v*dv / (2*sqrt(a*b))
    acc     = a * (1 - (v/v0)^4 - (s*/s)^2)
    v_tgt   = clamp(v + acc * dt_eval, 0, cruise)

Lane changes are requested via `request_lane_change("LEFT"|"RIGHT")` and
only start once a clearance check on the target lane passes; they persist
for `lane_change_time_s` and then hand control back to LANE_KEEP.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from fsd.core.logger import get
from fsd.core.types import (
    DetectedObject,
    LightState,
    PerceptionOutput,
    VehicleState,
)

_log = get("planning.behavior")


def _to_ego(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    """World delta -> ego frame. Returns (longitudinal, lateral[+left])."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    return dx * cy + dy * sy, -dx * sy + dy * cy


class BehaviorPlanner:
    """FSM producing (state_name, target_speed_mps) each tick."""

    LANE_KEEP = "LANE_KEEP"
    LANE_CHANGE_LEFT = "LANE_CHANGE_L"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_R"
    FOLLOW_LEAD = "FOLLOW_LEAD"
    STOP_AT_LIGHT = "STOP_AT_LIGHT"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    STATES = (
        LANE_KEEP, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT,
        FOLLOW_LEAD, STOP_AT_LIGHT, EMERGENCY_STOP,
    )

    def __init__(
        self,
        cruise_speed_mps: float = 13.9,
        min_ttc_s: float = 1.5,
        min_free_space_m: float = 8.0,
        time_headway_s: float = 1.6,
        standstill_gap_m: float = 3.0,
        idm_accel_mps2: float = 2.0,
        idm_brake_mps2: float = 3.0,
        comfort_decel_mps2: float = 2.5,
        stop_margin_m: float = 4.0,
        follow_enter_m: float = 25.0,
        follow_exit_m: float = 33.0,
        corridor_half_width_m: float = 2.0,
        ego_half_length_m: float = 2.4,
        lane_width_m: float = 3.5,
        lane_change_time_s: float = 4.0,
        min_dwell_s: float = 0.8,
        no_lane_speed_factor: float = 0.7,
        junction_zone_m: float = 30.0,
        junction_speed_factor: float = 0.55,
    ) -> None:
        self.cruise_speed = float(cruise_speed_mps)
        self.min_ttc = float(min_ttc_s)
        self.min_free_space = float(min_free_space_m)
        self.time_headway = float(time_headway_s)
        self.standstill_gap = float(standstill_gap_m)
        self.idm_a = float(idm_accel_mps2)
        self.idm_b = float(idm_brake_mps2)
        self.comfort_decel = float(comfort_decel_mps2)
        self.stop_margin = float(stop_margin_m)
        self.follow_enter = float(follow_enter_m)
        self.follow_exit = float(follow_exit_m)
        self.half_width = float(corridor_half_width_m)
        self.ego_half_length = float(ego_half_length_m)
        self.lane_width = float(lane_width_m)
        self.lane_change_time = float(lane_change_time_s)
        self.min_dwell = float(min_dwell_s)
        self.no_lane_speed_factor = float(no_lane_speed_factor)
        self.junction_zone = float(junction_zone_m)
        self.junction_factor = float(junction_speed_factor)

        self.state: str = self.LANE_KEEP
        self._state_since: float | None = None
        self._pending_lane_change: Optional[str] = None   # "LEFT" | "RIGHT"

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #
    def decide(
        self,
        perception: PerceptionOutput,
        ego: VehicleState,
    ) -> Tuple[str, float]:
        """Advance the FSM once. Returns (state_name, target_speed)."""
        now = ego.timestamp
        if self._state_since is None:
            self._state_since = now

        cruise = self._cruise_target(perception)

        # ---- gather constraints --------------------------------------- #
        lead = self._lead_vehicle(perception, ego)
        ttc = math.inf
        if lead is not None:
            _, gap, closing, _ = lead
            if closing > 0.05:
                ttc = gap / closing

        emergency = (ttc < self.min_ttc
                     or (perception.free_space_ahead < self.min_free_space
                         and ego.speed > 0.5))
        if emergency:
            self._transition(self.EMERGENCY_STOP, now)
            return self.state, 0.0
        if self.state == self.EMERGENCY_STOP:
            # threat cleared — recover to lane keep
            self._transition(self.LANE_KEEP, now)

        # ---- collect active speed caps (min wins) ---------------------- #
        caps = [cruise]
        want_stop_light = False
        # distance to the constraining stop line; falls back to free space
        # only when the monitor has no geometry (smoke tests without TLs)
        d_light = min(perception.free_space_ahead,
                      getattr(perception, "stop_line_m", math.inf))
        if perception.light == LightState.RED:
            want_stop_light = True
        elif perception.light == LightState.YELLOW:
            # stop only if we still can, comfortably
            d_stop = d_light - self.stop_margin
            want_stop_light = (
                self._stopping_distance(ego.speed, self.comfort_decel)
                < d_stop)

        if want_stop_light:
            d_eff = d_light - self.stop_margin
            caps.append(math.sqrt(max(0.0, 2.0 * self.comfort_decel * d_eff)))

        # junction approach — taper cruise down to `junction_factor` at the
        # entry (Autoware-style intersection velocity): blending is linear
        # in distance so there is no step change at the zone boundary.
        d_j = getattr(perception, "junction_dist", math.inf)
        if d_j < self.junction_zone:
            t = d_j / self.junction_zone            # 0 inside -> 1 at edge
            caps.append(cruise * (self.junction_factor
                                  + (1.0 - self.junction_factor) * t))

        # posted speed limit from the road section (CARLA ground truth)
        v_lim = getattr(perception, "speed_limit_mps", math.inf)
        if math.isfinite(v_lim):
            caps.append(v_lim)

        in_lane_change = self.state in (self.LANE_CHANGE_LEFT,
                                        self.LANE_CHANGE_RIGHT)
        follow_active = False
        if lead is not None:
            gap = lead[1]
            enter = self.follow_exit if self.state == self.FOLLOW_LEAD \
                else self.follow_enter
            if gap < enter:
                # spacing cap applies even mid-lane-change
                caps.append(self._idm_speed(ego.speed, lead))
                follow_active = not in_lane_change

        # ---- state selection (priority order) --------------------------- #
        if want_stop_light:
            self._transition(self.STOP_AT_LIGHT, now)
        elif in_lane_change:
            # lane change holds until its timer expires
            if now - self._state_since >= self.lane_change_time:
                self._transition(self.LANE_KEEP, now)
        elif self._pending_lane_change is not None:
            if self._lane_change_clear(self._pending_lane_change,
                                       perception, ego):
                self._transition(
                    self.LANE_CHANGE_LEFT
                    if self._pending_lane_change == "LEFT"
                    else self.LANE_CHANGE_RIGHT, now)
                self._pending_lane_change = None
        elif follow_active:
            self._transition(self.FOLLOW_LEAD, now)
        elif self.state in (self.FOLLOW_LEAD, self.STOP_AT_LIGHT):
            self._transition(self.LANE_KEEP, now)
        else:
            self._transition(self.LANE_KEEP, now)

        return self.state, max(0.0, min(caps))

    def request_lane_change(self, direction: str) -> bool:
        """Ask for a lane change; executed once the target lane is clear."""
        direction = direction.upper()
        if direction not in ("LEFT", "RIGHT"):
            _log.warning("bad lane-change direction %r", direction)
            return False
        self._pending_lane_change = direction
        return True

    def cancel_lane_change(self) -> None:
        self._pending_lane_change = None
        if self.state in (self.LANE_CHANGE_LEFT, self.LANE_CHANGE_RIGHT):
            self._transition(self.LANE_KEEP, None)

    # ------------------------------------------------------------------ #
    # internals                                                           #
    # ------------------------------------------------------------------ #
    def _transition(self, new_state: str, now: float | None) -> None:
        if new_state == self.state:
            return
        # dwell hysteresis — never for safety states
        if (now is not None and self._state_since is not None
                and new_state not in (self.EMERGENCY_STOP,)
                and self.state != self.EMERGENCY_STOP
                and now - self._state_since < self.min_dwell):
            return
        _log.info("%s -> %s", self.state, new_state)
        self.state = new_state
        if now is not None:
            self._state_since = now

    def _cruise_target(self, perception: PerceptionOutput) -> float:
        if not perception.lane.detected:
            return self.cruise_speed * self.no_lane_speed_factor
        return self.cruise_speed

    def _stopping_distance(self, v: float, decel: float) -> float:
        return v * v / (2.0 * max(decel, 0.1))

    def _lead_vehicle(
        self,
        perception: PerceptionOutput,
        ego: VehicleState,
    ) -> Optional[Tuple[DetectedObject, float, float, float]]:
        """Nearest object ahead inside our lane corridor.

        Returns (obj, bumper_gap_m, closing_speed_mps, obj_speed_mps).
        """
        best = None
        best_gap = math.inf
        for o in perception.objects:
            lon, lat = _to_ego(o.position.x - ego.x,
                               o.position.y - ego.y, ego.yaw)
            half_w = max(self.half_width, o.bbox_extent.y + 0.4)
            if lon < -1.0 or abs(lat) > half_w:
                continue
            gap = max(lon - o.bbox_extent.x - self.ego_half_length, 0.05)
            if gap >= best_gap:
                continue
            obj_speed = o.velocity.norm()
            best = (o, gap, ego.speed - obj_speed, obj_speed)
            best_gap = gap
        return best

    def _idm_speed(self, v: float, lead) -> float:
        """IDM target speed — keeps time-headway spacing behind the lead."""
        _, gap, _, lead_speed = lead
        dv = v - lead_speed
        s_star = (self.standstill_gap + v * self.time_headway
                  + v * dv / (2.0 * math.sqrt(self.idm_a * self.idm_b)))
        s = max(gap, 0.3)
        acc = self.idm_a * (1.0 - (v / max(self.cruise_speed, 0.5)) ** 4
                            - (s_star / s) ** 2)
        return min(self.cruise_speed, max(0.0, v + acc))

    def _lane_change_clear(
        self,
        direction: str,
        perception: PerceptionOutput,
        ego: VehicleState,
    ) -> bool:
        """No object inside the target-lane box: lon [-8, 25] m and one
        lane-width band of lateral offset in the requested direction."""
        sign = 1.0 if direction == "LEFT" else -1.0
        lat_lo, lat_hi = 0.4 * self.lane_width, 2.0 * self.lane_width
        for o in perception.objects:
            lon, lat = _to_ego(o.position.x - ego.x,
                               o.position.y - ego.y, ego.yaw)
            if -8.0 < lon < 25.0 and lat_lo < sign * lat < lat_hi:
                return False
        return True
