"""Scripted scenario catalogue — closed-loop tests on the synthetic world.

Each scenario perturbs the ``ScenarioScene`` over time (lead-vehicle script,
injected actors, light override, fault injection) and defines an explicit
pass/fail oracle over ground truth (footprint overlap, stop-line crossing)
plus stack behaviour (SAFE_STOP, braking, stopping).

No CARLA required — every scenario runs against ``AutopilotAgent(smoke=True)``.
"""
from __future__ import annotations

import math
from typing import Optional

from fsd.core.logger import get
from fsd.core.types import DriveMode, LightState
from fsd.eval.faults import DropSensor, FreezeHeartbeat
from fsd.scenarios.base import (
    Scenario,
    ScenarioEvent,
    ScenarioScene,
    ScriptedActor,
)

log = get("scenarios.library")


# ---------------------------------------------------------------------------
# Shared oracle helpers
# ---------------------------------------------------------------------------

class _EgoResponseMixin:
    """Bookkeeping shared by "ego must react" scenarios."""

    def _reset_obs(self) -> None:
        self._fired_at: Optional[float] = None
        self._collided = False
        self._min_gap = math.inf
        self._saw_safe_stop = False
        self._saw_brake = False
        self._stopped = False

    def _observe_response(self) -> None:
        """Call each tick after firing: watch how the stack reacts."""
        scene: ScenarioScene = self.scene  # type: ignore[assignment]
        if scene.mode == DriveMode.SAFE_STOP:
            self._saw_safe_stop = True
        try:
            if scene.ego is not None and scene.ego._cmd.brake > 0.2:
                self._saw_brake = True
        except AttributeError:
            pass
        if self._fired_at is not None and scene.ego_state().speed < 0.3:
            self._stopped = True

    @property
    def _responded(self) -> bool:
        return self._saw_safe_stop or self._saw_brake or self._stopped


# ---------------------------------------------------------------------------
# 1. LeadVehicleCutIn
# ---------------------------------------------------------------------------

class LeadVehicleCutIn(_EgoResponseMixin, Scenario):
    """A lead vehicle merges into the ego lane at short range.

    Script: a lead cruises far ahead (150 m) until ``trigger_t``; it then
    cuts in ``cut_gap_m`` ahead travelling ``cut_speed_mps``. TTC collapses
    below the safety floor almost immediately — the monitor is expected to
    veto to SAFE_STOP (or the planner to emergency-brake) before contact.

    PASS: bumper gap never goes negative AND the stack visibly responded
    (SAFE_STOP, hard brake, or a full stop before the obstacle).
    FAIL: collision, or the cut-in triggered and no response followed.
    """

    name = "lead_vehicle_cut_in"
    description = ("Lead vehicle cuts into the ego lane at ~12 m while ego "
                   "cruises; expect emergency brake / SAFE_STOP, no contact.")
    timeout_s = 18.0

    def __init__(self, trigger_t: float = 5.0, cut_gap_m: float = 12.0,
                 cut_speed_mps: float = 4.0) -> None:
        super().__init__()
        self.trigger_t = float(trigger_t)
        self.cut_gap_m = float(cut_gap_m)
        self.cut_speed_mps = float(cut_speed_mps)
        self._reset_obs()

    def setup(self, scene: ScenarioScene) -> None:
        super().setup(scene)
        scene.synthetic_lane(True)
        scene.set_lead(dist=150.0, speed=15.0)   # cruising far ahead

    def tick(self, t: float) -> Optional[ScenarioEvent]:
        scene = self.scene
        self._observe_response()
        if self._fired_at is None and t >= self.trigger_t:
            v = scene.ego_state().speed
            scene.set_lead(dist=self.cut_gap_m, speed=self.cut_speed_mps)
            self._fired_at = t
            return ScenarioEvent(
                t, "cut_in",
                f"lead merges {self.cut_gap_m:.0f} m ahead at "
                f"{self.cut_speed_mps:.1f} m/s (ego {v:.1f} m/s)")
        gap = scene.lead_bumper_gap()
        self._min_gap = min(self._min_gap, gap)
        if gap <= 0.0 and not self._collided:
            self._collided = True
            return ScenarioEvent(t, "collision",
                                 "bumper overlap with cut-in lead")
        return None

    def done(self, t: float) -> bool:
        if self._collided:
            return True
        if self._fired_at is None:
            return False
        # decided once the car has stood still after the cut-in, or the
        # post-event window has fully elapsed
        return self._stopped or (t - self._fired_at) > 6.0

    def evaluate(self, metrics) -> Optional[bool]:
        if self._collided:
            return False
        if self._fired_at is None:
            return None                      # trigger never reached
        return bool(self._min_gap > 0.0 and self._responded)


# ---------------------------------------------------------------------------
# 2. SuddenBraking
# ---------------------------------------------------------------------------

class SuddenBraking(_EgoResponseMixin, Scenario):
    """The lead vehicle slams its brakes ahead of the cruising ego.

    Script: nominal lead (45 m, 6.5 m/s) until ``trigger_t``; then the lead
    decelerates at ``lead_decel`` to a standstill — the classic "traffic
    ahead stops short" emergency.

    PASS: no bumper contact AND ego comes to a stop (or SAFE_STOP) behind
    the lead. FAIL: collision or no deceleration response.
    """

    name = "sudden_braking"
    description = ("Lead vehicle brakes to zero at ~6 m/s^2 in front of "
                   "cruising ego; expect controlled emergency stop.")
    timeout_s = 22.0

    def __init__(self, trigger_t: float = 6.0, lead_decel_mps2: float = 6.0,
                 min_ego_speed: float = 6.0) -> None:
        super().__init__()
        self.trigger_t = float(trigger_t)
        self.lead_decel = float(lead_decel_mps2)
        self.min_ego_speed = float(min_ego_speed)
        self._reset_obs()
        self._lead_stopped = False

    def setup(self, scene: ScenarioScene) -> None:
        super().setup(scene)
        scene.synthetic_lane(True)
        scene.set_lead(dist=45.0, speed=6.5)     # stock smoke-scene lead

    def tick(self, t: float) -> Optional[ScenarioEvent]:
        scene = self.scene
        self._observe_response()
        ev = None
        if self._fired_at is None:
            if (t >= self.trigger_t
                    and scene.ego_state().speed >= self.min_ego_speed):
                self._fired_at = t
                ev = ScenarioEvent(
                    t, "brake",
                    f"lead brakes at {self.lead_decel:.1f} m/s^2 "
                    f"(gap {scene.lead_bumper_gap():.1f} m)")
        elif scene.world is not None and scene.world.lead_speed > 0.0:
            scene.world.lead_speed = max(
                0.0, scene.world.lead_speed - self.lead_decel * scene.dt)
            if scene.world.lead_speed == 0.0 and not self._lead_stopped:
                self._lead_stopped = True
                ev = ScenarioEvent(t, "note", "lead at standstill")
        gap = scene.lead_bumper_gap()
        self._min_gap = min(self._min_gap, gap)
        if gap <= 0.0 and not self._collided:
            self._collided = True
            ev = ScenarioEvent(t, "collision",
                               "bumper overlap with braking lead")
        return ev

    def done(self, t: float) -> bool:
        if self._collided:
            return True
        if self._fired_at is None:
            return False
        return (self._lead_stopped and self._stopped) \
            or (t - self._fired_at) > 8.0

    def evaluate(self, metrics) -> Optional[bool]:
        if self._collided:
            return False
        if self._fired_at is None:
            return None
        return bool(self._min_gap > 0.0
                    and (self._stopped or self._saw_safe_stop))


# ---------------------------------------------------------------------------
# 3. JaywalkerPedestrian
# ---------------------------------------------------------------------------

class JaywalkerPedestrian(_EgoResponseMixin, Scenario):
    """A pedestrian steps off the kerb and crosses mid-block.

    Script: at ``spawn_t`` a pedestrian appears ``ahead_m`` down the road on
    the right shoulder (+lateral = left convention: spawned at
    ``-lateral_start``) and walks across at ``walk_speed``. It enters the
    ego corridor late — exactly like a real jaywalker — so TTC goes critical
    with little margin.

    PASS: no footprint overlap AND ego yielded (brake / SAFE_STOP / crawl)
    while the pedestrian was in the path. FAIL: contact, or blowing through
    the crossing with no response.
    """

    name = "jaywalker_pedestrian"
    description = ("Pedestrian crosses the lane mid-block at ~1 m/s; "
                   "expect yield/emergency stop, no contact.")
    timeout_s = 18.0

    def __init__(self, spawn_t: float = 3.0, ahead_m: float = 34.0,
                 lateral_start_m: float = -2.8, walk_speed_mps: float = 0.9
                 ) -> None:
        super().__init__()
        self.spawn_t = float(spawn_t)
        self.ahead_m = float(ahead_m)
        self.lateral_start_m = float(lateral_start_m)
        self.walk_speed = float(walk_speed_mps)
        self._reset_obs()
        self._ped = None
        self._ped_cleared = False
        self._yielded = False

    def setup(self, scene: ScenarioScene) -> None:
        super().setup(scene)
        scene.synthetic_lane(True)
        scene.hide_lead(True)                # clean road — no default lead

    def tick(self, t: float) -> Optional[ScenarioEvent]:
        scene = self.scene
        ev = None
        if self._ped is None and t >= self.spawn_t:
            heading = ("cross_left" if self.lateral_start_m < 0
                       else "cross_right")
            self._ped = scene.spawn(
                cls="pedestrian", ahead_m=self.ahead_m,
                lateral_m=self.lateral_start_m,
                speed_mps=self.walk_speed, heading=heading,
                confidence=0.88)
            ev = ScenarioEvent(
                t, "inject",
                f"pedestrian steps off at {self.ahead_m:.0f} m ahead, "
                f"crossing at {self.walk_speed:.1f} m/s")
        if self._ped is not None:
            self._ped.step(scene.dt)
            ego = scene.ego_state()
            lon, lat = self._ped.ego_frame(ego)
            if self._ped.overlaps_ego(ego) and not self._collided:
                self._collided = True
                return ScenarioEvent(t, "collision",
                                     "pedestrian footprint overlap")
            # yield bookkeeping: while the ped is ahead in/near our path,
            # did the stack shed speed?
            if 0.0 < lon < 40.0 and abs(lat) < 2.0:
                if (scene.mode != DriveMode.ENGAGED
                        or ego.speed < 2.5 or self._saw_brake):
                    self._yielded = True
            self._observe_response()
            if abs(lat) > 3.0 and not self._ped_cleared:
                self._ped_cleared = True
                scene.despawn(self._ped.obj_id)
                ev = ev or ScenarioEvent(t, "clear", "pedestrian cleared "
                                         "the roadway")
        return ev

    def done(self, t: float) -> bool:
        if self._collided:
            return True
        if self._ped is None:
            return False
        return (self._ped_cleared and (self._stopped or self._saw_safe_stop)) \
            or t - self.spawn_t > 8.0

    def evaluate(self, metrics) -> Optional[bool]:
        if self._collided:
            return False
        if self._ped is None:
            return None
        return bool(self._yielded or self._responded)


# ---------------------------------------------------------------------------
# 4. RedLightRunner
# ---------------------------------------------------------------------------

class RedLightRunner(_EgoResponseMixin, Scenario):
    """Signalised intersection: the light turns red and another vehicle
    runs the crossing traffic through the box.

    Script: a virtual stop bar sits ``stop_distance_m`` ahead (implemented as
    a free-space floor while the light constrains the ego — the perception
    equivalent of "stop line here"). Light goes GREEN → YELLOW → RED on a
    fixed schedule; at ``runner_t`` a cross-traffic vehicle enters the
    intersection from the right at ``runner_speed``.

    PASS: ego stops before the stop line and never crosses it while red,
    and no footprint overlap with the runner occurs.
    FAIL: collision, or crossing the stop line at speed on red.
    """

    name = "red_light_runner"
    description = ("Light goes yellow->red ahead; a cross-traffic vehicle "
                   "runs the intersection. Expect a stop at the line, no "
                   "red-light violation, no contact.")
    timeout_s = 20.0

    def __init__(self, stop_distance_m: float = 55.0,
                 yellow_t: float = 2.5, red_t: float = 3.5,
                 runner_t: float = 6.0, runner_speed_mps: float = 8.0
                 ) -> None:
        super().__init__()
        self.stop_distance_m = float(stop_distance_m)
        self.yellow_t = float(yellow_t)
        self.red_t = float(red_t)
        self.runner_t = float(runner_t)
        self.runner_speed = float(runner_speed_mps)
        self._reset_obs()
        self._light = LightState.GREEN
        self._stop_x = None
        self._runner = None
        self._ran_red = False
        self._stopped_at_line = False

    def setup(self, scene: ScenarioScene) -> None:
        super().setup(scene)
        scene.synthetic_lane(True)
        scene.hide_lead(True)
        self._stop_x = scene.ego_state().x + self.stop_distance_m
        scene.set_light(LightState.GREEN)
        scene.set_free_space_floor(self._stop_line_floor)

    # -- the virtual stop bar -------------------------------------------------
    def _stop_line_floor(self, ego) -> Optional[float]:
        """Distance to the stop bar while the light constrains us."""
        if self._light not in (LightState.YELLOW, LightState.RED):
            return None
        cy = math.cos(ego.yaw)
        if abs(cy) < 1e-3:
            return None
        d = (self._stop_x - ego.x) / max(cy, 1e-3)
        return max(d, 0.0) if d > -1.0 else None

    def tick(self, t: float) -> Optional[ScenarioEvent]:
        scene = self.scene
        ev = None

        # ---- light schedule -------------------------------------------- #
        new_light = self._light
        if t >= self.red_t:
            new_light = LightState.RED
        elif t >= self.yellow_t:
            new_light = LightState.YELLOW
        if new_light != self._light:
            self._light = new_light
            scene.set_light(new_light)
            ev = ScenarioEvent(t, "trigger", f"light -> {new_light.name}")

        # ---- the runner ------------------------------------------------- #
        if self._runner is None and t >= self.runner_t:
            ego = scene.ego_state()
            # spawn crossing from the right shoulder at intersection centre
            self._runner = scene.spawn(
                cls="vehicle", ahead_m=(self._stop_x + 6.0 - ego.x),
                lateral_m=-12.0, speed_mps=self.runner_speed,
                heading="cross_left", confidence=0.95)
            ev = ev or ScenarioEvent(
                t, "inject", "cross-traffic vehicle enters intersection")
        if isinstance(self._runner, ScriptedActor):
            self._runner.step(scene.dt)
            ego = scene.ego_state()
            if self._runner.overlaps_ego(ego) and not self._collided:
                self._collided = True
                return ScenarioEvent(t, "collision",
                                     "overlap with red-light runner")
            if abs(self._runner.ego_frame(ego)[1]) > 14.0:
                scene.despawn(self._runner.obj_id)
                self._runner = "cleared"

        # ---- red-light violation bookkeeping ---------------------------- #
        ego = scene.ego_state()
        if (self._light == LightState.RED and self._stop_x is not None
                and ego.x > self._stop_x + 0.5 and ego.speed > 0.5):
            self._ran_red = True
            return ScenarioEvent(
                t, "note", "VIOLATION: ego crossed stop line on red")
        if (ego.x < self._stop_x and ego.speed < 0.3
                and self._light == LightState.RED):
            self._stopped_at_line = True
        self._observe_response()
        return ev

    def done(self, t: float) -> bool:
        if self._collided or self._ran_red:
            return True
        # decided once the runner has passed and the outcome is stable
        if self._runner == "cleared" and (
                self._stopped_at_line or self._saw_safe_stop
                or self._ego().speed < 0.3):
            return True
        return t > self.runner_t + 8.0

    def evaluate(self, metrics) -> Optional[bool]:
        if self._collided or self._ran_red:
            return False
        if self._runner is None:
            return None                      # never even reached the event
        return bool(self._stopped_at_line or self._saw_safe_stop
                    or self._stopped)


# ---------------------------------------------------------------------------
# 5. SensorDropout
# ---------------------------------------------------------------------------

class SensorDropout(Scenario):
    """The object-detection feed drops mid-drive (perception staleness).

    Script: ``DropSensor('objects')`` is injected for
    ``[dropout_t, dropout_t + dropout_s]``. The agent must degrade onto
    last-good perception, the watchdog must flag the stale output, and the
    monitor must veto to SAFE_STOP — a car that keeps cruising blind fails.

    PASS: SAFE_STOP engaged within ``respond_within_s`` of the drop and ego
    brought to a stop. FAIL: the stack kept driving blind / never degraded.
    """

    name = "sensor_dropout"
    description = ("Object-detection feed drops for a window; watchdog must "
                   "flag stale perception and veto to SAFE_STOP.")
    timeout_s = 16.0

    def __init__(self, dropout_t: float = 4.0, dropout_s: float = 4.0,
                 respond_within_s: float = 3.0) -> None:
        super().__init__()
        self.dropout_t = float(dropout_t)
        self.dropout_s = float(dropout_s)
        self.respond_within_s = float(respond_within_s)
        self._fault = DropSensor("objects")
        self._cleared = False
        self._safe_at: Optional[float] = None
        self._stopped = False
        self._v_start = 0.0

    def setup(self, scene: ScenarioScene) -> None:
        super().setup(scene)
        scene.synthetic_lane(True)
        scene.set_lead(dist=60.0, speed=10.0)

    def tick(self, t: float) -> Optional[ScenarioEvent]:
        scene = self.scene
        ev = None
        if not self._fault.active and t >= self.dropout_t:
            scene.inject(self._fault)
            self._v_start = scene.ego_state().speed
            ev = ScenarioEvent(t, "inject",
                               f"objects feed dropped at ego "
                               f"{self._v_start:.1f} m/s")
        if self._fault.active and t >= self.dropout_t + self.dropout_s:
            self._fault.clear()
            scene._faults.remove(self._fault)
            self._cleared = True
            ev = ev or ScenarioEvent(t, "clear", "sensor feed restored "
                                     "(monitor latch expected to hold)")
        if self._safe_at is None and scene.mode == DriveMode.SAFE_STOP:
            self._safe_at = t
            ev = ev or ScenarioEvent(t, "note", "SAFE_STOP engaged")
        if scene.ego_state().speed < 0.3:
            self._stopped = True
        return ev

    def done(self, t: float) -> bool:
        if self._safe_at is not None and self._stopped:
            return True
        # give the stack the whole response window, then decide
        return t > self.dropout_t + self.respond_within_s + 3.0

    def evaluate(self, metrics) -> Optional[bool]:
        if self._safe_at is None:
            return False                     # kept driving blind
        timely = self._safe_at <= self.dropout_t + self.respond_within_s
        return bool(timely and self._stopped)


# ---------------------------------------------------------------------------
# 6. PlannerStall
# ---------------------------------------------------------------------------

class PlannerStall(Scenario):
    """The planning stage stalls — pipeline heartbeat loss.

    Script: at ``stall_t``, ``FreezeHeartbeat('planning')`` makes the
    trajectory stage raise ``HeartbeatLostError`` every call. The agent must
    command a safe stop on the failed tick and the watchdog must confirm the
    pipeline is not alive; the monitor then latches SAFE_STOP.

    PASS: mode reaches SAFE_STOP and the vehicle halts (planning failures
    recorded). FAIL: the car kept driving on a dead planner.
    """

    name = "planner_stall"
    description = ("Trajectory stage heartbeat lost mid-drive; expect "
                   "immediate safe-stop command + watchdog confirmation.")
    timeout_s = 14.0

    def __init__(self, stall_t: float = 3.0) -> None:
        super().__init__()
        self.stall_t = float(stall_t)
        self._fault = FreezeHeartbeat("planning")
        self._safe_at: Optional[float] = None
        self._stopped = False

    def setup(self, scene: ScenarioScene) -> None:
        super().setup(scene)
        scene.synthetic_lane(True)
        scene.set_lead(dist=80.0, speed=12.0)

    def tick(self, t: float) -> Optional[ScenarioEvent]:
        scene = self.scene
        ev = None
        if not self._fault.active and t >= self.stall_t:
            scene.inject(self._fault)
            ev = ScenarioEvent(t, "inject",
                               "planner heartbeat lost — stage stalled")
        if self._safe_at is None and scene.mode == DriveMode.SAFE_STOP:
            self._safe_at = t
            ev = ev or ScenarioEvent(t, "note", "SAFE_STOP engaged")
        if scene.ego_state().speed < 0.3:
            self._stopped = True
        return ev

    def done(self, t: float) -> bool:
        if self._safe_at is not None and self._stopped:
            return True
        return t > self.stall_t + 5.0

    def evaluate(self, metrics) -> Optional[bool]:
        plan_fails = getattr(metrics, "planning_failures", 0)
        if self._safe_at is None:
            return False
        return bool(self._stopped and plan_fails > 0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS = {
    s.name: s for s in (
        LeadVehicleCutIn,
        SuddenBraking,
        JaywalkerPedestrian,
        RedLightRunner,
        SensorDropout,
        PlannerStall,
    )
}


def build(name: str, **kwargs) -> Scenario:
    """Instantiate a registered scenario by name."""
    try:
        return SCENARIOS[name](**kwargs)
    except KeyError:
        raise KeyError(
            f"unknown scenario {name!r}; registered: {sorted(SCENARIOS)}")


__all__ = [
    "LeadVehicleCutIn",
    "SuddenBraking",
    "JaywalkerPedestrian",
    "RedLightRunner",
    "SensorDropout",
    "PlannerStall",
    "SCENARIOS",
    "build",
]
