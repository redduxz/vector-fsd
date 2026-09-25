"""Pluggable safety rules — the checks the SafetyMonitor runs every cycle.

Each rule is deliberately small and total:

* ``evaluate(ctx) -> SafetyEvent | None`` detects a violation and classifies
  its severity ("info" | "warning" | "critical").
* ``enforce(cmd, ctx) -> ControlCommand`` reshapes the demanded command so it
  stays inside the safe envelope (caps, tapering, rate limits). Rules must
  never mutate ``ctx.cmd``; enforcement works on a copy owned by the monitor.

Rules must never raise — but the monitor guards every call anyway, because a
faulting rule in the safety layer is itself a critical event.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from fsd.core.config import SafetyConfig
from fsd.core.types import (
    ControlCommand,
    DriveMode,
    PerceptionOutput,
    SafetyEvent,
    VehicleState,
)

# Severity ordering used by the monitor when arbitrating between rules.
SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

# Nominal actuator gains (fraction -> m/s^2) for a Model-3-class vehicle.
# Used to translate unitless throttle/brake demands into accel estimates.
NOMINAL_MAX_ACCEL_MPS2 = 6.0   # ~full throttle at road speeds
NOMINAL_MAX_BRAKE_MPS2 = 8.0   # ~full brake, dry asphalt

_EGO_HALF_LENGTH_M = 2.4   # front bumper to center, plus small margin
_EGO_HALF_WIDTH_M = 1.0    # mirror-to-center, plus small margin


def _finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def _clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class SafetyContext:
    """Everything a rule is allowed to look at for one evaluation cycle."""

    ego: VehicleState
    perception: Optional[PerceptionOutput]
    cmd: ControlCommand                 # planner demand — never mutate
    applied: Optional[ControlCommand]   # last command the monitor let through
    pipeline_alive: bool                # upstream heartbeat flag from caller
    now: float                          # time.time() at start of check
    dt: float                           # seconds since previous check
    cfg: SafetyConfig
    last_heartbeat: float               # last explicit pipeline heartbeat


class SafetyRule:
    """Base class for a check. Subclasses override evaluate and/or enforce."""

    name: str = "rule"
    enabled: bool = True

    def evaluate(self, ctx: SafetyContext) -> Optional[SafetyEvent]:
        return None

    def enforce(self, cmd: ControlCommand, ctx: SafetyContext) -> ControlCommand:
        return cmd

    def reset(self) -> None:
        """Drop any state carried between cycles (called on monitor reset)."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} enabled={self.enabled}>"


class TTCRule(SafetyRule):
    """Time-to-collision against fused objects in the ego corridor.

    An object counts if its lateral offset fits inside a corridor the width of
    the ego car plus the object's own half-width, and it is ahead of the front
    bumper. TTC = gap / closing-speed, evaluated along the ego heading.
    """

    name = "ttc"
    MIN_CLOSING_MPS = 0.1      # below this the gap is effectively stable
    WARN_FACTOR = 1.8          # warn at 1.8x the critical floor

    def evaluate(self, ctx: SafetyContext) -> Optional[SafetyEvent]:
        perc = ctx.perception
        if perc is None or not perc.objects:
            return None
        ego = ctx.ego
        if not _finite(ego.x, ego.y, ego.yaw, ego.speed):
            return None  # ego-state validity is the monitor's preflight job
        cy, sy = math.cos(ego.yaw), math.sin(ego.yaw)

        best_ttc = math.inf
        best_desc = ""
        for obj in perc.objects:
            px, py = obj.position.x, obj.position.y
            if not _finite(px, py, obj.velocity.x, obj.velocity.y):
                continue
            dx, dy = px - ego.x, py - ego.y
            fwd = dx * cy + dy * sy
            lat = -dx * sy + dy * cy
            if fwd <= 0.0:
                continue  # fully behind the bumper line
            corridor = _EGO_HALF_WIDTH_M + max(0.5, abs(obj.bbox_extent.y))
            if abs(lat) > corridor:
                continue  # outside the swept path
            gap = fwd - _EGO_HALF_LENGTH_M - max(0.0, abs(obj.bbox_extent.x))
            obj_speed_fwd = obj.velocity.x * cy + obj.velocity.y * sy
            closing = ego.speed - obj_speed_fwd
            if gap <= 0.0:
                ttc = 0.0  # already inside the safety envelope
            elif closing > self.MIN_CLOSING_MPS:
                ttc = gap / closing
            else:
                continue  # receding or pacing — no imminent collision
            if ttc < best_ttc:
                best_ttc = ttc
                best_desc = (
                    f"obj#{obj.obj_id} {obj.cls} gap={max(gap, 0.0):.1f}m "
                    f"closing={closing:.1f}m/s conf={obj.confidence:.2f}"
                )

        if best_ttc is math.inf:
            return None
        floor = ctx.cfg.min_ttc_s
        if best_ttc < floor:
            return SafetyEvent(
                "critical", self.name,
                f"TTC {best_ttc:.2f}s below floor {floor:.2f}s ({best_desc})")
        if best_ttc < floor * self.WARN_FACTOR:
            return SafetyEvent(
                "warning", self.name,
                f"TTC {best_ttc:.2f}s approaching floor {floor:.2f}s ({best_desc})")
        return None


class SpeedLimitRule(SafetyRule):
    """Caps vehicle speed and commanded acceleration.

    Evaluation flags overspeed; enforcement zeroes throttle at the cap, adds
    compensating brake when already over, tapers throttle just below the cap
    so the car settles under the limit instead of bouncing off it, and bounds
    throttle/brake fractions by the configured accel/decel budgets.
    """

    name = "speed_limit"
    TAPER_BAND_MPS = 2.0       # start tapering this far below the cap
    HARD_FACTOR = 1.35         # >35% over the cap is critical, not just a warning

    def evaluate(self, ctx: SafetyContext) -> Optional[SafetyEvent]:
        speed = ctx.ego.speed
        limit = ctx.cfg.max_speed_mps
        if not _finite(speed):
            return SafetyEvent("critical", self.name, "ego speed is NaN/inf — telemetry invalid")
        if speed > limit * self.HARD_FACTOR:
            return SafetyEvent(
                "critical", self.name,
                f"speed {speed:.1f}m/s exceeds hard limit {limit * self.HARD_FACTOR:.1f}m/s")
        if speed > limit:
            return SafetyEvent(
                "warning", self.name,
                f"speed {speed:.1f}m/s over limit {limit:.1f}m/s — clamping")
        return None

    def enforce(self, cmd: ControlCommand, ctx: SafetyContext) -> ControlCommand:
        cfg = ctx.cfg
        speed = ctx.ego.speed
        if not _finite(speed):
            cmd.throttle = 0.0
            cmd.brake = 1.0
            return cmd

        # Accel/decel budgets: bound the unitless demand by the nominal gains.
        thr_cap = _clampf(cfg.max_accel_mps2 / NOMINAL_MAX_ACCEL_MPS2, 0.0, 1.0)
        brk_cap = _clampf(cfg.max_brake_mps2 / NOMINAL_MAX_BRAKE_MPS2, 0.0, 1.0)
        cmd.throttle = min(cmd.throttle, thr_cap)
        cmd.brake = min(cmd.brake, brk_cap)

        margin = cfg.max_speed_mps - speed
        if margin <= 0.0:
            # Over the cap: kill throttle and add brake proportional to excess.
            cmd.throttle = 0.0
            over_ratio = -margin / max(cfg.max_speed_mps, 1e-3)
            cmd.brake = max(cmd.brake, _clampf(0.35 + 2.0 * over_ratio, 0.0, 1.0))
        elif margin < self.TAPER_BAND_MPS:
            cmd.throttle *= _clampf(margin / self.TAPER_BAND_MPS, 0.0, 1.0)
        return cmd


class FreeSpaceRule(SafetyRule):
    """Drivable-space check: the path ahead must be measurably clear.

    A perception blackout is represented upstream as free_space_ahead == 0,
    which this rule treats the same as a physically blocked path.
    """

    name = "free_space"
    WARN_FACTOR = 1.75

    def evaluate(self, ctx: SafetyContext) -> Optional[SafetyEvent]:
        perc = ctx.perception
        if perc is None:
            return None  # absence of perception is flagged by the monitor itself
        free = perc.free_space_ahead
        floor = ctx.cfg.min_free_space_m
        if not _finite(free):
            return SafetyEvent(
                "critical", self.name, "free-space reading is NaN/inf — treating as obstructed")
        if free < floor:
            return SafetyEvent(
                "critical", self.name,
                f"free space {free:.1f}m below minimum {floor:.1f}m")
        if free < floor * self.WARN_FACTOR:
            return SafetyEvent(
                "warning", self.name,
                f"free space {free:.1f}m nearing minimum {floor:.1f}m")
        return None


class WatchdogRule(SafetyRule):
    """Liveness: pipeline heartbeat, and staleness of the fused inputs.

    Two independent stall detectors: the explicit ``pipeline_alive`` flag /
    heartbeat timestamp the upstream loop must maintain, and the freshness of
    the perception/ego timestamps themselves. Any of them timing out means the
    stack is flying blind — that is a critical event.
    """

    name = "watchdog"

    def evaluate(self, ctx: SafetyContext) -> Optional[SafetyEvent]:
        timeout = ctx.cfg.watchdog_timeout_s
        if not ctx.pipeline_alive:
            return SafetyEvent(
                "critical", self.name, "pipeline reports not alive — heartbeat lost")
        hb_age = ctx.now - ctx.last_heartbeat
        if hb_age > timeout:
            return SafetyEvent(
                "critical", self.name,
                f"pipeline heartbeat {hb_age:.2f}s old (timeout {timeout:.2f}s)")
        perc = ctx.perception
        if perc is not None and _finite(perc.timestamp):
            age = ctx.now - perc.timestamp
            if age > timeout:
                return SafetyEvent(
                    "critical", self.name,
                    f"perception output {age:.2f}s stale (timeout {timeout:.2f}s)")
        if _finite(ctx.ego.timestamp):
            age = ctx.now - ctx.ego.timestamp
            if age > timeout:
                return SafetyEvent(
                    "critical", self.name,
                    f"ego state {age:.2f}s stale (timeout {timeout:.2f}s)")
        if ctx.dt > timeout * 2.0:
            return SafetyEvent(
                "warning", self.name,
                f"safety cycle overran: {ctx.dt * 1000:.0f}ms between checks")
        return None


class SteerRateRule(SafetyRule):
    """Rate-limits the steering demand.

    The limit is applied against the last *enforced* steer (falling back to
    the reported ego steer), so a planner cannot accumulate its way around the
    cap by ramping demand faster than we let it through.
    """

    name = "steer_rate"

    def _baseline(self, ctx: SafetyContext) -> float:
        if ctx.applied is not None and _finite(ctx.applied.steer):
            return ctx.applied.steer
        if _finite(ctx.ego.steer):
            return ctx.ego.steer
        return 0.0

    def evaluate(self, ctx: SafetyContext) -> Optional[SafetyEvent]:
        if ctx.dt <= 1e-4 or not _finite(ctx.cmd.steer):
            return None
        rate = abs(ctx.cmd.steer - self._baseline(ctx)) / ctx.dt
        hard = ctx.cfg.max_steer_rate * 1.5
        if rate > hard:
            return SafetyEvent(
                "warning", self.name,
                f"steer demand slews {rate:.2f}/s (limit {ctx.cfg.max_steer_rate:.2f}/s) — rate-limiting")
        return None

    def enforce(self, cmd: ControlCommand, ctx: SafetyContext) -> ControlCommand:
        if not _finite(cmd.steer):
            cmd.steer = self._baseline(ctx)
            return cmd
        dt = max(ctx.dt, 1e-3)
        max_delta = ctx.cfg.max_steer_rate * dt
        base = self._baseline(ctx)
        cmd.steer = _clampf(cmd.steer, base - max_delta, base + max_delta)
        return cmd


class LaneDepartureRule(SafetyRule):
    """Lane-keeping guard.

    Fully departing the lane is critical; drifting toward the edge is a
    warning that degrades the drive mode. Loss of lane tracking at speed is
    a warning — the car may still be fine, but confidence is gone.
    """

    name = "lane_departure"
    LOST_MIN_SPEED_MPS = 4.0   # below this, lost lane tracking is not actionable
    EXIT_MARGIN_M = 0.4        # fully out = past the edge plus this margin

    def evaluate(self, ctx: SafetyContext) -> Optional[SafetyEvent]:
        perc = ctx.perception
        if perc is None or perc.lane is None:
            return None
        lane = perc.lane
        if not lane.detected:
            if ctx.ego.speed > self.LOST_MIN_SPEED_MPS:
                return SafetyEvent(
                    "warning", self.name,
                    f"lane tracking lost at {ctx.ego.speed:.1f}m/s")
            return None
        half_w = lane.lane_width / 2.0
        if not _finite(lane.center_offset, half_w) or half_w <= 0.0:
            return None
        offset = abs(lane.center_offset)
        if offset > half_w + self.EXIT_MARGIN_M:
            return SafetyEvent(
                "critical", self.name,
                f"lane departure: |offset| {offset:.2f}m exceeds lane half-width {half_w:.2f}m")
        if offset > 0.6 * half_w:
            return SafetyEvent(
                "warning", self.name,
                f"drifting toward lane edge: |offset| {offset:.2f}m of {half_w:.2f}m half-width")
        return None


def default_rules() -> List[SafetyRule]:
    """The standard rule set, in enforcement order.

    Order matters for enforce(): caps and taper first, steering slew last so
    the final steer value is what leaves the gate.
    """
    return [
        TTCRule(),
        SpeedLimitRule(),
        FreeSpaceRule(),
        WatchdogRule(),
        SteerRateRule(),
        LaneDepartureRule(),
    ]
