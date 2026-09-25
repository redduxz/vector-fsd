"""FallbackController — the minimal-risk maneuver executed on SAFE_STOP.

When the monitor escalates, somebody still has to drive the car to a stop.
This controller produces a deterministic deceleration profile:

* braking follows a fraction of ``max_brake_mps2`` scaled to speed — firm at
  speed, easing off near the crawl so the car doesn't jolt or lock wheels;
* steering holds the heading captured at ``engage()`` via a simple
  proportional yaw-error correction — the car stops in the lane it was in;
* if the car is still moving after ``stop_timeout_s``, it escalates to the
  hand brake (a stopped car slightly askew beats a rolling one);
* once stopped, the hand brake stays on — the maneuver ends parked.

It is intentionally self-contained: no planner, no perception. The only
trusted input is the ego state.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

from fsd.core.logger import get
from fsd.core.types import ControlCommand, SafetyEvent, VehicleState
from fsd.safety.monitor import _resolve_safety_cfg
from fsd.safety.rules import NOMINAL_MAX_BRAKE_MPS2


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class FallbackController:
    """Controlled stop while holding the heading captured at engage time."""

    STOPPED_MPS = 0.15         # below this the maneuver is complete
    ESCALATION_MPS = 0.4       # still faster than this after timeout -> hand brake
    STEER_HOLD_LIMIT = 0.5     # never command a bigger correction than this
    CREEP_BRAKE = 0.35         # floor on brake demand while still moving

    def __init__(
        self,
        cfg=None,
        stop_timeout_s: float = 6.0,
        yaw_gain: float = 1.4,
    ) -> None:
        self.cfg = _resolve_safety_cfg(cfg)
        self.stop_timeout_s = float(stop_timeout_s)
        self.yaw_gain = float(yaw_gain)
        self.log = get("safety.fallback")

        self._engaged = False
        self._done = False
        self._escalated = False
        self._t0: Optional[float] = None
        self._hold_yaw = 0.0
        self._events: List[SafetyEvent] = []

    # ------------------------------------------------------------------ API

    def engage(self, ego: Optional[VehicleState], now: Optional[float] = None) -> None:
        """Start (or restart) the maneuver. Captures the current heading."""
        self._t0 = time.time() if now is None else now
        self._hold_yaw = ego.yaw if (ego is not None and math.isfinite(ego.yaw)) else 0.0
        self._engaged = True
        self._done = False
        self._escalated = False
        self._emit("warning", "fallback",
                   f"minimal-risk maneuver engaged (hold yaw {math.degrees(self._hold_yaw):.1f}deg)")
        self.log.warning("fallback engaged — controlled stop, holding heading")

    def update(self, ego: Optional[VehicleState], now: Optional[float] = None) -> ControlCommand:
        """One step of the maneuver -> the command to send this tick."""
        now = time.time() if now is None else now
        if not self._engaged:
            self.engage(ego, now)  # update implies intent; never idle

        speed = 0.0
        yaw = self._hold_yaw
        if ego is not None:
            if math.isfinite(ego.speed):
                speed = max(0.0, ego.speed)
            if math.isfinite(ego.yaw):
                yaw = ego.yaw

        elapsed = now - (self._t0 or now)

        # --- longitudinal: proportional decel, easing toward the crawl ---
        target_decel = min(0.7 * self.cfg.max_brake_mps2,
                           max(0.8, 1.2 * speed))
        brake = _clampf(target_decel / NOMINAL_MAX_BRAKE_MPS2, 0.0, 1.0)
        if speed < 0.5:
            brake = 1.0
        elif brake < self.CREEP_BRAKE:
            brake = self.CREEP_BRAKE

        # --- lateral: hold captured heading with a bounded correction ---
        yaw_err = _wrap_pi(self._hold_yaw - yaw)
        steer = _clampf(self.yaw_gain * yaw_err,
                        -self.STEER_HOLD_LIMIT, self.STEER_HOLD_LIMIT)

        # --- escalation: timeout reached but still rolling -> hand brake ---
        hand_brake = False
        if elapsed > self.stop_timeout_s and speed > self.ESCALATION_MPS:
            hand_brake = True
            brake = 1.0
            if not self._escalated:
                self._escalated = True
                self._emit("critical", "fallback",
                           f"still moving at {speed:.1f}m/s after {elapsed:.1f}s "
                           f"(timeout {self.stop_timeout_s:.1f}s) — hand brake")
                self.log.error("fallback escalating to hand brake at %.1fm/s", speed)

        # --- terminal state: fully stopped -> stay parked ---
        if speed < self.STOPPED_MPS:
            brake = 1.0
            hand_brake = True
            steer = _clampf(steer, -0.2, 0.2)
            if not self._done:
                self._done = True
                self._emit("info", "fallback",
                           f"vehicle stopped after {elapsed:.1f}s — holding with hand brake")
                self.log.info("fallback complete — vehicle stopped")

        return ControlCommand(
            throttle=0.0, brake=brake, steer=steer,
            hand_brake=hand_brake, reverse=False,
        ).clamp()

    def reset(self) -> None:
        self._engaged = False
        self._done = False
        self._escalated = False
        self._t0 = None

    # -------------------------------------------------------------- introspect

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def done(self) -> bool:
        return self._done

    @property
    def escalated(self) -> bool:
        return self._escalated

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._t0 is None else time.time() - self._t0

    @property
    def events(self) -> List[SafetyEvent]:
        return list(self._events)

    # -------------------------------------------------------------- internals

    def _emit(self, level: str, source: str, message: str) -> None:
        self._events.append(SafetyEvent(level, source, message))
