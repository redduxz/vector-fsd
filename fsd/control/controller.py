"""VehicleController — facade combining lateral + longitudinal (+ MPC-lite).

Per tick:

    steer_base  = Stanley(trajectory, ego)
    accel_base  = Longitudinal PID output (throttle/brake already split)
    (steer_mpc, accel_mpc) = MPCLite refinement, blended in at `mpc_blend`

The MPC blend is deliberately modest: the analytic controllers remain
primary, the shooting optimization nudges steer/accel where curvature or
speed tracking demand it — a poor man's feed-forward. If the trajectory
is empty we command a firm brake while decaying steer to straight.

Steer is rate-limited (normalized units/s) so the safety envelope's
max_steer_rate is honored even when the planners jump the reference path.
"""
from __future__ import annotations

import math

from fsd.core.logger import get
from fsd.core.types import (
    ControlCommand,
    PerceptionOutput,
    Trajectory,
    VehicleState,
)

from fsd.control.lateral import LateralController
from fsd.control.longitudinal import LongitudinalController
from fsd.control.mpc import MPCLite

_log = get("control.controller")


class VehicleController:
    """Top-level controller: Trajectory + state + perception -> command."""

    def __init__(
        self,
        wheelbase_m: float = 2.875,
        max_steer_deg: float = 60.0,
        max_steer_rate: float = 0.4,
        max_accel_mps2: float = 3.0,
        max_brake_mps2: float = 6.0,
        lookahead_m: float = 8.0,
        use_mpc: bool = True,
        mpc_blend: float = 0.35,
        mpc_min_speed: float = 1.0,
    ) -> None:
        self.lateral = LateralController(
            wheelbase_m=wheelbase_m, max_steer_deg=max_steer_deg)
        self.longitudinal = LongitudinalController(
            max_accel_mps2=max_accel_mps2, max_brake_mps2=max_brake_mps2)
        self.mpc = MPCLite(
            wheelbase_m=wheelbase_m, max_steer_deg=max_steer_deg,
            max_accel_mps2=max_accel_mps2, max_brake_mps2=max_brake_mps2)

        self.lookahead_m = float(lookahead_m)
        self.max_steer_rate = float(max_steer_rate)
        self.use_mpc = bool(use_mpc)
        self.mpc_blend = min(max(float(mpc_blend), 0.0), 1.0)
        self.mpc_min_speed = float(mpc_min_speed)
        self.max_accel = float(max_accel_mps2)
        self.max_brake = float(max_brake_mps2)

        self._prev_steer = 0.0
        self._prev_ts: float | None = None

    # ------------------------------------------------------------------ #
    def compute(
        self,
        trajectory: Trajectory,
        ego: VehicleState,
        perception: PerceptionOutput | None = None,
    ) -> ControlCommand:
        """Produce the clamped actuation command for this tick."""
        dt = self._dt(ego.timestamp)

        # Degenerate input -> controlled stop, unwind the wheel gently.
        if trajectory is None or trajectory.empty:
            _log.warning("empty trajectory — commanding stop")
            steer = self._rate_limit(0.0, dt)
            return ControlCommand(
                throttle=0.0, brake=0.8, steer=steer).clamp()

        # ---- lateral ---------------------------------------------------- #
        steer = self.lateral.steer(trajectory, ego, self.lookahead_m)

        # ---- longitudinal ------------------------------------------------ #
        target = max(trajectory.target_speed, 0.0)
        long_cmd = self.longitudinal.accel_cmd(target, ego, perception)
        a_base = (long_cmd.throttle * self.max_accel
                  - long_cmd.brake * self.max_brake)

        # ---- MPC-lite refinement ---------------------------------------- #
        if self.use_mpc and ego.speed >= self.mpc_min_speed:
            steer_mpc, a_mpc = self.mpc.optimize(
                trajectory, ego, target,
                base_steer_norm=steer, base_accel=a_base)
            w = self.mpc_blend
            steer = (1.0 - w) * steer + w * steer_mpc
            a_ref = (1.0 - w) * a_base + w * a_mpc
            blended = self.longitudinal.accel_to_command(a_ref, ego.speed)
            # Never let the optimizer weaken a safety brake, and never let
            # it add throttle while the PID is braking or holding a stop.
            blended.brake = max(blended.brake, long_cmd.brake)
            if long_cmd.brake > 0.0 or target < 0.3:
                blended.throttle = 0.0
            else:
                blended.throttle = min(blended.throttle,
                                       max(long_cmd.throttle * 1.5, 0.2))
            long_cmd = blended

        steer = self._rate_limit(steer, dt)

        return ControlCommand(
            throttle=long_cmd.throttle,
            brake=long_cmd.brake,
            steer=steer,
        ).clamp()

    # friendly alias for tick-style callers
    step = compute

    # ------------------------------------------------------------------ #
    def _rate_limit(self, steer: float, dt: float) -> float:
        max_d = self.max_steer_rate * dt
        limited = min(max(steer, self._prev_steer - max_d),
                      self._prev_steer + max_d)
        self._prev_steer = float(min(max(limited, -1.0), 1.0))
        return self._prev_steer

    def _dt(self, ts: float) -> float:
        if self._prev_ts is None:
            self._prev_ts = ts
            return 0.02
        dt = ts - self._prev_ts
        self._prev_ts = ts
        return min(max(dt, 1e-3), 0.25)
