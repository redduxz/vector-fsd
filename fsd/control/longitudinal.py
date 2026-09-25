"""Longitudinal control: speed-tracking PID mapped to throttle/brake.

Pipeline per tick:

1. Safety cap on the requested speed — never command a speed from which
   the vehicle could not stop within `free_space_ahead` using 70% of max
   brake (a comfort reserve).
2. TTC guard — if closing on the nearest object ahead faster than the
   minimum time-to-collision allows, zero the target and brake.
3. PID on speed error -> desired acceleration a_des.
4. accel -> actuator split:
       throttle = a_des / a_available(v)      (a_des > 0)
       brake    = -a_des / max_brake          (a_des < 0)
   with a speed-dependent taper a_available(v) = a_max * (1 - v/v_max)
   to model torque falloff at speed. Small |a| -> coasting deadband.

Returns a ControlCommand with only throttle/brake filled (steer = 0).
"""
from __future__ import annotations

import math

from fsd.core.logger import get
from fsd.core.types import (
    ControlCommand,
    PerceptionOutput,
    VehicleState,
)

from fsd.control.pid import PIDController

_log = get("control.longitudinal")


class LongitudinalController:
    """Speed-tracking PID with throttle/brake split and TTC guard."""

    def __init__(
        self,
        kp: float = 1.2,
        ki: float = 0.25,
        kd: float = 0.05,
        max_accel_mps2: float = 3.0,
        max_brake_mps2: float = 6.0,
        v_max_mps: float = 55.0,
        accel_deadband: float = 0.08,
        stop_margin_m: float = 2.0,
        min_ttc_s: float = 1.2,
        brake_hold: float = 0.35,
        ego_half_length_m: float = 2.4,
    ) -> None:
        self.max_accel = float(max_accel_mps2)
        self.max_brake = float(max_brake_mps2)
        self.v_max = float(v_max_mps)
        self.deadband = float(accel_deadband)
        self.stop_margin = float(stop_margin_m)
        self.min_ttc = float(min_ttc_s)
        self.brake_hold = float(brake_hold)
        self.ego_half_length = float(ego_half_length_m)

        self.pid = PIDController(
            kp, ki, kd,
            i_min=-max_brake_mps2 / max(ki, 1e-6),
            i_max=max_accel_mps2 / max(ki, 1e-6),
            out_min=-max_brake_mps2,
            out_max=max_accel_mps2,
            name="longitudinal",
        )
        self._last_ts: float | None = None

    # ------------------------------------------------------------------ #
    def accel_cmd(
        self,
        target_speed: float,
        ego: VehicleState,
        perception: PerceptionOutput | None = None,
    ) -> ControlCommand:
        """Track `target_speed`; return a throttle/brake-only command."""
        dt = self._dt(ego.timestamp)
        v_cmd = max(float(target_speed), 0.0)

        forced_brake = 0.0
        if perception is not None:
            # ---- free-space stopping cap ------------------------------- #
            d_eff = perception.free_space_ahead - self.stop_margin
            v_cap = math.sqrt(max(0.0, 2.0 * 0.7 * self.max_brake * d_eff))
            v_cmd = min(v_cmd, v_cap)

            # ---- TTC guard against the nearest object ahead ------------ #
            ttc, gap, closing = self._ttc(perception, ego)
            if ttc < self.min_ttc:
                v_cmd = 0.0
                # harder brake as TTC shrinks toward zero
                forced_brake = min(1.0, 0.4 + 0.6 * (1.0 - ttc / self.min_ttc))

        # ---- PID on speed error ---------------------------------------- #
        err = v_cmd - ego.speed
        a_des = self.pid.step(err, dt)

        cmd = self.accel_to_command(a_des, ego.speed)
        if forced_brake > 0.0:
            cmd.throttle = 0.0
            cmd.brake = max(cmd.brake, forced_brake)

        # hold a stopped car in place instead of rolling
        if v_cmd < 0.3 and ego.speed < 0.4:
            cmd.throttle = 0.0
            cmd.brake = max(cmd.brake, self.brake_hold)

        return cmd.clamp()

    # ------------------------------------------------------------------ #
    def accel_to_command(self, a_des: float, speed: float) -> ControlCommand:
        """Split a desired acceleration into throttle/brake in [0, 1]."""
        cmd = ControlCommand()
        if a_des > self.deadband:
            # torque taper: available accel falls off as v -> v_max
            taper = max(0.25, 1.0 - max(speed, 0.0) / self.v_max)
            a_avail = self.max_accel * taper
            cmd.throttle = a_des / max(a_avail, 0.1)
        elif a_des < -self.deadband:
            cmd.brake = -a_des / self.max_brake
        # else: deadband -> coast (both 0)
        return cmd.clamp()

    # ------------------------------------------------------------------ #
    def _ttc(self, perception: PerceptionOutput, ego: VehicleState):
        """Time-to-collision with the nearest object in our corridor."""
        cy, sy = math.cos(ego.yaw), math.sin(ego.yaw)
        best_ttc, best_gap, best_closing = math.inf, math.inf, 0.0
        for o in perception.objects:
            dx = o.position.x - ego.x
            dy = o.position.y - ego.y
            lon = dx * cy + dy * sy
            lat = -dx * sy + dy * cy
            if lon <= 0.0 or abs(lat) > 2.0:
                continue
            gap = max(lon - o.bbox_extent.x - self.ego_half_length, 0.05)
            closing = ego.speed - o.velocity.norm()
            ttc = gap / closing if closing > 0.05 else math.inf
            if ttc < best_ttc:
                best_ttc, best_gap, best_closing = ttc, gap, closing
        return best_ttc, best_gap, best_closing

    def _dt(self, ts: float) -> float:
        if self._last_ts is None:
            self._last_ts = ts
            return 0.02
        dt = ts - self._last_ts
        self._last_ts = ts
        return min(max(dt, 1e-3), 0.25)
