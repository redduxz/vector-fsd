"""PID controller with anti-windup and filtered derivative.

Discrete-time form used across the control stack:

    u[k] = Kp*e[k] + Ki * sum(e*dt) + Kd * d(e)/dt

Anti-windup: conditional integration — the integrator freezes whenever the
output is saturated *and* the error would push it deeper into saturation.
The integral term is additionally hard-clamped to [i_min, i_max].

The derivative channel is low-pass filtered (first-order, tau = deriv_tau)
to avoid amplifying measurement noise on every tick.
"""
from __future__ import annotations

import math

from fsd.core.logger import get

_log = get("control.pid")


class PIDController:
    """Scalar PID with clamping, conditional anti-windup and filtered D-term."""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        i_min: float = -1.0,
        i_max: float = 1.0,
        out_min: float = -math.inf,
        out_max: float = math.inf,
        deriv_tau: float = 0.05,
        name: str = "pid",
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_min = float(i_min)
        self.i_max = float(i_max)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.deriv_tau = max(float(deriv_tau), 1e-3)
        self.name = name

        self._integral = 0.0
        self._prev_error: float | None = None
        self._deriv = 0.0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear integrator and derivative history."""
        self._integral = 0.0
        self._prev_error = None
        self._deriv = 0.0

    @property
    def integral(self) -> float:
        return self._integral

    # ------------------------------------------------------------------ #
    def step(self, error: float, dt: float) -> float:
        """Advance one tick. `dt` is seconds since the previous call.

        A non-positive or absurdly large dt is treated as a stalled tick:
        proportional action still applies, integration/derivative are skipped.
        """
        error = float(error)
        dt_valid = 1e-4 < dt < 1.0

        # --- derivative on error, low-pass filtered --------------------- #
        if dt_valid and self._prev_error is not None:
            raw_d = (error - self._prev_error) / dt
            alpha = dt / (dt + self.deriv_tau)      # 1st-order LP coefficient
            self._deriv += alpha * (raw_d - self._deriv)
        elif self._prev_error is None:
            self._deriv = 0.0                       # avoid kick on first call
        self._prev_error = error

        p = self.kp * error
        d = self.kd * self._deriv

        # --- tentative output, then conditional integration ------------- #
        u_unsat = p + self.ki * self._integral + d
        u = min(max(u_unsat, self.out_min), self.out_max)

        if dt_valid:
            saturated_high = u >= self.out_max and error > 0.0
            saturated_low = u <= self.out_min and error < 0.0
            if not (saturated_high or saturated_low):
                self._integral += error * dt
                self._integral = min(max(self._integral, self.i_min), self.i_max)
                # re-evaluate with the new integral
                u_unsat = p + self.ki * self._integral + d
                u = min(max(u_unsat, self.out_min), self.out_max)

        return u

    # convenience: plain function-call syntax
    __call__ = step
