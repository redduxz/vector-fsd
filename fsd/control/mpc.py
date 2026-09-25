"""MPCLite — a lightweight shooting MPC, deliberately honest about scope.

This is NOT a full QP-based MPC. It is a candidate-control (a.k.a.
"brute-force shooting") approximation:

- the control space is discretized into a small grid of (steer, accel)
  pairs, held constant over a ~1.2 s horizon;
- each candidate is rolled forward with the kinematic bicycle model

      x'    = v * cos(yaw)
      y'    = v * sin(yaw)
      yaw'  = v / L * tan(delta)
      v'    = a            (clipped to [0, v_max])

- rollouts are scored on cross-track error to the reference path, heading
  error, speed error, and control effort; a second, finer pass refines
  around the stage-1 winner.

The returned (steer, accel) is the *first* control of the best rollout —
the classic receding-horizon choice. For a few dozen candidates and a
~12-step horizon this is cheap enough for a 20 Hz loop and noticeably
better than raw Stanley+PID around tight curvature, which is all we ask
of it. Real MPPI/SQP solvers can replace `optimize` without changing the
callers.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from fsd.core.logger import get
from fsd.core.types import Trajectory, VehicleState

_log = get("control.mpc")


class MPCLite:
    """Grid-search shooting MPC over the kinematic bicycle model."""

    def __init__(
        self,
        wheelbase_m: float = 2.875,
        max_steer_deg: float = 60.0,
        horizon_s: float = 1.2,
        dt: float = 0.1,
        v_max_mps: float = 16.7,
        max_accel_mps2: float = 3.0,
        max_brake_mps2: float = 6.0,
        w_cte: float = 6.0,
        w_yaw: float = 1.5,
        w_vel: float = 0.4,
        w_effort: float = 0.15,
        w_terminal: float = 2.0,
        refine_passes: int = 1,
    ) -> None:
        self.L = float(wheelbase_m)
        self.max_steer = math.radians(max_steer_deg)
        self.horizon_s = float(horizon_s)
        self.dt = float(dt)
        self.steps = max(int(round(horizon_s / dt)), 1)
        self.v_max = float(v_max_mps)
        self.max_accel = float(max_accel_mps2)
        self.max_brake = float(max_brake_mps2)
        self.w_cte = float(w_cte)
        self.w_yaw = float(w_yaw)
        self.w_vel = float(w_vel)
        self.w_effort = float(w_effort)
        self.w_terminal = float(w_terminal)
        self.refine_passes = int(refine_passes)

    # ------------------------------------------------------------------ #
    def optimize(
        self,
        trajectory: Trajectory,
        ego: VehicleState,
        target_speed: float,
        base_steer_norm: float = 0.0,
        base_accel: float = 0.0,
    ) -> Tuple[float, float]:
        """Return (steer_normalized, accel_mps2) for the current tick."""
        if trajectory.empty or len(trajectory.points) < 2:
            return float(base_steer_norm), float(base_accel)

        # reference arrays
        rx = np.array([p.x for p in trajectory.points])
        ry = np.array([p.y for p in trajectory.points])
        rdx, rdy = np.diff(rx), np.diff(ry)
        rhead = np.arctan2(rdy, rdx)
        rhead = np.append(rhead, rhead[-1])

        base_steer = base_steer_norm * self.max_steer

        # ---- stage 1: coarse absolute grid ----------------------------- #
        steers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * self.max_steer
        accels = np.array([-self.max_brake, -0.5 * self.max_brake,
                           0.0, 0.6 * self.max_accel, self.max_accel])
        best = self._search(rx, ry, rhead, ego, target_speed, steers, accels)

        # ---- stage 2: local refinement around the winner --------------- #
        for _ in range(self.refine_passes):
            s0, a0 = best
            s_fine = s0 + np.array([-0.25, -0.125, 0.0, 0.125, 0.25]) * self.max_steer
            a_fine = a0 + np.array([-0.5, -0.25, 0.0, 0.25, 0.5]) * self.max_accel
            fine = self._search(rx, ry, rhead, ego, target_speed,
                                np.clip(s_fine, -self.max_steer, self.max_steer),
                                np.clip(a_fine, -self.max_brake, self.max_accel))
            if fine is not None:
                best = fine

        steer_norm = float(np.clip(best[0] / self.max_steer, -1.0, 1.0))
        accel = float(np.clip(best[1], -self.max_brake, self.max_accel))
        return steer_norm, accel

    # ------------------------------------------------------------------ #
    def _search(
        self,
        rx: np.ndarray,
        ry: np.ndarray,
        rhead: np.ndarray,
        ego: VehicleState,
        v_ref: float,
        steers: np.ndarray,
        accels: np.ndarray,
    ) -> Tuple[float, float] | None:
        """Roll out every (steer, accel) pair; return the cheapest."""
        best_cost = math.inf
        best: Tuple[float, float] | None = None

        for delta in steers:
            tan_delta_over_l = math.tan(float(delta)) / self.L
            for accel in accels:
                cost = self._rollout_cost(
                    rx, ry, rhead, ego, v_ref,
                    float(delta), tan_delta_over_l, float(accel))
                if cost < best_cost:
                    best_cost = cost
                    best = (float(delta), float(accel))
        return best

    def _rollout_cost(
        self,
        rx: np.ndarray,
        ry: np.ndarray,
        rhead: np.ndarray,
        ego: VehicleState,
        v_ref: float,
        delta: float,
        tan_delta_over_l: float,
        accel: float,
    ) -> float:
        """Simulate one constant-control rollout and score it."""
        x, y, yaw, v = ego.x, ego.y, ego.yaw, max(ego.speed, 0.0)
        cost = 0.0
        dt = self.dt
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        for k in range(self.steps):
            # kinematic bicycle update
            x += v * cos_y * dt
            y += v * sin_y * dt
            yaw += v * tan_delta_over_l * dt
            v = min(max(v + accel * dt, 0.0), self.v_max)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)

            # nearest reference point -> cross-track + heading error
            d2 = (rx - x) ** 2 + (ry - y) ** 2
            i = int(np.argmin(d2))
            th = float(rhead[i])
            cte = (x - rx[i]) * -math.sin(th) + (y - ry[i]) * math.cos(th)
            yaw_err = _wrap(th - yaw)
            vel_err = v - v_ref

            w = self.w_terminal if k == self.steps - 1 else 1.0
            cost += w * (self.w_cte * cte * cte
                         + self.w_yaw * yaw_err * yaw_err
                         + self.w_vel * vel_err * vel_err)

        # control effort, normalized
        cost += self.w_effort * (
            (delta / self.max_steer) ** 2
            + (accel / self.max_brake if accel < 0 else accel / self.max_accel) ** 2
        )
        return cost


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
