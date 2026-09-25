"""Lateral control: Stanley path-tracking with a pure-pursuit fallback.

Stanley control law (Stanford, 2005):

    delta = psi - atan2(k * e_fa, v + k_soft)

where
    psi   = heading error, wrap(theta_path - theta_ego)
    e_fa  = signed cross-track error of the *front axle* to the nearest
            path point (positive when the vehicle is left of the path),
    k     = Stanley gain,
    k_soft= softening constant that keeps the atan() well-behaved at
            low speed.

Sign convention: positive steer = left. If the vehicle is left of the
path (e_fa > 0) the atan term commands a right turn — hence the minus.

Pure-pursuit fallback (used at crawl speeds where the Stanley cross-track
term saturates and heading estimates are noisy):

    delta = atan2(2 * L * sin(alpha), l_d)

with alpha the bearing to the lookahead point in the ego frame.
Both return a normalized steer in [-1, 1] after dividing by max_steer.
"""
from __future__ import annotations

import math

import numpy as np

from fsd.core.logger import get
from fsd.core.types import Trajectory, VehicleState

_log = get("control.lateral")


def _wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class LateralController:
    """Stanley controller with pure-pursuit fallback for low speed."""

    def __init__(
        self,
        wheelbase_m: float = 2.875,
        max_steer_deg: float = 60.0,
        k_stanley: float = 2.5,
        k_soft_mps: float = 1.2,
        low_speed_mps: float = 0.6,
        lookahead_gain: float = 0.35,
    ) -> None:
        self.wheelbase = float(wheelbase_m)
        self.max_steer_rad = math.radians(max_steer_deg)
        self.k_stanley = float(k_stanley)
        self.k_soft = float(k_soft_mps)
        self.low_speed = float(low_speed_mps)
        self.lookahead_gain = float(lookahead_gain)

    # ------------------------------------------------------------------ #
    def steer(
        self,
        trajectory: Trajectory,
        ego: VehicleState,
        lookahead_m: float = 8.0,
    ) -> float:
        """Return normalized steer in [-1, 1]. Positive = left."""
        pts = trajectory.points
        if trajectory.empty or len(pts) < 2:
            return 0.0

        px = np.array([p.x for p in pts], dtype=float)
        py = np.array([p.y for p in pts], dtype=float)
        # Path heading from successive positions — more robust than trusting
        # wp.yaw, which interpolators occasionally leave at 0.
        dx = np.diff(px)
        dy = np.diff(py)
        headings = np.arctan2(dy, dx)
        headings = np.append(headings, headings[-1])

        # Front-axle position (Stanley is defined w.r.t. the front axle).
        fx = ego.x + 0.5 * self.wheelbase * math.cos(ego.yaw)
        fy = ego.y + 0.5 * self.wheelbase * math.sin(ego.yaw)

        i_near = int(np.argmin((px - fx) ** 2 + (py - fy) ** 2))

        if ego.speed < self.low_speed:
            delta = self._pure_pursuit(px, py, ego, lookahead_m)
        else:
            theta_p = float(headings[i_near])
            psi = _wrap_angle(theta_p - ego.yaw)
            # signed cross-track error: + means vehicle left of the path
            nx, ny = -math.sin(theta_p), math.cos(theta_p)   # left normal
            e_fa = (fx - px[i_near]) * nx + (fy - py[i_near]) * ny
            delta = psi - math.atan2(
                self.k_stanley * e_fa, ego.speed + self.k_soft)

        return float(min(max(delta / self.max_steer_rad, -1.0), 1.0))

    # ------------------------------------------------------------------ #
    def _pure_pursuit(
        self,
        px: np.ndarray,
        py: np.ndarray,
        ego: VehicleState,
        lookahead_m: float,
    ) -> float:
        """Geometric pure pursuit. Returns steer angle in radians."""
        # Effective lookahead grows mildly with speed to damp oscillation.
        ld = max(lookahead_m, self.lookahead_gain * ego.speed + 2.0)

        d2 = (px - ego.x) ** 2 + (py - ego.y) ** 2
        candidates = np.nonzero(d2 >= ld * ld)[0]
        i_goal = int(candidates[0]) if candidates.size else int(np.argmin(d2))

        # bearing to goal in the ego frame
        gx, gy = px[i_goal] - ego.x, py[i_goal] - ego.y
        cy, sy = math.cos(-ego.yaw), math.sin(-ego.yaw)
        xl = gx * cy - gy * sy
        yl = gx * sy + gy * cy
        alpha = math.atan2(yl, xl)

        return math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
