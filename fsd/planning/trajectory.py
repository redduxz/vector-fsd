"""Trajectory planner — route + behavior decision -> Trajectory.

Steps per generate() call:

1. Slice the route forward from the closest point to ego and resample to
   uniform spacing (`point_spacing_m`) over the planning horizon.
2. Frenet-style lateral shift: the path is offset along its left normal
   n = (-sin yaw, cos yaw) by a decaying -center_offset term (so it passes
   through the ego's current lateral position and blends back to lane
   center over `centering_distance_m`), plus a smooth ramp to ±lane_width
   during LANE_CHANGE_L/R.
3. Per-point speed cap:
       v_i <= min(speed_limit, max_speed, sqrt(a_lat / |kappa_i|))
   where kappa is finite-differenced from the resampled heading.
4. Stop-shaped cap where required (FOLLOW_LEAD gap / light / emergency):
       v_i <= sqrt(2 * a_dec * max(0, d_stop - s_i))
   then a backward pass enforces deceleration feasibility along the whole
   profile, and a forward pass caps acceleration from the current speed.

Per-point speeds are carried in `Waypoint.speed_limit` (the planner's
commanded speed at that arc position); `Trajectory.target_speed` is the
behavior target clamped to the profile.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from fsd.core.logger import get
from fsd.core.types import (
    PerceptionOutput,
    Trajectory,
    VehicleState,
    Waypoint,
)
from fsd.planning.behavior import BehaviorPlanner

_log = get("planning.trajectory")


def _to_ego(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    return dx * cy + dy * sy, -dx * sy + dy * cy


class TrajectoryPlanner:
    """Route + decision + perception -> speed-profiled Trajectory."""

    def __init__(
        self,
        horizon_s: float = 4.0,
        point_spacing_m: float = 1.0,
        max_points: int = 60,
        max_speed_mps: float = 16.7,
        max_lat_accel_mps2: float = 2.5,
        comfort_decel_mps2: float = 3.0,
        emergency_decel_mps2: float = 6.0,
        max_accel_mps2: float = 2.0,
        standstill_gap_m: float = 3.0,
        stop_margin_m: float = 4.0,
        emergency_margin_m: float = 1.5,
        centering_distance_m: float = 15.0,
        lane_width_m: float = 3.5,
        ego_half_length_m: float = 2.4,
        min_horizon_speed_mps: float = 3.0,
    ) -> None:
        self.horizon_s = float(horizon_s)
        self.spacing = float(point_spacing_m)
        self.max_points = int(max_points)
        self.max_speed = float(max_speed_mps)
        self.a_lat = float(max_lat_accel_mps2)
        self.a_dec = float(comfort_decel_mps2)
        self.a_emerg = float(emergency_decel_mps2)
        self.a_acc = float(max_accel_mps2)
        self.standstill_gap = float(standstill_gap_m)
        self.stop_margin = float(stop_margin_m)
        self.emergency_margin = float(emergency_margin_m)
        self.centering_distance = float(centering_distance_m)
        self.lane_width = float(lane_width_m)
        self.ego_half_length = float(ego_half_length_m)
        self.min_horizon_speed = float(min_horizon_speed_mps)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        route: List[Waypoint],
        decision: Tuple[str, float],
        perception: PerceptionOutput,
        ego: VehicleState,
    ) -> Trajectory:
        if not route:
            _log.warning("empty route — empty trajectory")
            return Trajectory(points=[], target_speed=0.0,
                              horizon_s=self.horizon_s)

        name, decision_speed = decision
        decision_speed = max(0.0, float(decision_speed))

        # ---- 1. slice + resample the route ----------------------------- #
        horizon_m = self.horizon_s * max(decision_speed,
                                         self.min_horizon_speed)
        xs, ys, zs, lims = self._slice_resample(route, ego, horizon_m)
        if xs.size < 2:
            return Trajectory(points=[], target_speed=0.0,
                              horizon_s=self.horizon_s)

        n_pts = xs.size
        s = np.arange(n_pts) * self.spacing

        headings = np.arctan2(np.gradient(ys), np.gradient(xs))
        headings = np.unwrap(headings)

        # ---- 2. frenet-style lateral shift ------------------------------ #
        shift = self._lateral_shift(s, name, perception, ego)

        nx = -np.sin(headings)
        ny = np.cos(headings)
        px = xs + nx * shift
        py = ys + ny * shift

        # curvature of the shifted path: kappa = |x'y'' - y'x''|/(x'^2+y'^2)^1.5
        # (per-index derivatives — the uniform spacing cancels out)
        dx, dy = np.gradient(px), np.gradient(py)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
        kappa = np.abs(dx * ddy - dy * ddx) / denom

        # ---- 3. per-point speed caps ------------------------------------ #
        v_lim = np.minimum(lims, self.max_speed)
        v_curve = np.sqrt(self.a_lat / np.maximum(kappa, 1e-4))
        v = np.minimum(v_lim, v_curve)

        # ---- 4. stop shaping + feasibility passes ------------------------ #
        d_stop, decel = self._stop_distance(name, perception, ego)
        if d_stop is not None:
            v_stop = np.sqrt(np.maximum(
                0.0, 2.0 * decel * np.maximum(0.0, d_stop - s)))
            v = np.minimum(v, v_stop)

        # backward pass: v_i <= sqrt(v_{i+1}^2 + 2 a_dec ds)
        for i in range(n_pts - 2, -1, -1):
            feas = math.sqrt(v[i + 1] ** 2 + 2.0 * self.a_dec * self.spacing)
            if feas < v[i]:
                v[i] = feas
        # forward pass: v_{i+1} <= sqrt(v_i^2 + 2 a_acc ds), seeded at the
        # current speed so the profile is physically reachable
        v[0] = min(v[0], max(ego.speed, 0.0))
        for i in range(1, n_pts):
            feas = math.sqrt(v[i - 1] ** 2 + 2.0 * self.a_acc * self.spacing)
            v[i] = min(v[i], feas)

        points = [
            Waypoint(x=float(px[i]), y=float(py[i]), z=float(zs[i]),
                     yaw=float(headings[i]), speed_limit=float(v[i]))
            for i in range(n_pts)
        ]
        # target speed = what the controller should chase this tick: the
        # profile speed a couple of metres out, not v[0] (which is pinned to
        # the current speed and would latch a stopped car at 0 forever)
        i_tgt = min(3, n_pts - 1)
        target = min(decision_speed, float(v[i_tgt]))
        return Trajectory(points=points, target_speed=float(target),
                          horizon_s=self.horizon_s)

    # ------------------------------------------------------------------ #
    def _slice_resample(
        self,
        route: List[Waypoint],
        ego: VehicleState,
        horizon_m: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cut the route at the closest point and resample uniformly."""
        rx = np.array([p.x for p in route])
        ry = np.array([p.y for p in route])
        rz = np.array([p.z for p in route])
        rl = np.array([p.speed_limit for p in route])

        i0 = int(np.argmin((rx - ego.x) ** 2 + (ry - ego.y) ** 2))
        rx, ry, rz, rl = rx[i0:], ry[i0:], rz[i0:], rl[i0:]

        seg = np.hypot(np.diff(rx), np.diff(ry))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        reach = min(s[-1], horizon_m)
        if reach < self.spacing:
            return (np.array([]), np.array([]), np.array([]), np.array([]))

        n = min(int(reach / self.spacing) + 1, self.max_points)
        su = np.arange(n) * self.spacing
        # np.interp needs strictly increasing s — collapse duplicates
        s_u, idx = np.unique(s, return_index=True)
        return (np.interp(su, s_u, rx[idx]),
                np.interp(su, s_u, ry[idx]),
                np.interp(su, s_u, rz[idx]),
                np.interp(su, s_u, rl[idx]))

    # ------------------------------------------------------------------ #
    def _lateral_shift(
        self,
        s: np.ndarray,
        decision_name: str,
        perception: PerceptionOutput,
        ego: VehicleState,
    ) -> np.ndarray:
        """Signed lateral offset applied along the path's left normal."""
        shift = np.zeros_like(s)

        # re-centering: exponential decay of the measured center offset
        if perception.lane.detected and abs(
                perception.lane.center_offset) > 1e-3:
            tau = self.centering_distance / 3.0      # ~95 % merged at L
            shift += perception.lane.center_offset * np.exp(-s / tau)

        # lane change: smoothstep ramp to the adjacent lane
        if decision_name == BehaviorPlanner.LANE_CHANGE_LEFT:
            shift += self.lane_width * self._smoothstep(
                s, self._lc_distance(ego))
        elif decision_name == BehaviorPlanner.LANE_CHANGE_RIGHT:
            shift -= self.lane_width * self._smoothstep(
                s, self._lc_distance(ego))
        return shift

    @staticmethod
    def _smoothstep(s: np.ndarray, d: float) -> np.ndarray:
        t = np.clip(s / max(d, 1e-3), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _lc_distance(ego: VehicleState) -> float:
        # longer merges when faster; bounded for sanity
        return float(np.clip(2.5 * ego.speed, 18.0, 45.0))

    # ------------------------------------------------------------------ #
    def _stop_distance(
        self,
        decision_name: str,
        perception: PerceptionOutput,
        ego: VehicleState,
    ) -> Tuple[Optional[float], float]:
        """(distance-to-stop, decel) for decisions that need a stop cap."""
        if decision_name == BehaviorPlanner.EMERGENCY_STOP:
            return (max(perception.free_space_ahead - self.emergency_margin,
                        0.0), self.a_emerg)

        if decision_name == BehaviorPlanner.STOP_AT_LIGHT:
            return (max(perception.free_space_ahead - self.stop_margin,
                        0.0), self.a_dec)

        if decision_name == BehaviorPlanner.FOLLOW_LEAD:
            gap = self._lead_gap(perception, ego)
            d = min(gap - self.standstill_gap,
                    perception.free_space_ahead - self.stop_margin)
            return max(d, 0.0), self.a_dec

        return None, self.a_dec

    def _lead_gap(
        self,
        perception: PerceptionOutput,
        ego: VehicleState,
    ) -> float:
        """Bumper gap to the nearest object in our corridor (m)."""
        best = perception.free_space_ahead
        for o in perception.objects:
            lon, lat = _to_ego(o.position.x - ego.x,
                               o.position.y - ego.y, ego.yaw)
            half_w = max(2.0, o.bbox_extent.y + 0.4)
            if lon < -1.0 or abs(lat) > half_w:
                continue
            gap = lon - o.bbox_extent.x - self.ego_half_length
            if 0.0 <= gap < best:
                best = gap
        return max(best, 0.0)
