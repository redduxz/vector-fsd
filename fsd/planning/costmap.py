"""CostMap — drivable-cost field for scoring candidate paths.

A 2-D cost field is rasterized over the union of all candidate extents:

    field(x, y) = sum_over_objects( w_cls * exp(-d^2 / (2*sigma^2)) )
                  +  BIG  wherever d < r_inflated   (hard occupancy)

where d is the distance to an object's center and r_inflated covers the
object's bounding radius plus the ego radius and a safety margin. This is
the classic "soft + hard" costmap: blobs of gradient cost around every
perceived object, and effectively infinite cost where a collision is
certain.

Each candidate (a list of Waypoints, or a Trajectory) is scored by:

    J = w_field * mean(field along path)
      + w_lane  * mean((lat(s) - lat_desired(s))^2)     (if lane detected)
      + w_smooth* path curvature energy
      - w_prog  * forward progress along its own heading
      + w_oob   * (# out-of-field points)

`lat_desired(s) = -center_offset * ramp(s)` pulls the path toward lane
center — the lane-center line sits at lateral position -center_offset in
the candidate's start frame, reached over `converge_m` meters.

evaluate() returns the index of the cheapest candidate, or -1 if the
input is empty.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from fsd.core.logger import get
from fsd.core.types import PerceptionOutput, Trajectory

_log = get("planning.costmap")

_CLASS_WEIGHT = {
    "vehicle": 1.0,
    "pedestrian": 1.6,
    "cyclist": 1.3,
    "sign": 0.0,
    "misc": 0.8,
}


class CostMap:
    """Rasterized drivable-cost field + candidate path evaluator."""

    def __init__(
        self,
        resolution_m: float = 0.5,
        bound_margin_m: float = 6.0,
        ego_radius_m: float = 1.2,
        safety_margin_m: float = 0.5,
        sigma_m: float = 2.0,
        hard_cost: float = 1000.0,
        w_field: float = 8.0,
        w_lane: float = 4.0,
        w_smooth: float = 0.6,
        w_progress: float = 0.15,
        w_oob: float = 50.0,
        converge_m: float = 15.0,
        min_confidence: float = 0.2,
        max_cells: int = 400_000,
    ) -> None:
        self.resolution = float(resolution_m)
        self.bound_margin = float(bound_margin_m)
        self.ego_radius = float(ego_radius_m)
        self.safety_margin = float(safety_margin_m)
        self.sigma = float(sigma_m)
        self.hard_cost = float(hard_cost)
        self.w_field = float(w_field)
        self.w_lane = float(w_lane)
        self.w_smooth = float(w_smooth)
        self.w_progress = float(w_progress)
        self.w_oob = float(w_oob)
        self.converge = float(converge_m)
        self.min_conf = float(min_confidence)
        self.max_cells = int(max_cells)

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        candidates: Sequence,
        perception: PerceptionOutput,
    ) -> int:
        """Return the index of the lowest-cost candidate path."""
        paths = [self._as_xy(c) for c in candidates]
        paths = [p for p in paths if p.shape[0] >= 2]
        if not paths:
            _log.warning("no usable candidates")
            return -1
        if len(paths) != len(candidates):
            _log.debug("dropped %d degenerate candidates",
                       len(candidates) - len(paths))

        field, ox, oy, res = self._build_field(paths, perception)

        best_i, best_j = -1, math.inf
        for i, pts in enumerate(paths):
            j = self._path_cost(pts, field, ox, oy, res, perception)
            if j < best_j:
                best_i, best_j = i, j
        return best_i

    # ------------------------------------------------------------------ #
    # field construction                                                  #
    # ------------------------------------------------------------------ #
    def _build_field(
        self,
        paths: List[np.ndarray],
        perception: PerceptionOutput,
    ):
        """Occupancy + proximity field over the candidate bounds."""
        all_pts = np.vstack(paths)
        lo = all_pts.min(axis=0) - self.bound_margin
        hi = all_pts.max(axis=0) + self.bound_margin
        for o in perception.objects:
            lo = np.minimum(lo, [o.position.x, o.position.y])
            hi = np.maximum(hi, [o.position.x, o.position.y])
        lo -= self.bound_margin
        hi += self.bound_margin

        res = self.resolution
        nx = int((hi[0] - lo[0]) / res) + 1
        ny = int((hi[1] - lo[1]) / res) + 1
        while nx * ny > self.max_cells and res < 4.0:
            res *= 2.0
            nx = int((hi[0] - lo[0]) / res) + 1
            ny = int((hi[1] - lo[1]) / res) + 1
        if res != self.resolution:
            _log.debug("costmap res relaxed to %.2f m", res)

        gx, gy = np.meshgrid(
            lo[0] + np.arange(nx) * res,
            lo[1] + np.arange(ny) * res,
            indexing="ij",
        )
        field = np.zeros((nx, ny))
        two_sigma2 = 2.0 * self.sigma * self.sigma

        for o in perception.objects:
            if o.confidence < self.min_conf:
                continue
            w = _CLASS_WEIGHT.get(o.cls, 0.8)
            if w <= 0.0:
                continue
            r_obj = max(o.bbox_extent.x, o.bbox_extent.y)
            r_inflated = r_obj + self.ego_radius + self.safety_margin
            d2 = (gx - o.position.x) ** 2 + (gy - o.position.y) ** 2
            field += w * o.confidence * np.exp(-d2 / two_sigma2)
            field = np.where(d2 < r_inflated * r_inflated,
                             field + self.hard_cost * w, field)
        return field, float(lo[0]), float(lo[1]), res

    # ------------------------------------------------------------------ #
    # per-candidate scoring                                               #
    # ------------------------------------------------------------------ #
    def _path_cost(
        self,
        pts: np.ndarray,
        field: np.ndarray,
        ox: float,
        oy: float,
        res: float,
        perception: PerceptionOutput,
    ) -> float:
        nx, ny = field.shape
        fxi = (pts[:, 0] - ox) / res
        fyi = (pts[:, 1] - oy) / res
        oob = (fxi < 0) | (fyi < 0) | (fxi > nx - 1) | (fyi > ny - 1)
        ix = np.clip(fxi.astype(int), 0, nx - 1)
        iy = np.clip(fyi.astype(int), 0, ny - 1)
        field_cost = float(np.mean(field[ix, iy])) + \
            self.w_oob * float(np.mean(oob))

        # --- lane-centering term (candidate start frame) ---------------- #
        lane_cost = 0.0
        lane = perception.lane
        if lane.detected:
            h0 = math.atan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])
            nx_, ny_ = -math.sin(h0), math.cos(h0)          # left normal
            lat = (pts[:, 0] - pts[0, 0]) * nx_ + \
                  (pts[:, 1] - pts[0, 1]) * ny_
            s = np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))
            s = np.concatenate([[0.0], s])
            ramp = np.clip(s / self.converge, 0.0, 1.0)
            desired = -lane.center_offset * ramp
            lane_cost = float(np.mean((lat - desired) ** 2))

        # --- smoothness: heading-change energy --------------------------- #
        hdg = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
        d_hdg = np.diff(np.unwrap(hdg))
        smooth_cost = float(np.sum(d_hdg * d_hdg))

        # --- forward progress along start heading ------------------------ #
        fx = pts[-1, 0] - pts[0, 0]
        fy = pts[-1, 1] - pts[0, 1]
        h0 = math.atan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])
        progress = fx * math.cos(h0) + fy * math.sin(h0)

        return (self.w_field * field_cost
                + self.w_lane * lane_cost
                + self.w_smooth * smooth_cost
                - self.w_progress * progress)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_xy(candidate) -> np.ndarray:
        """Trajectory | List[Waypoint] | array-like -> Nx2 float array."""
        pts = candidate.points if isinstance(candidate, Trajectory) \
            else candidate
        arr = np.asarray(
            [(p.x, p.y) for p in pts], dtype=float) if len(pts) else \
            np.zeros((0, 2))
        return arr
