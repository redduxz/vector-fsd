"""Bird's-eye occupancy grid (world frame, log-odds, numpy only).

Cells accumulate log-odds evidence: raycast cells get free evidence,
endpoints get occupied evidence. Positive log-odds = occupied. The grid is
anchored at a world ``origin`` that can be recentered on the ego as it moves.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from fsd.core.logger import get

log = get("perception.occupancy")


class OccupancyGrid:
    """2-D BEV occupancy grid; cell (0,0) sits at world (ox, oy)."""

    def __init__(self, width_m: float = 120.0, height_m: float = 120.0,
                 resolution: float = 0.25, origin: Tuple[float, float] = (0.0, 0.0)):
        self.res = float(resolution)
        self.w = int(round(width_m / self.res))
        self.h = int(round(height_m / self.res))
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.lo_hit, self.lo_free = 0.9, -0.4
        self.lo_min, self.lo_max = -4.0, 4.0
        self.grid = np.zeros((self.h, self.w), dtype=np.float32)

    # --------------------------------------------------------------- basics
    def clear(self) -> None:
        self.grid.fill(0.0)

    def set_origin(self, ox: float, oy: float, keep: bool = True) -> None:
        """Move the world anchor (e.g. recenter on ego); optionally keep data."""
        if keep and (abs(ox - self.ox) < self.w * self.res and
                     abs(oy - self.oy) < self.h * self.res):
            dx = int(round((ox - self.ox) / self.res))
            dy = int(round((oy - self.oy) / self.res))
            new = np.zeros_like(self.grid)
            sx0, sx1 = max(0, -dx), min(self.w, self.w - dx)
            sy0, sy1 = max(0, -dy), min(self.h, self.h - dy)
            dx0, dx1 = max(0, dx), min(self.w, self.w + dx)
            dy0, dy1 = max(0, dy), min(self.h, self.h + dy)
            if sx1 > sx0 and sy1 > sy0:
                new[sy0:sy1, sx0:sx1] = self.grid[dy0:dy1, dx0:dx1]
            self.grid = new
        else:
            self.clear()
        self.ox, self.oy = float(ox), float(oy)

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.ox) / self.res), int((y - self.oy) / self.res)

    def in_bounds(self, x: float, y: float) -> bool:
        cx, cy = self._cell(x, y)
        return 0 <= cx < self.w and 0 <= cy < self.h

    # --------------------------------------------------------------- update
    def _bump(self, cx: int, cy: int, delta: float) -> None:
        if 0 <= cx < self.w and 0 <= cy < self.h:
            v = self.grid[cy, cx] + delta
            self.grid[cy, cx] = min(max(v, self.lo_min), self.lo_max)

    def mark(self, x: float, y: float, occupied: bool = True) -> None:
        """Add occupied/free evidence for the cell containing world (x, y)."""
        cx, cy = self._cell(x, y)
        self._bump(cx, cy, self.lo_hit if occupied else self.lo_free)

    def mark_points(self, points: Iterable[Sequence[float]],
                    occupied: bool = True) -> None:
        for p in points:
            self.mark(float(p[0]), float(p[1]), occupied)

    def raycast(self, ox: float, oy: float, x: float, y: float,
                hit: bool = True) -> None:
        """Bresenham from (ox,oy) to (x,y); free evidence along the ray,
        occupied evidence at the endpoint if ``hit``."""
        cx0, cy0 = self._cell(ox, oy)
        cx1, cy1 = self._cell(x, y)
        dx, dy = abs(cx1 - cx0), -abs(cy1 - cy0)
        sx = 1 if cx0 < cx1 else -1
        sy = 1 if cy0 < cy1 else -1
        err, cx, cy = dx + dy, cx0, cy0
        while True:
            if cx == cx1 and cy == cy1:
                self._bump(cx, cy, self.lo_hit if hit else self.lo_free)
                break
            self._bump(cx, cy, self.lo_free)
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                cx += sx
            if e2 <= dx:
                err += dx
                cy += sy

    def insert_scan(self, points: Iterable[Sequence[float]],
                    origin: Tuple[float, float],
                    max_range: Optional[float] = None) -> None:
        """Raycast each world-frame scan endpoint from the sensor origin."""
        ox, oy = origin
        for p in points:
            x, y = float(p[0]), float(p[1])
            if max_range is not None and \
                    math.hypot(x - ox, y - oy) > max_range:
                continue
            self.raycast(ox, oy, x, y, hit=True)

    # ---------------------------------------------------------------- query
    def log_odds(self, x: float, y: float) -> float:
        cx, cy = self._cell(x, y)
        if 0 <= cx < self.w and 0 <= cy < self.h:
            return float(self.grid[cy, cx])
        return 0.0

    def is_occupied(self, x: float, y: float, threshold: float = 0.6) -> bool:
        cx, cy = self._cell(x, y)
        return (0 <= cx < self.w and 0 <= cy < self.h and
                self.grid[cy, cx] > threshold)

    def occupied_fraction(self) -> float:
        return float(np.mean(self.grid > 0.6))

    def free_space_ahead(self, ego_x: float, ego_y: float, yaw: float,
                         max_dist: float = 80.0,
                         half_width: float = 1.0) -> float:
        """Meters of clear corridor along ``yaw`` before an occupied cell.

        Marches a corridor of ``2*half_width`` lateral clearance; the first
        occupied cell across the corridor wins. Leaving the grid bounds is
        reported as the distance to the grid edge (unknown beyond).
        """
        ux, uy = math.cos(yaw), math.sin(yaw)
        vx, vy = -uy, ux                          # lateral (right) unit vector
        n_lat = max(1, int(math.ceil(half_width / self.res)))
        lats = np.linspace(-half_width, half_width, 2 * n_lat + 1)
        step = self.res
        s = step
        while s < max_dist:
            bx, by = ego_x + ux * s, ego_y + uy * s
            if not self.in_bounds(bx, by):
                return s                          # ran out of known space
            for lat in lats:
                if self.is_occupied(bx + vx * lat, by + vy * lat):
                    return s
            s += step
        return max_dist

    def as_uint8(self) -> np.ndarray:
        """0 free / 128 unknown / 255 occupied — handy for debugging dumps."""
        return np.clip(128 + self.grid * 64, 0, 255).astype(np.uint8)
