"""Lane estimation for the FSD stack.

Two complementary paths:

* ``carla_waypoint`` path (map ground truth): consumes a ``carla.Waypoint``
  — optionally paired with the ego transform/state — and derives lane width,
  lateral offset, heading error and curvature from map geometry.
* Vision fallback: classic IPM pipeline (lane-pixel mask -> perspective warp
  to bird's-eye view -> sliding-window polynomial fit). numpy only, no cv2.

Sign conventions (CARLA/UE axes: +x forward, +y right):
    center_offset > 0 -> ego is displaced to the RIGHT of the lane center line
    heading_error > 0 -> lane direction is rotated right relative to ego yaw
    curvature     > 0 -> road bends to the right ahead
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from fsd.core.logger import get
from fsd.core.types import LaneInfo

try:
    import carla
except ImportError:  # CARLA optional — vision path still works
    carla = None

log = get("perception.lane")


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 projective transform mapping src->dst (>=4 point correspondences)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    A, b = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -X * x, -X * y])
        b.append(X)
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -Y * x, -Y * y])
        b.append(Y)
    h = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)[0]
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def warp(src_img: np.ndarray, M_dst_to_src: np.ndarray,
         out_shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor perspective warp; M maps dst (col,row) -> src (col,row)."""
    H, W = out_shape
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    p = np.stack([cols.ravel(), rows.ravel(), np.ones(H * W)])
    q = M_dst_to_src @ p
    w = q[2]
    ok = np.abs(w) > 1e-6
    sx = np.zeros(H * W)
    sy = np.zeros(H * W)
    np.divide(q[0], w, out=sx, where=ok)
    np.divide(q[1], w, out=sy, where=ok)
    sx = np.round(sx).astype(np.int64)
    sy = np.round(sy).astype(np.int64)
    ok &= (sx >= 0) & (sx < src_img.shape[1]) & (sy >= 0) & (sy < src_img.shape[0])
    out = np.zeros(H * W, dtype=src_img.dtype)
    out[ok] = src_img[sy[ok], sx[ok]]
    return out.reshape(out_shape)


class LaneDetector:
    """Produces ``LaneInfo`` from CARLA map data or a front camera image."""

    def __init__(self, bev_w: int = 200, bev_h: int = 320, mpp: float = 0.2,
                 assumed_width: float = 3.5,
                 src_frac: Optional[np.ndarray] = None):
        self.bev_w, self.bev_h = bev_w, bev_h
        self.mpp = mpp                       # bird's-eye meters per pixel
        self.assumed_width = assumed_width
        # trapezoid in normalized image coords: TL, TR, BR, BL
        self.src_frac = np.asarray(src_frac if src_frac is not None else
                                   [[0.44, 0.60], [0.56, 0.60],
                                    [0.98, 0.97], [0.02, 0.97]], float)
        self._H_cache: dict = {}

    # ------------------------------------------------------------------ API
    def detect(self, image=None, carla_waypoint=None, carla_map=None) -> LaneInfo:
        """Best available lane estimate; never raises on missing inputs."""
        if carla_waypoint is not None:
            info = self._from_waypoint(carla_waypoint, carla_map)
            if info is not None:
                return info
            log.debug("waypoint path failed, falling back to vision")
        if image is not None:
            return self._from_image(image)
        return LaneInfo(0.5 * self.assumed_width, 0.5 * self.assumed_width,
                        0.0, 0.0, 0.0, self.assumed_width, detected=False)

    # ----------------------------------------------------------- CARLA path
    def _from_waypoint(self, wp_in, carla_map) -> Optional[LaneInfo]:
        wp, ego = self._split_input(wp_in)
        if wp is None:
            return None
        if not hasattr(wp, "transform") and carla_map is not None:
            # a bare carla.Location was passed — project it onto the road
            wp = carla_map.get_waypoint(wp, project_to_road=True)
        if not hasattr(wp, "transform"):
            return None
        tr = wp.transform
        wx, wy = tr.location.x, tr.location.y
        lane_yaw = math.radians(tr.rotation.yaw)
        width = float(getattr(wp, "lane_width", self.assumed_width))

        center_off, head_err = 0.0, 0.0
        if ego is not None:
            ex, ey, eyaw = ego
            dx, dy = ex - wx, ey - wy
            # lateral component along the lane-right direction (-sin, cos)
            center_off = -math.sin(lane_yaw) * dx + math.cos(lane_yaw) * dy
            head_err = _wrap_angle(lane_yaw - eyaw)

        kappa = self._curvature(wp, carla_map)
        left = 0.5 * width + center_off
        right = 0.5 * width - center_off
        return LaneInfo(left, right, center_off, head_err, kappa, width, True)

    @staticmethod
    def _split_input(obj):
        """Accepts waypoint | (waypoint, ego) | dict; returns (wp, (x,y,yaw)|None)."""
        wp, ego = obj, None
        if isinstance(obj, dict):
            wp = obj.get("waypoint") or obj.get("wp") or obj.get("location")
            ego = obj.get("ego") or obj.get("transform") or obj.get("vehicle")
        elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
            wp, ego = obj[0], obj[1]
        return wp, LaneDetector._ego_xy_yaw(ego)

    @staticmethod
    def _ego_xy_yaw(e):
        """Extract (x, y, yaw_rad) from carla.Transform/Actor, VehicleState, dict."""
        if e is None:
            return None
        if hasattr(e, "rotation") and hasattr(e, "location"):      # carla.Transform
            return e.location.x, e.location.y, math.radians(e.rotation.yaw)
        if hasattr(e, "get_transform"):                            # carla.Actor
            t = e.get_transform()
            return t.location.x, t.location.y, math.radians(t.rotation.yaw)
        if all(hasattr(e, k) for k in ("x", "y", "yaw")):          # VehicleState
            return float(e.x), float(e.y), float(e.yaw)
        if isinstance(e, dict):
            return float(e["x"]), float(e["y"]), float(e.get("yaw", 0.0))
        return None

    @staticmethod
    def _curvature(wp, carla_map) -> float:
        """Signed curvature of the circumscribed circle through wp, wp+5m, wp+10m."""
        pts = [(wp.transform.location.x, wp.transform.location.y)]
        try:
            for d in (5.0, 10.0):
                nxt = wp.next(d)
                if nxt:
                    p = nxt[0].transform.location
                    pts.append((p.x, p.y))
        except Exception:
            pass
        if len(pts) < 3 and carla_map is not None and hasattr(wp, "transform"):
            tr = wp.transform
            yaw = math.radians(tr.rotation.yaw)
            for d in (5.0, 10.0):
                try:
                    loc = carla.Location(x=tr.location.x + d * math.cos(yaw),
                                         y=tr.location.y + d * math.sin(yaw)) \
                        if carla is not None else None
                    if loc is None:
                        break
                    w = carla_map.get_waypoint(loc, project_to_road=True)
                    pts.append((w.transform.location.x, w.transform.location.y))
                except Exception:
                    break
        if len(pts) < 3:
            return 0.0
        (x1, y1), (x2, y2), (x3, y3) = pts[:3]
        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if a * b * c < 1e-6:
            return 0.0
        return float(2.0 * cross / (a * b * c))

    # ----------------------------------------------------------- image path
    def _from_image(self, image) -> LaneInfo:
        img = np.asarray(image)
        mask = self._lane_mask(img)
        M = self._bev_matrix(img.shape[:2])
        bev = warp(mask.astype(np.uint8), M, (self.bev_h, self.bev_w)).astype(bool)
        left, right = self._fit_lanes(bev)
        return self._metrics(left, right)

    def _bev_matrix(self, hw: Tuple[int, int]) -> np.ndarray:
        """dst(BEV px) -> src(image px) homography, cached per image size."""
        if hw not in self._H_cache:
            H, W = hw
            src = self.src_frac * np.array([W, H])
            dst = np.array([[0, 0], [self.bev_w - 1, 0],
                            [self.bev_w - 1, self.bev_h - 1],
                            [0, self.bev_h - 1]], float)
            self._H_cache[hw] = homography(dst, src)
        return self._H_cache[hw]

    @staticmethod
    def _lane_mask(img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        if img.ndim == 2:
            g = img.astype(np.float32)
            color = np.zeros((H, W), bool)
        else:
            f = img.astype(np.float32)
            r, gc, b = f[..., 0], f[..., 1], f[..., 2]
            g = 0.299 * r + 0.587 * gc + 0.114 * b
            color = ((r > 170) & (gc > 170) & (b > 170)) | \
                    ((r > 150) & (gc > 130) & (b < 140))      # white | yellow
        gx = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        mask = color | ((np.abs(gx) > 40.0) & (g > 90.0))
        mask[: int(0.55 * H)] = False                        # above horizon-ish
        mask[:, : int(0.02 * W)] = False
        mask[:, int(0.98 * W):] = False
        return mask

    def _fit_lanes(self, bev: np.ndarray):
        """Sliding-window search -> poly coeffs c(r) = a r^2 + b r + c0 per side."""
        H, W = bev.shape
        hist = bev[int(0.66 * H):].sum(axis=0)
        mid = W // 2
        peaks = (int(np.argmax(hist[:mid])), int(np.argmax(hist[mid:])) + mid)
        out = []
        for side, start in enumerate(peaks):
            if hist[start] < 5:
                out.append(None)
                continue
            r, c = self._window_points(bev, start)
            out.append(np.polyfit(r, c, 2) if len(r) >= 25 else None)
        return out

    @staticmethod
    def _window_points(mask: np.ndarray, start_col: int, n_win: int = 9):
        H, W = mask.shape
        win_h = max(1, H // n_win)
        half = max(8, W // 14)
        rs, cs, c = [], [], start_col
        for w in range(n_win):
            r1, r0 = H - (w + 1) * win_h, H - w * win_h
            lo, hi = max(0, c - half), min(W, c + half)
            ys, xs = np.nonzero(mask[r1:r0, lo:hi])
            if len(ys):
                rs.append(ys + r1)
                cs.append(xs + lo)
                c = int(round(xs.mean() + lo))
        if not rs:
            return np.empty(0), np.empty(0)
        return np.concatenate(rs), np.concatenate(cs)

    def _metrics(self, left, right) -> LaneInfo:
        H, W, mpp = self.bev_h, self.bev_w, self.mpp
        rb = H - 1                                     # nearest BEV row (ego)
        if left is None and right is None:
            return LaneInfo(0.5 * self.assumed_width, 0.5 * self.assumed_width,
                            0.0, 0.0, 0.0, self.assumed_width, detected=False)
        w_px = self.assumed_width / mpp
        if left is not None and right is not None:
            cl, cr = np.polyval(left, rb), np.polyval(right, rb)
            if cr - cl > 1.0:
                w_px = cr - cl
            center_poly = 0.5 * (left + right)
        elif left is not None:                         # mirror with assumed width
            center_poly = left.copy()
            center_poly[-1] += 0.5 * w_px
        else:
            center_poly = right.copy()
            center_poly[-1] -= 0.5 * w_px

        lane_c = np.polyval(center_poly, rb)
        center_off = (0.5 * W - lane_c) * mpp          # + = ego right of center
        a, b = center_poly[0], center_poly[1]
        B = -(2.0 * a * rb + b)                        # dy/ds at s = 0
        A = a / mpp                                    # y(s) in meters
        head_err = math.atan(B)
        kappa = 2.0 * A / max((1.0 + B * B) ** 1.5, 1e-6)
        width = float(np.clip(w_px * mpp, 2.0, 6.0))
        return LaneInfo(0.5 * width + center_off, 0.5 * width - center_off,
                        center_off, head_err, float(kappa), width, True)
