"""Multi-sensor object fusion.

Camera detections (``DetectedObject``s), radar returns and lidar point clouds
are merged into persistent tracks. Association is greedy nearest-neighbor with
a distance gate; each track runs a constant-velocity Kalman filter on
[x, y, vx, vy]. Radar contributes a linear radial-velocity measurement
(v_obj . u = v_ego . u - closing_rate), lidar contributes clustered positions,
camera contributes class + extent.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fsd.core.logger import get
from fsd.core.types import DetectedObject, Vec3

try:
    import carla
except ImportError:
    carla = None

log = get("perception.fusion")


@dataclass
class _Meas:
    x: float
    y: float
    z: float = 0.0
    cls: str = "misc"
    extent: Tuple[float, float, float] = (0.5, 0.5, 0.8)
    conf: float = 0.5
    vel: Optional[Tuple[float, float]] = None       # absolute (vx, vy), world
    radial: Optional[Tuple[float, float, float]] = None  # (v_radial, ux, uy)


class _Track:
    """Constant-velocity Kalman track: state [x, y, vx, vy], world frame."""
    _next_id = 1

    def __init__(self, m: _Meas):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.x = np.array([m.x, m.y,
                           m.vel[0] if m.vel else 0.0,
                           m.vel[1] if m.vel else 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 9.0, 9.0])
        self.z = m.z
        self.cls = m.cls
        self.extent = np.asarray(m.extent, float)
        self.conf = m.conf
        self.hits = 1
        self.misses = 0

    def predict(self, dt: float, q: float = 2.0) -> None:
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                     dtype=float)
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * (q * q)
        self.misses += 1
        self.conf *= 0.85

    def _kf(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def update(self, m: _Meas) -> None:
        r = max(1.0 - m.conf, 0.1)
        self._kf(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float),
                 np.array([m.x, m.y]), np.eye(2) * (0.6 ** 2) * r)
        if m.vel is not None:
            self._kf(np.array([[0, 0, 1, 0], [0, 0, 0, 1]], float),
                     np.asarray(m.vel), np.eye(2) * (1.0 ** 2) * r)
        if m.radial is not None:
            vr, ux, uy = m.radial
            self._kf(np.array([[0, 0, ux, uy]], float),
                     np.array([vr]), np.array([[0.5 ** 2]]))
        self.z = m.z
        self.hits += 1
        self.misses = 0
        self.conf = min(1.0, 0.6 * self.conf + 0.4 * m.conf + 0.1)
        if m.cls != "misc":
            self.cls = m.cls
        if m.extent is not None:
            self.extent = 0.7 * self.extent + 0.3 * np.asarray(m.extent, float)

    def to_object(self) -> DetectedObject:
        return DetectedObject(
            obj_id=self.id, cls=self.cls,
            position=Vec3(float(self.x[0]), float(self.x[1]), float(self.z)),
            velocity=Vec3(float(self.x[2]), float(self.x[3]), 0.0),
            bbox_extent=Vec3(*map(float, self.extent)),
            confidence=float(np.clip(self.conf, 0.0, 1.0)))


class SensorFusion:
    """Fuses camera objects + radar points + lidar points into tracks."""

    def __init__(self, gate_m: float = 4.0, max_misses: int = 5,
                 min_hits: int = 1, cluster_eps: float = 1.2,
                 cluster_min_pts: int = 4, max_sensor_range: float = 90.0):
        self.gate_m = gate_m
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.cluster_eps = cluster_eps
        self.cluster_min_pts = cluster_min_pts
        self.max_sensor_range = max_sensor_range
        self._tracks: Dict[int, _Track] = {}
        self._t: Optional[float] = None

    # ------------------------------------------------------------------ API
    def fuse(self, camera_objs: Optional[Sequence[DetectedObject]] = None,
             radar_points=None, lidar_points=None, ego=None
             ) -> List[DetectedObject]:
        now = time.time()
        dt = float(np.clip(now - self._t, 1e-3, 0.5)) if self._t else 0.1
        self._t = now
        for tr in self._tracks.values():
            tr.predict(dt)

        ex, ey, eyaw, espeed = _ego_pose(ego)
        meas: List[_Meas] = []
        meas += self._cam_meas(camera_objs or [])
        meas += self._lidar_meas(lidar_points, ex, ey, eyaw)
        meas += self._radar_meas(radar_points, ex, ey, eyaw, espeed)

        meas.sort(key=lambda m: -m.conf)
        for m in meas:
            tr = self._associate(m)
            if tr is None:
                tr = _Track(m)
                self._tracks[tr.id] = tr
            else:
                tr.update(m)
        self._tracks = {i: t for i, t in self._tracks.items()
                        if t.misses <= self.max_misses}
        return [t.to_object() for t in self._tracks.values()
                if t.hits >= self.min_hits and t.misses <= 2]

    # ------------------------------------------------------------ association
    def _associate(self, m: _Meas) -> Optional[_Track]:
        best, best_d = None, self.gate_m
        for tr in self._tracks.values():
            d = math.hypot(m.x - tr.x[0], m.y - tr.x[1])
            if d < best_d:
                best, best_d = tr, d
        return best

    # --------------------------------------------------------------- camera
    @staticmethod
    def _cam_meas(objs: Sequence[DetectedObject]) -> List[_Meas]:
        out = []
        for o in objs:
            vel = (o.velocity.x, o.velocity.y) \
                if o.velocity is not None else None
            out.append(_Meas(o.position.x, o.position.y, o.position.z, o.cls,
                             (o.bbox_extent.x, o.bbox_extent.y, o.bbox_extent.z),
                             o.confidence, vel=vel))
        return out

    # ----------------------------------------------------------------- lidar
    def _lidar_meas(self, pts, ex, ey, eyaw) -> List[_Meas]:
        raw = _lidar_array(pts)
        if len(raw):
            rng = np.hypot(raw[:, 0], raw[:, 1])
            raw = raw[rng <= self.max_sensor_range]
        pts = _to_ego_xy(raw, ex, ey, eyaw)
        out = []
        for g in self._cluster(pts):
            c = pts[g]
            span = 0.5 * (c.max(0) - c.min(0))
            ext = tuple(float(np.clip(s, 0.3, 4.0)) for s in span)
            out.append(_Meas(float(c[:, 0].mean()), float(c[:, 1].mean()),
                             float(c[:, 2].mean()), "misc", ext, 0.5))
        return out

    def _cluster(self, pts: np.ndarray) -> List[List[int]]:
        """Grid-hash single-link clustering (DBSCAN-lite), O(n)."""
        n = len(pts)
        if n < self.cluster_min_pts:
            return []
        keys = np.floor(pts[:, :2] / self.cluster_eps).astype(np.int64)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        cells: Dict[Tuple[int, int], int] = {}
        for i, (kx, ky) in enumerate(keys):
            for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0)):
                j = cells.get((kx + dx, ky + dy))
                if j is not None:
                    pi, pj = find(i), find(j)
                    if pi != pj:
                        parent[pi] = pj
            cells[(kx, ky)] = i
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return [g for g in groups.values() if len(g) >= self.cluster_min_pts]

    # ----------------------------------------------------------------- radar
    def _radar_meas(self, pts, ex, ey, eyaw, espeed) -> List[_Meas]:
        out = []
        for x, y, z, closing in _radar_tuples(pts, self.max_sensor_range):
            wx = ex + math.cos(eyaw) * x - math.sin(eyaw) * y
            wy = ey + math.sin(eyaw) * x + math.cos(eyaw) * y
            ux, uy = wx - ex, wy - ey
            n = math.hypot(ux, uy)
            if n < 1.0:
                continue
            ux, uy = ux / n, uy / n
            # closing speed positive = approaching; ego motion compensates
            v_radial = espeed * (math.cos(eyaw) * ux + math.sin(eyaw) * uy) - closing
            out.append(_Meas(wx, wy, z, "misc", (0.5, 0.5, 0.8), 0.4,
                             radial=(v_radial, ux, uy)))
        return out


# ------------------------------------------------------------------ parsing
def _ego_pose(ego) -> Tuple[float, float, float, float]:
    """(x, y, yaw_rad, speed_mps) from VehicleState / carla actor / dict."""
    if ego is None:
        return 0.0, 0.0, 0.0, 0.0
    if hasattr(ego, "get_transform"):
        t = ego.get_transform()
        try:
            v = ego.get_velocity()
            sp = math.hypot(v.x, v.y)
        except Exception:
            sp = 0.0
        return t.location.x, t.location.y, math.radians(t.rotation.yaw), sp
    if all(hasattr(ego, k) for k in ("x", "y", "yaw")):
        return float(ego.x), float(ego.y), float(ego.yaw), \
            float(getattr(ego, "speed", 0.0))
    if isinstance(ego, dict):
        return (float(ego.get("x", 0.0)), float(ego.get("y", 0.0)),
                float(ego.get("yaw", 0.0)), float(ego.get("speed", 0.0)))
    return 0.0, 0.0, 0.0, 0.0


def _lidar_array(pts) -> np.ndarray:
    """Nx3 ego-frame points from ndarray, sequence, or carla.LidarMeasurement."""
    if pts is None:
        return np.empty((0, 3))
    if isinstance(pts, np.ndarray):
        a = pts.reshape(-1, pts.shape[-1])[:, :3].astype(float)
        return a
    rows = []
    for p in pts:
        q = getattr(p, "point", p)          # carla.LidarDetection -> .point
        rows.append((float(q.x), float(q.y), float(q.z)))
    return np.asarray(rows, float) if rows else np.empty((0, 3))


def _to_ego_xy(pts: np.ndarray, ex, ey, eyaw) -> np.ndarray:
    """Rotate ego-frame points into the world frame."""
    if not len(pts):
        return pts
    c, s = math.cos(eyaw), math.sin(eyaw)
    out = pts.copy()
    out[:, 0] = ex + c * pts[:, 0] - s * pts[:, 1]
    out[:, 1] = ey + s * pts[:, 0] + c * pts[:, 1]
    return out


def _radar_tuples(pts, max_range: float):
    """Yields (x, y, z, closing_speed) ego-frame; accepts ndarray rows or
    carla.RadarDetection (depth/azimuth/altitude/velocity)."""
    if pts is None:
        return
    if isinstance(pts, np.ndarray):
        a = np.atleast_2d(pts)
        for row in a:
            x, y = float(row[0]), float(row[1])
            z = float(row[2]) if len(row) > 2 else 0.0
            vr = float(row[3]) if len(row) > 3 else 0.0
            if math.hypot(x, y) <= max_range:
                yield x, y, z, vr
        return
    for p in pts:
        if all(hasattr(p, k) for k in ("depth", "azimuth", "altitude")):
            ca, cb = math.cos(p.azimuth), math.cos(p.altitude)
            x = p.depth * cb * ca
            y = p.depth * cb * math.sin(p.azimuth)
            z = p.depth * math.sin(p.altitude)
            if p.depth <= max_range:
                yield float(x), float(y), float(z), float(p.velocity)
        else:
            q = getattr(p, "point", p)
            x, y = float(q.x), float(q.y)
            z = float(getattr(q, "z", 0.0))
            vr = float(getattr(p, "velocity", 0.0))
            if math.hypot(x, y) <= max_range:
                yield x, y, z, vr
