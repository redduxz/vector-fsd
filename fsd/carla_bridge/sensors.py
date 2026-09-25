"""Sensor suite for the ego vehicle.

Attaches the full NVIDIA-style rig to a spawned CARLA actor:

    - RGB camera            1280x720 @ cfg.vehicle.camera_hz   (sensor.camera.rgb)
    - Semantic camera       1280x720                           (sensor.camera.semantic_segmentation)
    - LiDAR                 cfg.vehicle.lidar_channels ch      (sensor.lidar.ray_cast)
    - Radar                 cfg.vehicle.radar_hz               (sensor.other.radar)
    - GNSS + IMU                                               (sensor.other.gnss / imu)
    - Collision detector    event stream                       (sensor.other.collision)
    - Lane invasion         event stream                       (sensor.other.lane_invasion)

Every sensor callback pushes a :class:`SensorReading` onto a per-sensor
``queue.Queue``. Consumers drain the queue and keep the freshest reading via
:meth:`SensorSuite.latest_frame`, so the bridge never allocates unbounded
sensor memory and the perception stack always sees the newest frame.

Everything compiles and imports without CARLA installed — ``import carla`` is
guarded at module level and only dereferenced inside :meth:`attach`.
"""
from __future__ import annotations

import queue
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from fsd.core.config import Config
from fsd.core.logger import get
from fsd.core.types import Vec3

try:  # CARLA is an optional runtime dep — the module must import without it.
    import carla
except Exception:  # pragma: no cover - carla missing or broken egg
    carla = None  # type: ignore[assignment]

log = get("carla_bridge.sensors")

#: Name keys used by latest_frame()/snapshot() for streaming sensors.
STREAM_SENSORS = ("camera_rgb", "camera_sem", "lidar", "radar", "gnss", "imu")
#: Event-type sensors: discrete events accumulated between ticks.
EVENT_SENSORS = ("collision", "lane_invasion")


@dataclass
class SensorReading:
    """One parsed sensor frame.

    ``data`` carries the canonical payload per sensor kind:

        camera_rgb   -> HxWx3 uint8 RGB array
        camera_sem   -> HxW uint8 semantic label array (CARLA tag ids)
        lidar        -> Nx4 float32 point cloud (x, y, z, intensity)
        radar        -> Nx4 float32 (altitude, azimuth, depth, velocity)
        gnss         -> dict(latitude, longitude, altitude)
        imu          -> dict(accel=Vec3, gyro=Vec3, compass=float)
    """
    name: str
    frame: int
    timestamp: float               # sim seconds (sensor data timestamp)
    data: Any
    extra: Dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.time)  # wall clock


@dataclass
class SensorEvent:
    """Discrete event (collision / lane invasion)."""
    name: str
    frame: int
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Raw CARLA payload -> numpy/dict parsers. Pure functions, no CARLA types leak.
# --------------------------------------------------------------------------

def _parse_rgb(image) -> SensorReading:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4)
    rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])  # BGRA -> RGB
    return rgb, {}


def _parse_sem(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4)
    # CARLA writes the semantic tag into the R channel (index 2 of BGRA).
    return np.ascontiguousarray(arr[:, :, 2]), {}


def _parse_lidar(meas):
    pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
    return np.ascontiguousarray(pts), {"channels": meas.channels}


def _parse_radar(meas):
    rows = [[d.altitude, d.azimuth, d.depth, d.velocity] for d in meas]
    if rows:
        data = np.asarray(rows, dtype=np.float32)
    else:
        data = np.zeros((0, 4), dtype=np.float32)
    return data, {}


def _parse_gnss(m):
    return {"latitude": m.latitude, "longitude": m.longitude,
            "altitude": m.altitude}, {}


def _parse_imu(m):
    return {
        "accel": Vec3(m.accelerometer.x, m.accelerometer.y, m.accelerometer.z),
        "gyro": Vec3(m.gyroscope.x, m.gyroscope.y, m.gyroscope.z),
        "compass": m.compass,
    }, {}


def _parse_collision(evt):
    other = evt.other_actor
    imp = evt.normal_impulse
    return {
        "other_actor_id": other.id if other is not None else -1,
        "other_actor_type": other.type_id if other is not None else "unknown",
        "impulse": Vec3(imp.x, imp.y, imp.z),
    }, {}


def _parse_lane(evt):
    return {
        "markings": [
            {"type": str(m.type), "color": str(m.color),
             "lane_change": str(m.lane_change)}
            for m in evt.crossed_lane_markings
        ]
    }, {}


def _try_set(bp, key: str, value) -> None:
    """set_attribute that tolerates missing attrs across CARLA versions."""
    try:
        if bp.has_attribute(key):
            bp.set_attribute(key, str(value))
    except Exception as exc:  # pragma: no cover - version-dependent attrs
        log.debug("blueprint attr %s rejected: %s", key, exc)


class SensorSuite:
    """Owns every sensor actor attached to the ego vehicle.

    Usage::

        suite = SensorSuite(cfg)
        suite.attach(world, ego_actor)
        ...
        frame = suite.latest_frame("camera_rgb")     # newest SensorReading
        events = suite.poll_events()                  # {name: [SensorEvent]}
        suite.destroy()
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._actors: Dict[str, Any] = {}
        self._queues: Dict[str, "queue.Queue[SensorReading]"] = {
            n: queue.Queue() for n in STREAM_SENSORS
        }
        self._latest: Dict[str, SensorReading] = {}
        self._events: Dict[str, Deque[SensorEvent]] = {
            n: deque(maxlen=256) for n in EVENT_SENSORS
        }
        self._counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ spec

    def _sensor_specs(self) -> Dict[str, Dict[str, Any]]:
        """Blueprint ids, spawn transforms and attributes per sensor.

        Built lazily because the transforms need a live ``carla`` module.
        """
        c = carla
        veh = self.cfg.vehicle
        cam_tf = c.Transform(c.Location(x=1.6, z=1.7), c.Rotation(pitch=-8.0))
        hz = max(1.0, float(veh.camera_hz))
        return {
            "camera_rgb": dict(
                bp="sensor.camera.rgb", tf=cam_tf, parser=_parse_rgb,
                attrs={"image_size_x": 1280, "image_size_y": 720, "fov": 90,
                       "sensor_tick": round(1.0 / hz, 4)}),
            "camera_sem": dict(
                bp="sensor.camera.semantic_segmentation", tf=cam_tf,
                parser=_parse_sem,
                attrs={"image_size_x": 1280, "image_size_y": 720, "fov": 90,
                       "sensor_tick": round(1.0 / hz, 4)}),
            "lidar": dict(
                bp="sensor.lidar.ray_cast",
                tf=c.Transform(c.Location(x=0.0, z=2.0)),
                parser=_parse_lidar,
                attrs={"channels": veh.lidar_channels, "range": 120.0,
                       "rotation_frequency": hz,
                       "points_per_second": veh.lidar_channels * 20000,
                       "upper_fov": 15.0, "lower_fov": -25.0,
                       "sensor_tick": round(1.0 / hz, 4)}),
            "radar": dict(
                bp="sensor.other.radar",
                tf=c.Transform(c.Location(x=2.2, z=0.6)),
                parser=_parse_radar,
                attrs={"horizontal_fov": 30.0, "vertical_fov": 10.0,
                       "range": 100.0, "points_per_second": 1500,
                       "sensor_tick": round(1.0 / max(1.0, float(veh.radar_hz)), 4)}),
            "gnss": dict(
                bp="sensor.other.gnss",
                tf=c.Transform(c.Location(x=0.0, z=1.8)),
                parser=_parse_gnss,
                attrs={"sensor_tick": round(1.0 / hz, 4)}),
            "imu": dict(
                bp="sensor.other.imu",
                tf=c.Transform(c.Location(x=0.0, z=1.8)),
                parser=_parse_imu,
                attrs={"sensor_tick": round(1.0 / hz, 4)}),
            "collision": dict(
                bp="sensor.other.collision", tf=c.Transform(),
                parser=_parse_collision, attrs={}),
            "lane_invasion": dict(
                bp="sensor.other.lane_invasion", tf=c.Transform(),
                parser=_parse_lane, attrs={}),
        }

    # ---------------------------------------------------------------- attach

    def attach(self, world, vehicle) -> None:
        """Spawn + listen on every sensor, attached to ``vehicle``.

        ``world`` is the :class:`~fsd.carla_bridge.world.CarlaWorld` wrapper
        (its ``spawn_actor`` honours attach_to). Must be called while the
        server can process spawn commands (i.e. before the first tick when
        running synchronously).
        """
        if carla is None:
            raise RuntimeError(
                "SensorSuite.attach requires the 'carla' package "
                "(CARLA simulator Python API) — not importable.")

        lib = world.blueprint_library
        for name, spec in self._sensor_specs().items():
            try:
                bp = lib.find(spec["bp"])
            except (RuntimeError, IndexError):
                log.warning("blueprint %s not found — sensor %s disabled",
                            spec["bp"], name)
                continue
            for k, v in spec["attrs"].items():
                _try_set(bp, k, v)
            actor = world.spawn_actor(bp, spec["tf"], attach_to=vehicle)
            if actor is None:
                log.warning("spawn failed for sensor %s", name)
                continue
            self._actors[name] = actor
            self._counts[name] = 0
            actor.listen(self._make_callback(name, spec["parser"]))
        log.info("sensor suite attached: %s", sorted(self._actors))

    def _make_callback(self, name: str,
                       parser: Callable[[Any], tuple]) -> Callable[[Any], None]:
        def _cb(msg) -> None:
            try:
                data, extra = parser(msg)
            except Exception as exc:
                log.debug("parse failed for %s: %s", name, exc)
                return
            self._counts[name] = self._counts.get(name, 0) + 1
            if name in self._events:
                self._events[name].append(SensorEvent(
                    name=name, frame=getattr(msg, "frame", -1),
                    timestamp=getattr(msg, "timestamp", time.time()),
                    data=data))
            else:
                self._queues[name].put(SensorReading(
                    name=name, frame=getattr(msg, "frame", -1),
                    timestamp=getattr(msg, "timestamp", time.time()),
                    data=data, extra=extra))
        return _cb

    # ---------------------------------------------------------------- access

    def latest_frame(self, name: str) -> Optional[SensorReading]:
        """Drain the sensor queue and return the freshest reading (or None)."""
        q = self._queues.get(name)
        if q is None:
            return self._latest.get(name)
        newest: Optional[SensorReading] = None
        while True:
            try:
                newest = q.get_nowait()
            except queue.Empty:
                break
        if newest is not None:
            self._latest[name] = newest
        return self._latest.get(name)

    def snapshot(self) -> Dict[str, Optional[SensorReading]]:
        """Latest reading of every streaming sensor."""
        return {n: self.latest_frame(n) for n in STREAM_SENSORS}

    def poll_events(self, name: Optional[str] = None
                    ) -> Dict[str, List[SensorEvent]]:
        """Return queued discrete events since the last call and clear them."""
        names = [name] if name else list(EVENT_SENSORS)
        out: Dict[str, List[SensorEvent]] = {}
        for n in names:
            dq = self._events.get(n)
            if dq is None:
                out[n] = []
                continue
            out[n] = list(dq)
            dq.clear()
        return out

    def count(self, name: str) -> int:
        """Total callbacks received for a sensor (health/watchdog)."""
        return self._counts.get(name, 0)

    def frame_age_s(self, name: str) -> Optional[float]:
        """Wall-clock age of the freshest reading — None if never seen."""
        r = self._latest.get(name)
        return None if r is None else time.time() - r.received_at

    # Convenience accessors -------------------------------------------------

    def camera_rgb(self) -> Optional[SensorReading]:
        return self.latest_frame("camera_rgb")

    def camera_semantic(self) -> Optional[SensorReading]:
        return self.latest_frame("camera_sem")

    def lidar(self) -> Optional[SensorReading]:
        return self.latest_frame("lidar")

    def radar(self) -> Optional[SensorReading]:
        return self.latest_frame("radar")

    def gnss(self) -> Optional[SensorReading]:
        return self.latest_frame("gnss")

    def imu(self) -> Optional[SensorReading]:
        return self.latest_frame("imu")

    # ---------------------------------------------------------------- teardown

    def destroy(self) -> None:
        for name, actor in list(self._actors.items()):
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception as exc:
                log.debug("destroy %s failed: %s", name, exc)
        self._actors.clear()
        log.info("sensor suite destroyed")
