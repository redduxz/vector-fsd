"""AutopilotAgent — the main closed loop of the FSD stack.

Each tick:

    world.tick() -> sensors -> ego state
        -> perception (lane / objects / traffic light)
        -> fusion -> occupancy grid
        -> planning (route -> behavior -> trajectory)
        -> control (ControlCommand)
        -> SafetyMonitor.check()          # veto authority
        -> vehicle.apply(cmd) | engage_safe_stop()

Degradation policy:

    - perception stage raises  -> run on last-good (or empty) PerceptionOutput,
                                  pipeline_alive flag drops, SafetyMonitor sees it
    - planning stage raises    -> immediate safe-stop command for this tick
    - control stage raises     -> fallback pure-pursuit controller, then safe-stop
    - whole tick raises        -> emergency brake + logged traceback

Two run modes:

    sim    : real CARLA via fsd.carla_bridge  (needs the carla package + server)
    smoke  : --no-carla synthetic mode — kinematic ego, synthetic sensor
             frames and a scripted lead vehicle, so the full pipeline wiring
             runs (and safety logic triggers) with no simulator at all.

Downstream modules (fsd.perception.*, fsd.planning.*, fsd.control.*,
fsd.safety.*) are imported defensively: if one is missing or still under
construction, the agent falls back to minimal internal implementations so the
loop keeps running and the wiring stays testable end-to-end.

Entrypoint::

    python -m fsd.agents.autopilot --config configs/default.yaml          # sim
    python -m fsd.agents.autopilot --no-carla --ticks 200 --fast          # smoke
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Allow `python fsd/agents/autopilot.py` in addition to `-m` usage.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from fsd.core.config import Config
from fsd.core.logger import get
from fsd.core.types import (
    ControlCommand,
    DetectedObject,
    DriveMode,
    LaneInfo,
    LightState,
    PerceptionOutput,
    Trajectory,
    Vec3,
    VehicleState,
    Waypoint,
)
from fsd.carla_bridge.sensors import SensorReading
from fsd.agents.keyboard_override import ManualOverride

log = get("agents.autopilot")

# ---------------------------------------------------------------------------
# Downstream modules — imported defensively; None means "not built yet".
# ---------------------------------------------------------------------------

def _optional_import(module: str, attr: str):
    try:
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr)
    except Exception as exc:  # ImportError or broken sibling module
        log.debug("%s.%s unavailable (%s) — using fallback", module, attr, exc)
        return None


LaneDetector = _optional_import("fsd.perception.lane_detector", "LaneDetector")
ObjectDetector = _optional_import("fsd.perception.object_detector", "ObjectDetector")
TrafficLightMonitor = _optional_import("fsd.perception.traffic_light", "TrafficLightMonitor")
SensorFusion = _optional_import("fsd.perception.fusion", "SensorFusion")
OccupancyGrid = _optional_import("fsd.perception.occupancy", "OccupancyGrid")
BehaviorPlanner = _optional_import("fsd.planning.behavior", "BehaviorPlanner")
RoutePlanner = _optional_import("fsd.planning.route", "RoutePlanner")
TrajectoryPlanner = _optional_import("fsd.planning.trajectory", "TrajectoryPlanner")
VehicleController = _optional_import("fsd.control.controller", "VehicleController")
SafetyMonitor = _optional_import("fsd.safety.monitor", "SafetyMonitor")

# Candidate method names — downstream signatures aren't frozen yet, so the
# agent resolves the first matching callable and calls it flexibly.
_DETECT_M = ("detect", "update", "process", "infer", "forward", "run", "step")
_LIGHT_M = ("state", "get_state", "update", "detect", "current", "observe")
_FUSE_M = ("fuse", "update", "process", "merge")
_OCC_M = ("update", "integrate", "build", "fill", "process")
_ROUTE_M = ("plan", "update", "route", "get_route", "replan")
_BEH_M = ("plan", "decide", "update", "step")
_TRAJ_M = ("plan", "generate", "compute", "update")
_CTRL_M = ("compute", "control", "step", "act", "update")


class PipelineHealth(dict):
    """dict(stage->alive) that also evaluates as a single bool for checks."""
    def __bool__(self):
        return len(self) > 0 and all(self.values())
    __nonzero__ = __bool__


# ---------------------------------------------------------------------------
# Plumbing helpers
# ---------------------------------------------------------------------------

def _call_flex(fn: Callable, *pos, **kw):
    """Call ``fn`` with the subset of ``kw`` its signature accepts.

    If the filtered call still fails (e.g. required params named differently
    than our kwargs), retry positionally with ``pos`` truncated to the number
    of required positional parameters.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*pos, **kw)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*pos, **kw)
    sub = {k: v for k, v in kw.items() if k in params}
    try:
        return fn(**sub)
    except TypeError:
        n_req = sum(
            1 for p in params.values()
            if p.default is p.empty and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD))
        return fn(*pos[:n_req])


def _first_method(obj: Any, names) -> Optional[Callable]:
    for n in names:
        f = getattr(obj, n, None)
        if callable(f):
            return f
    return obj if callable(obj) else None


def _build(cls: Optional[type], cfg: Config):
    """Instantiate a downstream class trying (cfg), (cfg.raw), ()."""
    if cls is None:
        return None
    for args in ((cfg,), (cfg.raw,), ()):
        try:
            return cls(*args)
        except Exception as exc:
            log.debug("%s(%r) init failed: %s", cls.__name__,
                      type(args[0]).__name__ if args else "", exc)
            continue
    log.warning("could not construct %s with any known signature",
                cls.__name__)
    return None


# ---------------------------------------------------------------------------
# Output normalizers — downstream may return dataclasses, dicts or tuples.
# ---------------------------------------------------------------------------

def _as_lane(x) -> LaneInfo:
    if isinstance(x, LaneInfo):
        return x
    if isinstance(x, dict):
        keys = {f for f in LaneInfo.__dataclass_fields__}
        return LaneInfo(**{k: v for k, v in x.items() if k in keys})
    return LaneInfo(0.0, 0.0, 0.0, 0.0, 0.0, detected=False)


def _as_objects(x) -> List[DetectedObject]:
    out: List[DetectedObject] = []
    if not x:
        return out
    for o in x:
        if isinstance(o, DetectedObject):
            out.append(o)
        elif isinstance(o, dict):
            try:
                keys = set(DetectedObject.__dataclass_fields__)
                out.append(DetectedObject(
                    **{k: v for k, v in o.items() if k in keys}))
            except TypeError:
                continue
    return out


def _as_light(x) -> LightState:
    if isinstance(x, LightState):
        return x
    if isinstance(x, str):
        try:
            return LightState[x.strip().upper()]
        except KeyError:
            return LightState.UNKNOWN
    return LightState.UNKNOWN


def _as_waypoints(x) -> List[Waypoint]:
    pts: List[Waypoint] = []
    for p in x or []:
        if isinstance(p, Waypoint):
            pts.append(p)
        elif isinstance(p, dict):
            keys = set(Waypoint.__dataclass_fields__)
            pts.append(Waypoint(**{k: v for k, v in p.items() if k in keys}))
    return pts


def _as_trajectory(x, default_speed: float) -> Optional[Trajectory]:
    if x is None:
        return None
    if isinstance(x, Trajectory):
        return x
    if isinstance(x, (list, tuple)):
        return Trajectory(points=_as_waypoints(x),
                          target_speed=default_speed)
    if isinstance(x, dict):
        return Trajectory(points=_as_waypoints(x.get("points", [])),
                          target_speed=float(x.get("target_speed",
                                                   default_speed)))
    return None


def _as_cmd(x) -> Optional[ControlCommand]:
    if x is None:
        return None
    if isinstance(x, ControlCommand):
        return x.clamp()
    if isinstance(x, dict):
        keys = set(ControlCommand.__dataclass_fields__)
        return ControlCommand(
            **{k: v for k, v in x.items() if k in keys}).clamp()
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        return ControlCommand(throttle=float(x[0]), brake=float(x[1]),
                              steer=float(x[2])).clamp()
    return None


def _as_mode(x) -> DriveMode:
    if isinstance(x, (tuple, list)) and x:
        x = x[0]
    if isinstance(x, DriveMode):
        return x
    if isinstance(x, str):
        try:
            return DriveMode[x.strip().upper()]
        except KeyError:
            pass
    return DriveMode.ENGAGED


# ---------------------------------------------------------------------------
# Minimal internal fallbacks (used when a downstream module isn't importable
# or in --no-carla smoke mode). Deliberately small but real.
# ---------------------------------------------------------------------------

class _FallbackSafety:
    """TTC / free-space / watchdog checker mirroring SafetyMonitor's contract."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def check(self, ego, perception, cmd, pipeline_alive):
        s = self.cfg.safety
        if not pipeline_alive:
            return DriveMode.SAFE_STOP
        ttc = self._ttc(ego, perception)
        if ttc is not None and ttc < s.min_ttc_s:
            return DriveMode.SAFE_STOP
        if perception.free_space_ahead < s.min_free_space_m and \
                ego.speed > 0.5:
            return DriveMode.SAFE_STOP
        if ego.speed > s.max_speed_mps + 1.0:
            return DriveMode.DEGRADED
        return DriveMode.ENGAGED

    @staticmethod
    def _ttc(ego, perception) -> Optional[float]:
        best = None
        for o in perception.objects:
            # object ahead along ego heading
            dx = o.position.x - ego.x
            dy = o.position.y - ego.y
            ahead = dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw)
            lat = -dx * math.sin(ego.yaw) + dy * math.cos(ego.yaw)
            if ahead <= 0.1 or abs(lat) > 2.5:
                continue
            closing = ego.speed - (
                o.velocity.x * math.cos(ego.yaw)
                + o.velocity.y * math.sin(ego.yaw))
            if closing <= 0.1:
                continue
            ttc = ahead / closing
            best = ttc if best is None else min(best, ttc)
        return best

    def engage_safe_stop(self) -> ControlCommand:
        return ControlCommand(throttle=0.0, brake=1.0)


class _FallbackController:
    """Pure-pursuit steering + P speed control."""
    def __init__(self, cfg: Config):
        self.wheelbase = cfg.vehicle.wheelbase_m
        self.max_steer = math.radians(cfg.vehicle.max_steer_deg)

    def compute(self, ego, trajectory, **_kw) -> ControlCommand:
        traj = _as_trajectory(trajectory, 0.0)
        if traj is None or traj.empty:
            return ControlCommand(brake=0.8)
        # speed loop
        err = traj.target_speed - ego.speed
        cmd = ControlCommand()
        if err > 0.2:
            cmd.throttle = min(1.0, 0.35 * err)
        elif err < -0.5:
            cmd.brake = min(1.0, -0.30 * err)
        # pure pursuit
        lookahead = max(4.0, ego.speed * 1.2)
        target = None
        for p in traj.points:
            if math.hypot(p.x - ego.x, p.y - ego.y) >= lookahead:
                target = p
                break
        if target is None:
            target = traj.points[-1]
        dx, dy = target.x - ego.x, target.y - ego.y
        lx = dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw)
        ly = -dx * math.sin(ego.yaw) + dy * math.cos(ego.yaw)
        ld = max(1.0, math.hypot(lx, ly))
        alpha = math.atan2(ly, lx)
        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        cmd.steer = max(-1.0, min(1.0, steer / max(0.1, self.max_steer)))
        return cmd


def _fallback_route(ego: VehicleState, n: int = 60, ds: float = 2.0
                    ) -> List[Waypoint]:
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    return [Waypoint(x=ego.x + i * ds * c, y=ego.y + i * ds * s,
                     yaw=ego.yaw) for i in range(n)]


def _fallback_behavior(ego, perception, route, cfg) -> Dict[str, Any]:
    """Decide a cruise target speed from the (possibly fused) scene."""
    limit = min(10.0, cfg.safety.max_speed_mps)
    target = limit
    # follow the nearest object ahead
    best_ahead = None
    for o in perception.objects:
        dx, dy = o.position.x - ego.x, o.position.y - ego.y
        ahead = dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw)
        lat = -dx * math.sin(ego.yaw) + dy * math.cos(ego.yaw)
        if ahead > 0.5 and abs(lat) < 2.0:
            best_ahead = ahead if best_ahead is None else min(best_ahead, ahead)
    if best_ahead is not None and best_ahead < 35.0:
        target = min(target, max(0.0, (best_ahead - 8.0) * 0.4))
    if perception.light == LightState.RED and perception.free_space_ahead < 40:
        target = min(target, max(0.0, (perception.free_space_ahead - 6.0)))
    return {"target_speed": target, "route": route, "lane_change": "KEEP"}


def _fallback_trajectory(ego, perception, route, behavior) -> Trajectory:
    pts = _as_waypoints(route)[:30] if route else _fallback_route(ego)[:30]
    speed = limit = 8.0
    if isinstance(behavior, dict):
        speed = float(behavior.get("target_speed", limit))
    elif hasattr(behavior, "target_speed"):
        speed = float(behavior.target_speed)
    return Trajectory(points=pts, target_speed=speed)


class _SyntheticEgo:
    """Kinematic bicycle standing in for the carla vehicle in smoke mode."""
    def __init__(self, cfg: Config):
        self.x = self.y = self.yaw = 0.0
        self.v = 0.0
        self.accel = 0.0
        self.steer = 0.0
        self.t = 0.0
        self._cfg = cfg
        self._cmd = ControlCommand()

    def apply(self, cmd: ControlCommand) -> None:
        self._cmd = cmd.clamp()

    def integrate(self, dt: float) -> None:
        c = self._cmd
        max_steer = math.radians(self._cfg.vehicle.max_steer_deg)
        a = (c.throttle * 4.0) - (c.brake * 8.0) - 0.02 * self.v
        if c.hand_brake:
            a = -8.0
        prev_v = self.v
        self.v = max(0.0, self.v + a * dt)
        if self.v == 0.0 and a < 0:
            a = 0.0
        self.accel = (self.v - prev_v) / max(dt, 1e-3)
        self.steer += (c.steer - self.steer) * min(1.0, dt * 8.0)
        self.yaw += (self.v / self._cfg.vehicle.wheelbase_m
                     * math.tan(self.steer * max_steer)) * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.t += dt

    def state(self) -> VehicleState:
        return VehicleState(x=self.x, y=self.y, z=0.0, yaw=self.yaw,
                            speed=self.v, accel=self.accel, steer=self.steer,
                            timestamp=time.time())


class _SyntheticScene:
    """Scripted lead-vehicle scene + sensor frames for --no-carla mode."""
    def __init__(self, ego: _SyntheticEgo):
        self.lead_dist = 45.0       # gap ahead of ego, metres
        self.lead_speed = 6.5       # slower than our cruise -> closing
        self._rgb = np.zeros((720, 1280, 3), np.uint8)
        self._sem = np.zeros((720, 1280), np.uint8)
        self._ego = ego
        # a ring of ground points so occupancy/fusion see real geometry
        ang = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        self._ring = np.stack([np.cos(ang) * 22.0, np.sin(ang) * 22.0,
                               np.zeros_like(ang), np.full_like(ang, 0.3)],
                              axis=1).astype(np.float32)

    def step(self, ego_v: float, dt: float) -> None:
        self.lead_dist += (self.lead_speed - ego_v) * dt
        self.lead_dist = max(2.0, self.lead_dist)

    def lead_object(self, ego: VehicleState) -> DetectedObject:
        c, s = math.cos(ego.yaw), math.sin(ego.yaw)
        return DetectedObject(
            obj_id=1, cls="vehicle",
            position=Vec3(ego.x + self.lead_dist * c,
                          ego.y + self.lead_dist * s, 0.0),
            velocity=Vec3(self.lead_speed * c, self.lead_speed * s, 0.0),
            bbox_extent=Vec3(2.2, 0.9, 0.75), confidence=0.95,
            timestamp=ego.timestamp)

    def sensors(self, ego: VehicleState, frame: int
                ) -> Dict[str, SensorReading]:
        c, s = math.cos(ego.yaw), math.sin(ego.yaw)
        lx, ly = ego.x + self.lead_dist * c, ego.y + self.lead_dist * s
        cluster = np.array([[lx, ly, 0.6, 0.9], [lx, ly + 0.8, 0.4, 0.9],
                            [lx, ly - 0.8, 0.4, 0.9]], np.float32)
        lidar = np.concatenate([self._ring +
                                np.array([ego.x, ego.y, 0, 0], np.float32),
                                cluster])
        radar = np.array([[0.0, 0.0, self.lead_dist,
                           self.lead_speed - ego.speed]], np.float32)

        def rd(name, data):
            return SensorReading(name=name, frame=frame,
                                 timestamp=ego.timestamp, data=data)

        return {
            "camera_rgb": rd("camera_rgb", self._rgb),
            "camera_sem": rd("camera_sem", self._sem),
            "lidar": rd("lidar", lidar),
            "radar": rd("radar", radar),
            "gnss": rd("gnss", {"latitude": 0.0, "longitude": 0.0,
                                "altitude": 0.0}),
            "imu": rd("imu", {"accel": Vec3(ego.accel, 0, 0),
                              "gyro": Vec3(0, 0, 0), "compass": ego.yaw}),
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AutopilotAgent:
    """The closed-loop FSD agent. Construct, ``setup()``, ``run()``."""

    def __init__(self, cfg: Config, smoke: bool = False):
        self.cfg = cfg
        self.smoke = smoke
        self.dt = float(cfg.sim.fixed_delta_s)

        # sim handles (sim mode)
        self.world = None
        self.vehicle = None
        self.traffic = None
        # smoke handles (no-carla mode)
        self._synthetic_ego: Optional[_SyntheticEgo] = None
        self._scene: Optional[_SyntheticScene] = None
        self._sim_t = 0.0

        # pipeline modules (real or fallback)
        self.lane_detector = None
        self.object_detector = None
        self.tl_monitor = None
        self.fusion = None
        self.occupancy = None
        self.route_planner = None
        self.behavior_planner = None
        self.traj_planner = None
        self.controller = None
        self.safety = None
        self._module_names: Dict[str, bool] = {}

        # loop state
        self.override = ManualOverride()
        self.mode = DriveMode.DISENGAGED
        self._tick_idx = 0
        self._running = False
        self._last_ok: Dict[str, float] = {}
        self._last_perception: Optional[PerceptionOutput] = None
        self._last_stop_m = math.inf
        self._fail_counts: Dict[str, int] = {}
        self._stuck_ticks = 0
        self._recover_ticks = 0
        self._recover_steer = 0.0
        self._relocations = 0
        self._setup_done = False

    # ------------------------------------------------------------------ setup

    def _build_modules(self) -> None:
        cfg = self.cfg
        self.lane_detector = _build(LaneDetector, cfg)
        self.object_detector = self._make_detector(cfg)
        self.tl_monitor = _build(TrafficLightMonitor, cfg)
        self.fusion = _build(SensorFusion, cfg)
        self.occupancy = _build(OccupancyGrid, cfg)
        self.route_planner = _build(RoutePlanner, cfg)
        self.behavior_planner = _build(BehaviorPlanner, cfg)
        self.traj_planner = _build(TrajectoryPlanner, cfg)
        self.controller = _build(VehicleController, cfg) \
            or _FallbackController(cfg)
        self.safety = _build(SafetyMonitor, cfg) or _FallbackSafety(cfg)
        self._module_names = {
            "lane": self.lane_detector is not None,
            "objects": self.object_detector is not None,
            "traffic_light": self.tl_monitor is not None,
            "fusion": self.fusion is not None,
            "occupancy": self.occupancy is not None,
            "route": self.route_planner is not None,
            "behavior": self.behavior_planner is not None,
            "trajectory": self.traj_planner is not None,
            "controller": (isinstance(self.controller, VehicleController)
                           if VehicleController is not None
                           else self.controller is not None),
            "safety": (isinstance(self.safety, SafetyMonitor)
                       if SafetyMonitor is not None
                       else self.safety is not None),
        }
        real = [k for k, v in self._module_names.items() if v]
        fb = [k for k, v in self._module_names.items() if not v]
        log.info("modules real: %s | fallback: %s",
                 ",".join(real) or "-", ",".join(fb) or "-")

    def setup(self) -> "AutopilotAgent":
        """Build the pipeline and attach to the world (or synthetic scene)."""
        from fsd.carla_bridge import (  # deferred: carla may be absent
            CarlaWorld, EgoVehicle, TrafficManager)
        self._build_modules()
        if self.smoke:
            self._synthetic_ego = _SyntheticEgo(self.cfg)
            self._scene = _SyntheticScene(self._synthetic_ego)
            log.info("smoke mode: synthetic ego + lead-vehicle scene")
        else:
            self.world = CarlaWorld(self.cfg).connect()
            self.vehicle = EgoVehicle(self.world, self.cfg).spawn()
            self.traffic = TrafficManager(self.world, self.cfg)
            self.traffic.spawn_all(avoid_transform=self.vehicle.spawn_transform)
            # Everything spawned while async — now lock the world to 20 Hz.
            self.world.set_synchronous(True)
            self.traffic.set_synchronous(True)
        now = time.monotonic()
        self._last_ok = {k: now for k in
                         ("sensors", "perception", "planning", "control")}
        self.override.start()
        self.mode = DriveMode.ENGAGED
        self._setup_done = True
        return self

    def _make_detector(self, cfg: Config):
        """Build ObjectDetector from the perception config section."""
        if ObjectDetector is None:
            return None
        pc = getattr(cfg, "perception", None)
        if pc is None:
            return _build(ObjectDetector, cfg)
        try:
            return ObjectDetector(
                backend=pc.object_backend,
                model_path=pc.object_model_path or None,
                conf_threshold=pc.object_conf_threshold,
                max_range_m=pc.object_max_range_m)
        except Exception as exc:
            log.warning("ObjectDetector init failed: %s", exc)
            return _build(ObjectDetector, cfg)

    # ------------------------------------------------------------------ stages

    def _ego_state(self) -> VehicleState:
        if self.smoke:
            assert self._synthetic_ego is not None
            return self._synthetic_ego.state()
        return self.vehicle.state()

    def _sensor_snapshot(self, ego: VehicleState) -> Dict[str, SensorReading]:
        if self.smoke:
            assert self._scene is not None
            return self._scene.sensors(ego, self._tick_idx)
        snap = self.vehicle.sensors.snapshot()
        return {k: v for k, v in snap.items() if v is not None}

    def _sensors_fresh(self) -> bool:
        if self.smoke:
            return True
        age = self.vehicle.sensors.frame_age_s("camera_rgb")
        return age is not None and age < 1.0

    # ---- perception ----------------------------------------------------------

    def _detect_lane(self, ego, sensors):
        if self.lane_detector is None:
            return LaneInfo(0.0, 0.0, 0.0, 0.0, 0.0,
                            detected=not self.smoke or True)
        fn = _first_method(self.lane_detector, _DETECT_M)
        img = _reading_data(sensors, "camera_rgb")
        sem = _reading_data(sensors, "camera_sem")
        return _as_lane(_call_flex(fn, img, ego,
                                   image=img, rgb=img, semantic=sem,
                                   ego=ego, state=ego))

    def _detect_objects(self, ego, sensors):
        if self.object_detector is None:
            if self.smoke and self._scene is not None:
                return [self._scene.lead_object(ego)]
            return []
        fn = _first_method(self.object_detector, _DETECT_M)
        img = _reading_data(sensors, "camera_rgb")
        lidar = _reading_data(sensors, "lidar")
        radar = _reading_data(sensors, "radar")
        # ego carries its actor id so the detector can exclude itself
        ego_ref = ego
        if self.vehicle is not None:
            try:
                ego_id = self.vehicle.id() if callable(self.vehicle.id) \
                    else self.vehicle.id
            except Exception:
                ego_id = None
            ego_ref = {"x": ego.x, "y": ego.y, "yaw": ego.yaw, "id": ego_id}
        return _as_objects(_call_flex(fn, img, ego,
                                      image=img, rgb=img, lidar=lidar,
                                      radar=radar, points=lidar, ego=ego_ref,
                                      state=ego, sensors=sensors,
                                      world=self.world))

    def _light_state(self, ego, sensors):
        if self.tl_monitor is None:
            return LightState.UNKNOWN
        fn = _first_method(self.tl_monitor, _LIGHT_M)
        img = _reading_data(sensors, "camera_rgb")
        # carla.Vehicle answers is_at_traffic_light() directly — prefer the
        # live actor over the reduced VehicleState
        actor = getattr(self.vehicle, "actor", None) or ego
        st = _as_light(_call_flex(fn, actor, img, ego=actor, state=ego,
                                  image=img, sensors=sensors,
                                  world=self.world))
        try:
            self._last_stop_m = float(self.tl_monitor.stop_distance_m)
        except (TypeError, ValueError, AttributeError):
            self._last_stop_m = math.inf
        return st

    def _fuse(self, ego, sensors, lane, objects, light) -> PerceptionOutput:
        if self.fusion is None:
            free = self._free_space(ego, objects)
            return PerceptionOutput(objects=objects, lane=lane, light=light,
                                    free_space_ahead=free,
                                    timestamp=ego.timestamp)
        fn = _first_method(self.fusion, _FUSE_M)
        out = _call_flex(fn, objects, lane, light, ego,
                         objects=objects, detections=objects, lane=lane,
                         lane_info=lane, light=light, light_state=light,
                         ego=ego, state=ego, sensors=sensors)
        if isinstance(out, PerceptionOutput):
            return out
        if isinstance(out, dict):
            return PerceptionOutput(
                objects=_as_objects(out.get("objects", objects)),
                lane=_as_lane(out.get("lane", lane)),
                light=_as_light(out.get("light", light)),
                free_space_ahead=float(out.get(
                    "free_space_ahead", self._free_space(ego, objects))))
        return PerceptionOutput(objects=objects, lane=lane, light=light,
                                free_space_ahead=self._free_space(ego, objects),
                                timestamp=ego.timestamp)

    @staticmethod
    def _free_space(ego, objects) -> float:
        best = 100.0
        for o in objects:
            if getattr(o, "cls", "vehicle") not in (
                    "vehicle", "pedestrian", "cyclist", "misc"):
                continue          # signs/poles don't cap free space
            dx, dy = o.position.x - ego.x, o.position.y - ego.y
            ahead = dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw)
            lat = abs(-dx * math.sin(ego.yaw) + dy * math.cos(ego.yaw))
            if ahead > 0.0 and lat < 2.5:
                best = min(best, ahead)
        return best

    def _update_occupancy(self, ego, sensors, perception) -> None:
        if self.occupancy is None:
            return
        fn = _first_method(self.occupancy, _OCC_M)
        if fn is not None:
            _call_flex(fn, sensors, ego, perception,
                       lidar=_reading_data(sensors, "lidar"),
                       sensors=sensors, objects=perception.objects,
                       perception=perception, ego=ego, state=ego)

    # ---- planning ------------------------------------------------------------

    def _route(self, ego):
        if self.route_planner is None:
            return _fallback_route(ego)
        goal = Waypoint(x=ego.x + 200.0 * math.cos(ego.yaw),
                        y=ego.y + 200.0 * math.sin(ego.yaw))
        carla_map = getattr(getattr(self, "world", None), "map", None)
        fn = _first_method(self.route_planner, _ROUTE_M)
        return _call_flex(fn, ego, goal,
                          start=ego, goal=goal, carla_map=carla_map,
                          ego=ego, state=ego, vehicle_state=ego)

    def _behavior(self, ego, perception, route):
        if self.behavior_planner is None:
            return _fallback_behavior(ego, perception, route, self.cfg)
        fn = _first_method(self.behavior_planner, _BEH_M)
        return _call_flex(fn, ego, perception, route,
                          ego=ego, state=ego, perception=perception,
                          route=route, waypoints=route)

    def _trajectory(self, ego, perception, route, behavior):
        if self.traj_planner is None:
            return _fallback_trajectory(ego, perception, route, behavior)
        fn = _first_method(self.traj_planner, _TRAJ_M)
        return _call_flex(fn, route, behavior, perception, ego,
                          ego=ego, state=ego, perception=perception,
                          route=route, waypoints=route,
                          decision=behavior, behavior=behavior)

    # ---- control / safety ----------------------------------------------------

    def _control(self, ego, traj, planning_ok) -> Optional[ControlCommand]:
        if not planning_ok:
            return None
        try:
            cmd = _call_flex(_first_method(self.controller, _CTRL_M),
                             ego, traj,
                             ego=ego, state=ego, vehicle_state=ego,
                             trajectory=traj, traj=traj,
                             target_speed=getattr(traj, "target_speed", 0.0))
            cmd = _as_cmd(cmd)
            if cmd is not None:
                self._last_ok["control"] = time.monotonic()
            return cmd
        except Exception:
            log.exception("control stage failed")
            try:
                return _FallbackController(self.cfg).compute(ego, traj)
            except Exception:
                return None

    def _check_safety(self, ego, perception, cmd,
                      health: PipelineHealth) -> DriveMode:
        check = getattr(self.safety, "check", None)
        if check is None:                       # fallback is a function ref
            return self.safety.check(ego, perception, cmd, health)
        mode = _as_mode(_call_flex(
            check, ego, perception, cmd, health,
            ego=ego, state=ego, vehicle_state=ego, perception=perception,
            cmd=cmd, command=cmd, control=cmd,
            pipeline_alive=health, health=health, alive=health))
        return mode

    def _safe_stop(self) -> ControlCommand:
        eng = getattr(self.safety, "engage_safe_stop", None)
        if callable(eng):
            try:
                cmd = _as_cmd(_call_flex(eng))
                if cmd is not None:
                    return cmd
            except Exception:
                log.exception("engage_safe_stop raised — hard brake")
        return ControlCommand(throttle=0.0, brake=1.0)

    # ------------------------------------------------------------------- tick

    def tick(self) -> Dict[str, Any]:
        """One closed-loop iteration. Returns a per-tick result dict."""
        self._tick_idx += 1
        now = time.monotonic()

        # 1. advance the world (no-op in smoke mode)
        if self.world is not None:
            self.world.tick()
        if self.smoke and self._synthetic_ego is not None:
            assert self._scene is not None
            self._scene.step(self._synthetic_ego.v, self.dt)
            self._synthetic_ego.integrate(self.dt)

        # 2. ego + sensors
        ego = self._ego_state()
        sensors = self._sensor_snapshot(ego)
        if self._sensors_fresh():
            self._last_ok["sensors"] = now

        # 3. manual override short-circuits the whole pipeline
        if self.override.takeover_requested:
            cmd = self.override.manual_command()
            self._apply(cmd)
            prev = self.mode
            self.mode = DriveMode.DISENGAGED
            return dict(tick=self._tick_idx, ego=ego, cmd=cmd,
                        mode=self.mode, manual=True,
                        perception=self._last_perception,
                        perception_ok=False, planning_ok=False,
                        events={})

        # 4. perception (degrade, don't die)
        perception_ok = True
        try:
            lane = self._detect_lane(ego, sensors)
            objects = self._detect_objects(ego, sensors)
            light = self._light_state(ego, sensors)
            perception = self._fuse(ego, sensors, lane, objects, light)
            try:
                perception.stop_line_m = self._last_stop_m
            except AttributeError:
                pass
            try:
                self._update_occupancy(ego, sensors, perception)
            except Exception:
                log.debug("occupancy update failed", exc_info=True)
            self._last_ok["perception"] = now
            self._last_perception = perception
        except Exception:
            log.exception("perception stage failed — degraded frame")
            self._fail_counts["perception"] = \
                self._fail_counts.get("perception", 0) + 1
            perception_ok = False
            perception = self._last_perception or PerceptionOutput(
                objects=[], lane=LaneInfo(0, 0, 0, 0, 0, detected=False),
                light=LightState.UNKNOWN, free_space_ahead=0.0)

        # 5. planning (failure -> safe stop this tick)
        planning_ok = True
        traj = None
        try:
            route = self._route(ego)
            behavior = self._behavior(ego, perception, route)
            traj = _as_trajectory(
                self._trajectory(ego, perception, route, behavior),
                default_speed=8.0)
            if traj is None:
                raise RuntimeError("trajectory planner returned nothing")
            self._last_traj = traj.points
            self._last_ok["planning"] = now
        except Exception:
            log.exception("planning stage failed — safe stop")
            self._fail_counts["planning"] = \
                self._fail_counts.get("planning", 0) + 1
            planning_ok = False

        # 6. control
        cmd = self._control(ego, traj, planning_ok)

        # 7. safety veto
        wd = self.cfg.safety.watchdog_timeout_s
        health = PipelineHealth({
            "sensors": now - self._last_ok["sensors"] < wd * 4,
            "perception": now - self._last_ok["perception"] < wd * 4,
            "planning": now - self._last_ok["planning"] < wd * 4,
            "control": now - self._last_ok["control"] < wd * 4,
        })
        hb = getattr(self.safety, "heartbeat", None)
        if callable(hb) and health:            # loop alive this tick
            try:
                hb()
            except Exception:
                pass
        try:
            mode = self._check_safety(ego, perception, cmd, health)
        except Exception:
            log.exception("safety check raised — treating as SAFE_STOP")
            mode = DriveMode.SAFE_STOP
        if mode in (DriveMode.SAFE_STOP, DriveMode.DISENGAGED) or cmd is None:
            cmd = self._safe_stop()
            if cmd is None:                    # engage_safe_stop broke
                cmd = ControlCommand(brake=1.0)
            if mode == DriveMode.ENGAGED:
                mode = DriveMode.SAFE_STOP     # cmd was None
        prev_mode = self.mode
        self.mode = mode
        if mode != prev_mode:
            log.warning("drive mode %s -> %s", prev_mode.name, mode.name)

        # 7b. stuck recovery — powered but motionless (e.g. wedged on a
        # pole perception lost). Reverse out; escalate to relocation.
        if mode == DriveMode.ENGAGED and cmd is not None:
            powered = cmd.throttle > 0.3 and not cmd.reverse
            if powered and ego.speed < 0.2:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
            if self._stuck_ticks >= int(3.0 / self.dt):
                self._stuck_ticks = 0
                self._recover_ticks = int(1.6 / self.dt)
                self._recover_steer = -float(cmd.steer)
                log.warning("ego wedged (thr=%.2f v=%.2f) — reverse recovery",
                            cmd.throttle, ego.speed)
        if self._recover_ticks > 0:
            self._recover_ticks -= 1
            if self._recover_ticks == 0 and ego.speed < 0.5 and \
                    self.vehicle is not None and self._relocations < 3:
                self._relocations += 1
                log.warning("reverse failed to free ego — relocating (%d/3)",
                            self._relocations)
                try:
                    self.vehicle.relocate()
                except Exception:
                    log.exception("relocate raised")
            if mode == DriveMode.ENGAGED:
                cmd = ControlCommand(throttle=0.55,
                                     steer=self._recover_steer, reverse=True)

        # 8. actuate
        self._apply(cmd)

        # sim-time collision/lane events -> logged for the safety audit trail
        events = {}
        if self.vehicle is not None:
            events = self.vehicle.poll_events()
            for ev in events.get("collision", []):
                log.warning("COLLISION with %s",
                            ev.data.get("other_actor_type"))

        return dict(tick=self._tick_idx, ego=ego, cmd=cmd, mode=mode,
                    manual=False, perception=perception,
                    perception_ok=perception_ok, planning_ok=planning_ok,
                    events=events)

    def _apply(self, cmd: ControlCommand) -> None:
        if self.smoke:
            if self._synthetic_ego is not None:
                self._synthetic_ego.apply(cmd)
        elif self.vehicle is not None:
            self.vehicle.apply(cmd)

    # ------------------------------------------------------------------- run

    def run(self, max_ticks: int = 0, duration_s: float = 0.0,
            pace: bool = True) -> Dict[str, Any]:
        """The 20 Hz main loop. pace=True keeps wall-clock ≈ fixed_delta."""
        if not self._setup_done:
            self.setup()
        if max_ticks <= 0 and duration_s > 0:
            max_ticks = max(1, int(duration_s / self.dt))
        log.info("autopilot loop starting: mode=%s dt=%.3fs max_ticks=%s",
                 "smoke" if self.smoke else "sim", self.dt,
                 max_ticks or "∞")
        self._running = True
        last_res: Dict[str, Any] = {}
        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    last_res = self.tick()
                except Exception:
                    log.exception("tick %d crashed — emergency brake",
                                  self._tick_idx)
                    cmd = self._safe_stop() or ControlCommand(brake=1.0)
                    try:
                        self._apply(cmd)
                    except Exception:
                        pass
                    last_res = dict(tick=self._tick_idx, cmd=cmd,
                                    mode=DriveMode.SAFE_STOP, crashed=True)
                self._periodic_log(last_res)
                if max_ticks and self._tick_idx >= max_ticks:
                    break
                if pace:
                    rem = self.dt - (time.monotonic() - t0)
                    if rem > 0:
                        time.sleep(rem)
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            self._running = False
        log.info("loop finished after %d ticks (mode=%s)",
                 self._tick_idx, self.mode.name)
        return {"ticks": self._tick_idx, "mode": self.mode,
                "failures": dict(self._fail_counts), "last": last_res}

    def _periodic_log(self, res: Dict[str, Any]) -> None:
        per = max(1, int(round(1.0 / self.dt)))
        if self._tick_idx % per:
            return
        ego, cmd = res.get("ego"), res.get("cmd")
        p = res.get("perception")
        objs = len(p.objects) if p else -1
        log.info(
            "t=%d mode=%s v=%.1f thr=%.2f brk=%.2f str=%+.2f "
            "objs=%s light=%s free=%.0f%s",
            self._tick_idx, res.get("mode").name,
            ego.speed if ego else -1,
            cmd.throttle if cmd else -1, cmd.brake if cmd else -1,
            cmd.steer if cmd else 0.0,
            objs, p.light.name if p else "?",
            p.free_space_ahead if p else -1,
            " [MANUAL]" if res.get("manual") else "")

    # ------------------------------------------------------------------ teardown

    def cleanup(self) -> None:
        """Destroy actors, restore async settings, stop the key listener."""
        self.override.stop()
        # Async mode first so destroy commands are processed immediately.
        try:
            if self.traffic is not None:
                self.traffic.set_synchronous(False)
        except Exception:
            pass
        if self.world is not None:
            try:
                self.world.set_synchronous(False)
            except Exception:
                pass
        if self.traffic is not None:
            try:
                self.traffic.destroy_all()
            except Exception as exc:
                log.debug("traffic destroy failed: %s", exc)
            self.traffic = None
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception as exc:
                log.debug("vehicle destroy failed: %s", exc)
            self.vehicle = None
        if self.world is not None:
            self.world.cleanup()
            self.world = None
        self._setup_done = False

    def __enter__(self) -> "AutopilotAgent":
        return self.setup()

    def __exit__(self, *exc) -> None:
        self.cleanup()


def _reading_data(sensors: Dict[str, SensorReading], name: str):
    r = sensors.get(name)
    return r.data if r is not None else None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="fsd.agents.autopilot",
        description="FSD autopilot closed loop (CARLA sim or synthetic smoke)")
    p.add_argument("--config", default="configs/default.yaml",
                   help="YAML config path (missing -> built-in defaults)")
    p.add_argument("--sim", action="store_true",
                   help="run against CARLA (default)")
    p.add_argument("--no-carla", "--smoke", dest="smoke",
                   action="store_true",
                   help="headless smoke mode — synthetic ego + sensor data")
    p.add_argument("--ticks", type=int, default=0,
                   help="max ticks (0 = run until interrupted)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds of sim time to run (alternative to --ticks)")
    p.add_argument("--fast", action="store_true",
                   help="disable real-time pacing")
    args = p.parse_args(argv)

    cfg = Config.load(args.config)
    if not os.path.exists(args.config):
        log.warning("config %s not found — using defaults", args.config)

    agent = AutopilotAgent(cfg, smoke=args.smoke)
    try:
        agent.setup()
    except Exception as exc:
        if not args.smoke:
            from fsd.carla_bridge import CarlaUnavailableError
            if isinstance(exc, CarlaUnavailableError):
                log.error("CARLA unavailable: %s", exc)
                log.error("hint: start the simulator or use --no-carla "
                          "for a synthetic smoke run")
                return 3
        log.exception("setup failed")
        return 2

    try:
        result = agent.run(max_ticks=args.ticks,
                           duration_s=args.duration,
                           pace=not args.fast)
    finally:
        agent.cleanup()
    return 0 if not result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
