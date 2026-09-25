"""Fault-injection hooks for scenario testing.

Each fault is a small object with ``apply(scene)`` / ``clear()``. Faults patch
methods on the *agent instance* (or its module objects) and restore them on
``clear()`` — nothing here mutates the stack's source or leaks state between
runs. Use via ``scene.inject(fault)`` so the runner cleans up automatically.

Fault catalogue
---------------

``DropSensor``       a pipeline stage raises — the agent degrades onto
                     last-good perception, then staleness rules fire.
``DelayStage``       a stage's outputs are replayed from ``lag`` calls ago —
                     the loop stays alive but sees stale data.
``SpoofObject``      appends a phantom ``DetectedObject`` to detections —
                     false-positive / adversarial-object testing.
``FreezeHeartbeat``  stops the heartbeat: either the safety monitor's explicit
                     ``heartbeat()`` channel, or a named pipeline stage (its
                     ``_last_ok`` freshness dies, so the watchdog trips).

Stage-name map (``DropSensor``/``DelayStage``/``FreezeHeartbeat``) — the keys
of ``_STAGE_METHODS``: ``sensors``, ``objects``, ``lane``, ``traffic_light``,
``fusion``, ``route``, ``behavior``, ``trajectory``, ``control``.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from fsd.core.logger import get
from fsd.core.types import DetectedObject, Vec3

log = get("eval.faults")


class SensorDropoutError(RuntimeError):
    """Raised inside a wrapped stage while a DropSensor fault is active."""


class HeartbeatLostError(RuntimeError):
    """Raised inside a wrapped stage while a FreezeHeartbeat fault is active."""


#: public stage name -> AutopilotAgent method implementing that stage
_STAGE_METHODS: Dict[str, str] = {
    "sensors": "_sensor_snapshot",
    "objects": "_detect_objects",
    "lane": "_detect_lane",
    "traffic_light": "_light_state",
    "fusion": "_fuse",
    "route": "_route",
    "behavior": "_behavior",
    "trajectory": "_trajectory",
    "control": "_control",
}

#: planning-family stages grouped under the "planning" alias
_EXTRA_ALIASES: Dict[str, Tuple[str, ...]] = {
    "perception": ("objects", "lane", "traffic_light"),
    "planning": ("trajectory",),
}


def _stages(spec) -> List[str]:
    """Normalize a stage spec (str or sequence) into agent method names."""
    if isinstance(spec, str):
        spec = [spec]
    out: List[str] = []
    for s in spec:
        key = str(s).lower()
        if key in _EXTRA_ALIASES:
            out.extend(_EXTRA_ALIASES[key])
        elif key in _STAGE_METHODS:
            out.append(key)
        else:
            raise ValueError(
                f"unknown stage {s!r}; known: {sorted(_STAGE_METHODS)} "
                f"+ aliases {sorted(_EXTRA_ALIASES)}")
    return [_STAGE_METHODS[k] for k in dict.fromkeys(out)]


class Fault:
    """Base class: records every patch so ``clear()`` restores exactly."""

    name = "fault"

    def __init__(self) -> None:
        self._patched: List[Tuple[object, str, object]] = []
        self.active = False
        self.scene = None

    # ------------------------------------------------------------------ api
    def apply(self, scene) -> None:
        self.scene = scene
        self.active = True

    def clear(self) -> None:
        for obj, attr, orig in reversed(self._patched):
            try:
                setattr(obj, attr, orig)
            except Exception:
                log.debug("%s: restore %s failed", self.name, attr)
        self._patched.clear()
        self.active = False

    # ------------------------------------------------------------------ util
    def _patch(self, obj, attr: str, replacement) -> None:
        orig = getattr(obj, attr)
        self._patched.append((obj, attr, orig))
        setattr(obj, attr, replacement)
        log.info("fault %s applied: %s.%s", self.name,
                 type(obj).__name__, attr)

    def _patch_agent_stage(self, method: str, wrapper_fn) -> None:
        """Wrap an AutopilotAgent stage method with ``wrapper_fn(orig)``."""
        agent = self.scene.agent
        orig = getattr(agent, method)
        self._patch(agent, method, wrapper_fn(orig))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} active={self.active}>"


class DropSensor(Fault):
    """Make pipeline stages raise ``SensorDropoutError`` while active.

    In smoke mode the detector stage *is* the sensor front-end, so dropping
    e.g. ``objects`` is exactly a sensor-feed loss: the agent degrades onto
    last-good perception, the watchdog flags staleness, and the monitor
    arbitrates SAFE_STOP.

    ``stage`` may be a name or a list — see ``_STAGE_METHODS`` plus the
    aliases ``perception`` and ``planning``.
    """

    name = "drop_sensor"

    def __init__(self, stage="objects") -> None:
        super().__init__()
        self.methods = _stages(stage)

    def apply(self, scene) -> None:
        super().apply(scene)
        for method in self.methods:
            def make_wrapper(m):
                def wrap(orig):
                    def dropped(*a, **kw):
                        raise SensorDropoutError(
                            f"{m} feed lost (injected DropSensor)")
                    return dropped
                return wrap
            self._patch_agent_stage(method, make_wrapper(method))


class DelayStage(Fault):
    """Replay a stage's outputs from ``lag_ticks`` calls ago.

    A real wall-clock sleep would only prove the watchdog notices a slow
    loop; replaying honest-but-old outputs simulates *pipeline latency* — the
    consumer sees data ``lag_ticks * dt`` seconds stale while the loop itself
    stays on schedule. For perception stages the stale ``timestamp`` is
    caught by ``WatchdogRule`` once it exceeds ``watchdog_timeout_s``.
    """

    name = "delay_stage"

    def __init__(self, stage="fusion", lag_ticks: int = 12) -> None:
        super().__init__()
        if lag_ticks < 1:
            raise ValueError("lag_ticks must be >= 1")
        self.methods = _stages(stage)
        self.lag_ticks = int(lag_ticks)

    def apply(self, scene) -> None:
        super().apply(scene)
        lag = self.lag_ticks
        for method in self.methods:
            def make_wrapper(lag=lag):
                def wrap(orig):
                    hist: List = []

                    def lagged(*a, **kw):
                        out = orig(*a, **kw)
                        hist.append(out)
                        if len(hist) > lag:
                            hist.pop(0)
                        # <= lag entries: serve the oldest frame we have —
                        # a stage warming up replays its first output.
                        return hist[0]

                    return lagged
                return wrap
            self._patch_agent_stage(method, make_wrapper())


class SpoofObject(Fault):
    """Inject a phantom :class:`DetectedObject` into the detection stage.

    Two placement modes:

    * ``follow_ego=True`` (default): the ghost is re-projected at
      ``(ahead_m, lateral_m)`` in the ego frame every call — an object that
      keeps a fixed offset no matter how the car moves.
    * ``follow_ego=False``: the ghost's world pose is captured at ``apply``
      time; it stays put (or integrates ``velocity``) while ego drives on.

    Optionally pass ``track``: ``track(ego, t_wall) -> (x, y, vx, vy)`` for
    fully scripted ghosts.
    """

    name = "spoof_object"
    _spoof_ids = [8999]

    def __init__(self, cls: str = "vehicle", ahead_m: float = 20.0,
                 lateral_m: float = 0.0, speed_mps: float = 0.0,
                 obj_id: Optional[int] = None, confidence: float = 0.92,
                 extent: Optional[Vec3] = None,
                 follow_ego: bool = True,
                 track: Optional[Callable] = None) -> None:
        super().__init__()
        self.cls = cls
        self.ahead_m = float(ahead_m)
        self.lateral_m = float(lateral_m)
        self.speed_mps = float(speed_mps)
        self.confidence = float(confidence)
        self.follow_ego = bool(follow_ego)
        self.track = track
        self.obj_id = (obj_id if obj_id is not None
                       else self._next_id())
        if extent is not None:
            self.extent = extent
        else:
            self.extent = {
                "vehicle": Vec3(2.2, 0.9, 0.75),
                "pedestrian": Vec3(0.3, 0.3, 0.9),
                "cyclist": Vec3(0.9, 0.35, 0.85),
                "sign": Vec3(0.2, 0.2, 0.6),
            }.get(cls, Vec3(0.5, 0.5, 0.8))
        # fixed-pose state (follow_ego=False)
        self._fx = self._fy = self._fvx = self._fvy = None
        self._t0 = None

    @classmethod
    def _next_id(cls) -> int:
        cls._spoof_ids[0] += 1
        return cls._spoof_ids[0]

    # ------------------------------------------------------------------ api
    def apply(self, scene) -> None:
        super().apply(scene)
        agent = scene.agent
        if not self.follow_ego and self.track is None:
            ego = scene.ego_state()
            cy, sy = math.cos(ego.yaw), math.sin(ego.yaw)
            self._fx = ego.x + self.ahead_m * cy - self.lateral_m * sy
            self._fy = ego.y + self.ahead_m * sy + self.lateral_m * cy
            self._fvx = self.speed_mps * cy
            self._fvy = self.speed_mps * sy
            self._t0 = time.time()

        orig = agent._detect_objects

        def hooked(ego, sensors, _o=orig):
            objs = list(_o(ego, sensors) or [])
            objs.append(self._make(ego))
            return objs

        self._patch(agent, "_detect_objects", hooked)

    def _make(self, ego) -> DetectedObject:
        if self.track is not None:
            x, y, vx, vy = self.track(ego, time.time())
        elif self.follow_ego:
            cy, sy = math.cos(ego.yaw), math.sin(ego.yaw)
            x = ego.x + self.ahead_m * cy - self.lateral_m * sy
            y = ego.y + self.ahead_m * sy + self.lateral_m * cy
            vx, vy = self.speed_mps * cy, self.speed_mps * sy
        else:
            dt = time.time() - (self._t0 or time.time())
            x = self._fx + self._fvx * dt
            y = self._fy + self._fvy * dt
            vx, vy = self._fvx, self._fvy
        return DetectedObject(
            obj_id=self.obj_id, cls=self.cls,
            position=Vec3(x, y, 0.0), velocity=Vec3(vx, vy, 0.0),
            bbox_extent=self.extent, confidence=self.confidence,
            timestamp=ego.timestamp)


class FreezeHeartbeat(Fault):
    """Stop a liveness heartbeat.

    ``stage="monitor"`` patches ``agent.safety.heartbeat`` into a no-op —
    ``_last_heartbeat`` ages past ``watchdog_timeout_s`` and WatchdogRule
    fires a critical ("pipeline heartbeat lost"), while the planner keeps
    producing otherwise-valid commands.

    Any other stage name (``"planning"``, ``"objects"``, ...) patches that
    agent stage to raise ``HeartbeatLostError`` — the agent marks the stage
    dead for the tick, its ``_last_ok`` freshness expires, and the watchdog
    reports the pipeline not alive.
    """

    name = "freeze_heartbeat"

    def __init__(self, stage: str = "monitor") -> None:
        super().__init__()
        self.stage = stage.lower()
        if self.stage == "monitor":
            self.methods: List[str] = []
        else:
            self.methods = _stages(self.stage)

    def apply(self, scene) -> None:
        super().apply(scene)
        agent = scene.agent
        if self.stage == "monitor":
            safety = getattr(agent, "safety", None)
            hb = getattr(safety, "heartbeat", None)
            if callable(hb):
                def dead_hb():
                    return None  # heartbeat silently dropped
                self._patch(safety, "heartbeat", dead_hb)
            else:
                # fallback safety has no heartbeat channel — stall the
                # trajectory stage so pipeline_alive still degrades.
                log.info("%s: no heartbeat channel — stalling _trajectory",
                         self.name)
                self._patch_stage_raise("_trajectory")
            return
        for method in self.methods:
            self._patch_stage_raise(method)

    def _patch_stage_raise(self, method: str) -> None:
        def wrap(orig, _m=method):
            def stalled(*a, **kw):
                raise HeartbeatLostError(
                    f"{_m} stalled (injected FreezeHeartbeat)")
            return stalled
        self._patch_agent_stage(method, wrap)


__all__ = [
    "Fault",
    "DropSensor",
    "DelayStage",
    "SpoofObject",
    "FreezeHeartbeat",
    "SensorDropoutError",
    "HeartbeatLostError",
]
