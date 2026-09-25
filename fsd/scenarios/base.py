"""Scenario framework — scripted closed-loop tests on the synthetic world.

Terminology
-----------

``Scenario``       A scripted perturbation + pass/fail oracle.
``ScenarioScene``  The mutable-world handle handed to ``Scenario.setup()``.
``ScriptedActor``  A kinematic obstacle injected into the detector output.
``ScenarioEvent``  A timestamped marker the runner records for the report.
``Verdict``        PASS / FAIL / TIMEOUT.

A scenario never touches the agent loop itself. It mutates the world through
:class:`ScenarioScene` (lead-vehicle knobs, scripted actors, light override,
free-space floor) and/or injects faults via ``scene.inject(fault)`` — see
:mod:`fsd.eval.faults`. The runner then calls ``agent.tick()`` once per step,
so the full perception → planning → control → safety stack stays in the loop.

Everything here targets ``AutopilotAgent(smoke=True)`` — the synthetic
kinematic ego + lead-vehicle scene. No CARLA server is required.

Injection model
---------------

``ScenarioScene.install()`` wraps three agent stage methods with thin
instance-attribute shims; the original methods run unchanged inside them:

* ``agent._detect_objects`` — appended with live scripted actors (and, when
  ``hide_lead`` is set, the built-in synthetic lead is filtered out).
* ``agent._light_state`` — replaced by a scenario-controlled light override.
* ``agent._fuse`` — post-processed with a free-space floor so scenarios can
  model virtual constraints (e.g. a stop bar) without spawning fake objects.
"""
from __future__ import annotations

import itertools
import math
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from fsd.core.logger import get
from fsd.core.types import (
    DetectedObject,
    DriveMode,
    LaneInfo,
    LightState,
    Vec3,
    VehicleState,
)

log = get("scenarios.base")

#: obj_id used by _SyntheticScene.lead_object()
LEAD_OBJ_ID = 1
#: ego footprint constants (mirror fsd.safety.rules)
EGO_HALF_LENGTH_M = 2.4
EGO_HALF_WIDTH_M = 1.0

#: default class -> bbox half-extents (x fwd, y right, z up)
DEFAULT_EXTENTS = {
    "vehicle": Vec3(2.2, 0.9, 0.75),
    "pedestrian": Vec3(0.3, 0.3, 0.9),
    "cyclist": Vec3(0.9, 0.35, 0.85),
    "sign": Vec3(0.2, 0.2, 0.6),
    "misc": Vec3(0.5, 0.5, 0.8),
}


class Verdict(Enum):
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()


@dataclass
class ScenarioEvent:
    """A timestamped marker emitted by ``Scenario.tick`` for the run log."""
    t: float
    kind: str        # inject | trigger | collision | clear | note
    message: str

    def __str__(self) -> str:
        return f"[t={self.t:6.2f}] {self.kind}: {self.message}"


# Backwards-friendly alias: the spec signature is ``tick(t) -> Optional[Event]``
Event = ScenarioEvent


@dataclass
class ScriptedActor:
    """A kinematic obstacle injected into the object-detection stage.

    Positions/velocities are world frame. ``step`` does Euler integration with
    optional constant acceleration. ``to_object`` renders the actor as a
    :class:`DetectedObject` for the perception pipeline.
    """
    cls: str = "vehicle"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    extent: Optional[Vec3] = None
    confidence: float = 0.9
    obj_id: int = field(default_factory=lambda: next(ScriptedActor._ids))

    _ids = itertools.count(7000)

    def __post_init__(self) -> None:
        if self.extent is None:
            self.extent = DEFAULT_EXTENTS.get(self.cls, DEFAULT_EXTENTS["misc"])

    def step(self, dt: float) -> None:
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def to_object(self, timestamp: float) -> DetectedObject:
        return DetectedObject(
            obj_id=self.obj_id, cls=self.cls,
            position=Vec3(self.x, self.y, self.z),
            velocity=Vec3(self.vx, self.vy, 0.0),
            bbox_extent=self.extent, confidence=self.confidence,
            timestamp=timestamp)

    def ego_frame(self, ego: VehicleState) -> tuple:
        """(longitudinal, lateral[+left]) offset of this actor from ``ego``."""
        cy, sy = math.cos(ego.yaw), math.sin(ego.yaw)
        dx, dy = self.x - ego.x, self.y - ego.y
        return dx * cy + dy * sy, -dx * sy + dy * cy

    def bumper_gap(self, ego: VehicleState) -> float:
        """Longitudinal bumper clearance along ego's forward axis (m)."""
        lon, _ = self.ego_frame(ego)
        return lon - EGO_HALF_LENGTH_M - abs(self.extent.x)

    def overlaps_ego(self, ego: VehicleState) -> bool:
        """Ground-truth footprint overlap test (collision oracle)."""
        lon, lat = self.ego_frame(ego)
        return (abs(lon) < EGO_HALF_LENGTH_M + abs(self.extent.x)
                and abs(lat) < EGO_HALF_WIDTH_M + abs(self.extent.y))


class ScenarioScene:
    """Mutable-world handle passed to ``Scenario.setup()``.

    Wraps the agent's synthetic world and the three injection points
    described in the module docstring. The runner installs the wrappers once
    per run via :meth:`install`; :meth:`teardown` restores originals and
    clears any injected faults.
    """

    def __init__(self, agent):
        self.agent = agent
        self.dt = float(agent.dt)
        self.actors: Dict[int, ScriptedActor] = {}
        self._faults: List = []
        self._lead_hidden = False
        self._light_override: Optional[LightState] = None
        self._free_floor_fn: Optional[
            Callable[[VehicleState], Optional[float]]] = None
        self._synthetic_lane = False
        self._lane_x0 = 0.0
        self._lane_y0 = 0.0
        self._lane_yaw0 = 0.0
        self._lane_width = 3.5
        self._orig: Dict[str, object] = {}
        self._installed = False

    # ------------------------------------------------------------- world I/O

    @property
    def world(self):
        """The ``_SyntheticScene`` instance (smoke-mode world)."""
        return self.agent._scene

    @property
    def ego(self):
        """The ``_SyntheticEgo`` kinematic vehicle."""
        return self.agent._synthetic_ego

    def ego_state(self) -> VehicleState:
        """Fresh ego VehicleState (reads through the agent's own accessor)."""
        return self.agent._ego_state()

    @property
    def mode(self) -> DriveMode:
        return self.agent.mode

    # ---- lead vehicle knobs -------------------------------------------------

    def set_lead(self, dist: Optional[float] = None,
                 speed: Optional[float] = None) -> None:
        """Move/pace the built-in synthetic lead vehicle."""
        if self.world is not None:
            if dist is not None:
                self.world.lead_dist = float(dist)
            if speed is not None:
                self.world.lead_speed = float(speed)

    def hide_lead(self, hidden: bool = True) -> None:
        """Drop the built-in lead from the detection list entirely."""
        self._lead_hidden = bool(hidden)

    def lead_bumper_gap(self) -> float:
        """Bumper clearance to the built-in lead (m); inf when absent."""
        if self.world is None or self._lead_hidden:
            return math.inf
        # lead_object() emits bbox_extent.x = 2.2 m
        return self.world.lead_dist - EGO_HALF_LENGTH_M - 2.2

    # ---- scripted actors ----------------------------------------------------

    def spawn(self, actor: Optional[ScriptedActor] = None, *,
              cls: str = "vehicle", ahead_m: float = 0.0,
              lateral_m: float = 0.0, speed_mps: float = 0.0,
              heading: object = "forward", extent: Optional[Vec3] = None,
              confidence: float = 0.9,
              accel_mps2: float = 0.0) -> ScriptedActor:
        """Spawn a :class:`ScriptedActor` relative to the current ego pose.

        ``lateral_m`` follows the stack convention: positive = ego's left.
        ``heading`` may be ``"forward"`` (along ego heading), ``"cross_left"``,
        ``"cross_right"``, or a world-frame yaw in radians. ``accel_mps2``
        applies constant acceleration along the chosen heading.
        """
        ego = self.ego_state()
        cy, sy = math.cos(ego.yaw), math.sin(ego.yaw)
        x = ego.x + ahead_m * cy + lateral_m * -sy
        y = ego.y + ahead_m * sy + lateral_m * cy
        if heading == "forward":
            hx, hy = cy, sy
        elif heading == "cross_left":      # move toward +lat (ego's left)
            hx, hy = -sy, cy
        elif heading == "cross_right":     # move toward -lat (ego's right)
            hx, hy = sy, -cy
        else:                               # explicit world yaw
            hx, hy = math.cos(float(heading)), math.sin(float(heading))
        a = actor or ScriptedActor()
        a.cls = cls if actor is None else a.cls
        a.x, a.y = x, y
        a.vx, a.vy = speed_mps * hx, speed_mps * hy
        a.ax, a.ay = accel_mps2 * hx, accel_mps2 * hy
        if extent is not None:
            a.extent = extent
        elif actor is None:
            a.extent = DEFAULT_EXTENTS.get(cls, DEFAULT_EXTENTS["misc"])
        a.confidence = confidence if actor is None else a.confidence
        self.actors[a.obj_id] = a
        return a

    def despawn(self, obj_id: int) -> None:
        self.actors.pop(obj_id, None)

    def step_actors(self, dt: Optional[float] = None) -> None:
        """Integrate every scripted actor by ``dt`` (default: scene dt)."""
        for a in self.actors.values():
            a.step(self.dt if dt is None else dt)

    def first_collision(self) -> Optional[ScriptedActor]:
        """Return the first scripted actor overlapping the ego footprint."""
        ego = self.ego_state()
        for a in self.actors.values():
            if a.overlaps_ego(ego):
                return a
        return None

    # ---- pipeline overrides -------------------------------------------------

    def set_light(self, state: Optional[LightState]) -> None:
        """Override the reported traffic-light state (None = passthrough)."""
        self._light_override = state

    def set_free_space_floor(
            self, fn: Optional[Callable[[VehicleState], Optional[float]]]
    ) -> None:
        """Cap ``perception.free_space_ahead`` with ``fn(ego)`` each tick.

        ``fn`` returns the distance to the nearest virtual constraint (or
        None/inf for none). Used to model stop lines / closed gates without
        injecting a fake physical object.
        """
        self._free_floor_fn = fn

    def synthetic_lane(self, enabled: bool = True,
                       lane_width: float = 3.5) -> None:
        """Feed a geometrically-consistent lane when ``_detect_lane`` runs.

        The smoke cameras produce black frames, so the vision lane detector
        reports ``detected=False`` and the stack rides at DEGRADED with
        lane-loss warnings. Enabling this reports a real LaneInfo tracking the
        ego's offset from its initial straight path — honest geometry, and it
        makes lane-center RMSE measurable in eval.
        """
        self._synthetic_lane = bool(enabled)
        ego = self.ego_state()
        self._lane_x0, self._lane_y0 = ego.x, ego.y
        self._lane_yaw0 = ego.yaw
        self._lane_width = float(lane_width)

    # ---- fault injection ----------------------------------------------------

    def inject(self, fault):
        """Apply a fault (see fsd.eval.faults) and register it for teardown."""
        fault.apply(self)
        self._faults.append(fault)
        return fault

    # ---- install / teardown --------------------------------------------------

    def install(self) -> "ScenarioScene":
        """Wrap the agent's injection-point stage methods. Idempotent."""
        if self._installed:
            return self
        agent = self.agent

        orig_objects = agent._detect_objects
        orig_light = agent._light_state
        orig_fuse = agent._fuse
        orig_lane = agent._detect_lane
        self._orig = {
            "_detect_objects": orig_objects,
            "_light_state": orig_light,
            "_fuse": orig_fuse,
            "_detect_lane": orig_lane,
        }

        def objects_hook(ego, sensors):
            objs = list(orig_objects(ego, sensors) or [])
            if self._lead_hidden:
                objs = [o for o in objs if o.obj_id != LEAD_OBJ_ID]
            objs.extend(a.to_object(ego.timestamp)
                        for a in self.actors.values())
            return objs

        def light_hook(ego, sensors):
            if self._light_override is not None:
                return self._light_override
            return orig_light(ego, sensors)

        def fuse_hook(ego, sensors, lane, objects, light):
            p = orig_fuse(ego, sensors, lane, objects, light)
            if p is not None and self._free_floor_fn is not None:
                try:
                    d = self._free_floor_fn(ego)
                except Exception:
                    d = None
                if d is not None and math.isfinite(d):
                    p.free_space_ahead = min(p.free_space_ahead,
                                             max(0.0, float(d)))
            return p

        def lane_hook(ego, sensors):
            lane = orig_lane(ego, sensors)
            if not self._synthetic_lane:
                return lane
            # Ego's lateral displacement from its initial straight path,
            # mapped into lane-center offset (+offset = displaced right).
            lat = (-(ego.x - self._lane_x0) * math.sin(self._lane_yaw0)
                   + (ego.y - self._lane_y0) * math.cos(self._lane_yaw0))
            heading_err = ego.yaw - self._lane_yaw0
            return LaneInfo(
                left_offset=self._lane_width / 2.0 + lat,
                right_offset=self._lane_width / 2.0 - lat,
                center_offset=-lat,
                heading_error=heading_err,
                curvature=0.0,
                lane_width=self._lane_width,
                detected=True)

        agent._detect_objects = objects_hook
        agent._light_state = light_hook
        agent._fuse = fuse_hook
        agent._detect_lane = lane_hook
        self._installed = True
        return self

    def teardown(self) -> None:
        """Restore original stage methods and clear injected faults."""
        for f in reversed(self._faults):
            try:
                f.clear()
            except Exception:
                log.debug("fault %r clear failed", f, exc_info=True)
        self._faults.clear()
        for attr, orig in self._orig.items():
            try:
                setattr(self.agent, attr, orig)
            except Exception:
                pass
        self._orig.clear()
        self.actors.clear()
        self._installed = False


class Scenario(ABC):
    """Base class for a scripted closed-loop test.

    Lifecycle::

        scene = ScenarioScene(agent).install()
        scenario.setup(scene)
        while t < scenario.timeout_s and not scenario.done(t):
            ev = scenario.tick(t)          # perturb world, maybe emit Event
            res = agent.tick()             # full stack runs
        verdict = scenario.evaluate(metrics)  # True PASS / False FAIL / None
    """

    name: str = "scenario"
    description: str = ""
    timeout_s: float = 25.0

    def __init__(self) -> None:
        self.scene: Optional[ScenarioScene] = None

    # ------------------------------------------------------------- interface

    def setup(self, scene: ScenarioScene) -> None:
        """Bind the scene handle and configure initial world state."""
        self.scene = scene

    def tick(self, t: float) -> Optional[ScenarioEvent]:
        """Advance the script to sim time ``t``; optionally emit an Event."""
        return None

    def done(self, t: float) -> bool:
        """Early-exit oracle: the outcome is decided, stop the run."""
        return False

    def evaluate(self, metrics) -> Optional[bool]:
        """Pass/fail oracle. True=PASS, False=FAIL, None=undecided (TIMEOUT)."""
        return None

    # ------------------------------------------------------------- helpers

    def _ego(self) -> VehicleState:
        assert self.scene is not None
        return self.scene.ego_state()

    def _mode(self) -> DriveMode:
        assert self.scene is not None
        return self.scene.mode

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"
