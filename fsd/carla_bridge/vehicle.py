"""Ego vehicle — blueprint spawn, sensor rig, state extraction, actuation."""
from __future__ import annotations

import math
import random
import time
from typing import Any, Optional

from fsd.core.config import Config
from fsd.core.logger import get
from fsd.core.types import ControlCommand, VehicleState

from fsd.carla_bridge.sensors import SensorSuite
from fsd.carla_bridge.world import CarlaWorld, require_carla

log = get("carla_bridge.vehicle")


class EgoVehicle:
    """The autonomous hero vehicle and everything bolted onto it."""

    def __init__(self, world: CarlaWorld, cfg: Config):
        self.world = world
        self.cfg = cfg
        self.actor: Any = None
        self.sensors = SensorSuite(cfg)
        self.spawn_transform: Any = None
        self._rng = random.Random(cfg.sim.seed)

    # ------------------------------------------------------------------ spawn

    def _blueprint(self):
        lib = self.world.blueprint_library
        bp = None
        try:
            bp = lib.find(self.cfg.vehicle.blueprint)
        except (RuntimeError, IndexError):
            log.warning("blueprint %s missing — falling back to any vehicle",
                        self.cfg.vehicle.blueprint)
            try:
                bp = next(iter(lib.filter("vehicle.*")))
            except StopIteration:
                raise RuntimeError("no vehicle blueprints in library")
        try:
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero")
        except Exception:
            pass
        return bp

    def spawn(self, spawn_index: Optional[int] = None,
              transform: Any = None) -> "EgoVehicle":
        """Spawn the ego actor (call while the world is async) + sensors."""
        require_carla()
        bp = self._blueprint()
        points = self.world.spawn_points
        if not points:
            raise RuntimeError("map exposes no spawn points")

        candidates = [transform] if transform is not None else []
        if not candidates:
            start = spawn_index if spawn_index is not None else \
                self._rng.randrange(len(points))
            order = list(range(len(points)))
            # Deterministic-but-jittered walk so retries land on real points.
            self._rng.shuffle(order)
            order.remove(start) if start in order else None
            candidates = [points[start]] + [points[i] for i in order[:8]]

        self.actor = None
        for tf in candidates:
            self.actor = self.world.spawn_actor(bp, tf)
            if self.actor is not None:
                self.spawn_transform = tf
                break
        if self.actor is None:
            raise RuntimeError("could not spawn ego vehicle at any spawn point")

        self.sensors.attach(self.world, self.actor)
        loc = self.spawn_transform.location
        log.info("ego spawned: id=%d bp=%s at (%.1f, %.1f, %.1f)",
                 self.actor.id, self.cfg.vehicle.blueprint, loc.x, loc.y, loc.z)
        return self

    # ------------------------------------------------------------------ state

    @property
    def id(self) -> int:
        return self.actor.id if self.actor is not None else -1

    @property
    def transform(self):
        return self.actor.get_transform() if self.actor is not None else None

    @property
    def is_alive(self) -> bool:
        try:
            return bool(self.actor is not None and self.actor.is_alive)
        except Exception:
            return False

    def state(self) -> VehicleState:
        """Kinematic snapshot in the shared VehicleState contract."""
        tf = self.actor.get_transform()
        loc, rot = tf.location, tf.rotation
        vel = self.actor.get_velocity()
        acc = self.actor.get_acceleration()
        fwd = tf.get_forward_vector()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        accel = acc.x * fwd.x + acc.y * fwd.y + acc.z * fwd.z  # longitudinal
        try:
            steer = float(self.actor.get_control().steer)
        except Exception:
            steer = 0.0
        try:
            ts = self.world.snapshot.timestamp.elapsed_seconds
        except Exception:
            ts = time.time()
        return VehicleState(x=loc.x, y=loc.y, z=loc.z,
                            yaw=math.radians(rot.yaw), speed=speed,
                            accel=accel, steer=steer, timestamp=ts)

    # ----------------------------------------------------------------- control

    def apply(self, cmd: ControlCommand) -> None:
        """Translate the shared ControlCommand into a carla.VehicleControl."""
        if self.actor is None:
            return
        c = require_carla()
        cmd.clamp()
        self.actor.apply_control(c.VehicleControl(
            throttle=float(cmd.throttle),
            brake=float(cmd.brake),
            steer=float(cmd.steer),
            hand_brake=bool(cmd.hand_brake),
            reverse=bool(cmd.reverse),
        ))

    # ----------------------------------------------------------------- events

    def collision_count(self) -> int:
        return self.sensors.count("collision")

    def poll_events(self):
        """New collision/lane-invasion events since last call."""
        return self.sensors.poll_events()

    # ---------------------------------------------------------------- teardown

    def destroy(self) -> None:
        self.sensors.destroy()
        if self.actor is not None:
            try:
                self.actor.destroy()
            except Exception as exc:
                log.debug("ego destroy failed: %s", exc)
            self.actor = None
            log.info("ego destroyed")
