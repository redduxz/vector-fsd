"""Ambient traffic — autopilot vehicles + AI pedestrians via the CARLA
TrafficManager.

Call :meth:`spawn_all` while the world is still asynchronous, then
:meth:`set_synchronous` together with ``CarlaWorld.set_synchronous`` — the
TrafficManager must be told separately that the world is synced.
"""
from __future__ import annotations

import math
import random
from typing import Any, List

from fsd.core.config import Config
from fsd.core.logger import get
from fsd.carla_bridge.world import CarlaWorld, require_carla

log = get("carla_bridge.traffic")

#: Keep this much clearance around the ego spawn when seeding traffic.
_MIN_EGO_CLEARANCE_M = 12.0


class TrafficManager:
    """Owns every non-ego actor: autopilot vehicles and wandering walkers."""

    def __init__(self, world: CarlaWorld, cfg: Config):
        self.world = world
        self.cfg = cfg
        self._rng = random.Random(cfg.sim.seed + 1)
        self.vehicles: List[Any] = []
        self.walkers: List[Any] = []
        self.controllers: List[Any] = []
        self._tm: Any = None
        self.tm_port = int(
            cfg.raw.get("sim", {}).get("traffic_manager_port", 8000))

    # ------------------------------------------------------------------ tm

    @property
    def tm(self):
        if self._tm is None:
            self._tm = self.world.client.get_trafficmanager(self.tm_port)
        return self._tm

    def set_synchronous(self, enabled: bool = True) -> None:
        """Put the TrafficManager into/out-of sync stepping."""
        try:
            self.tm.set_synchronous_mode(enabled)
            self.tm.set_random_device_seed(self.cfg.sim.seed)
        except Exception as exc:
            log.warning("traffic manager sync-mode change failed: %s", exc)

    # ------------------------------------------------------------------ spawn

    def _vehicle_blueprints(self) -> List[Any]:
        lib = self.world.blueprint_library
        bps = list(lib.filter("vehicle.*"))
        # Skip 2-wheelers and the hero bp — ambient fleet stays conventional.
        def _ok(bp):
            try:
                wheels = int(bp.get_attribute("number_of_wheels").as_int())
            except Exception:
                wheels = 4
            return wheels >= 4
        bps = [b for b in bps if _ok(b)] or list(lib.filter("vehicle.*"))
        return bps

    def _spawn_vehicles(self, count: int, avoid_loc) -> int:
        points = list(self.world.spawn_points)
        self._rng.shuffle(points)
        bps = self._vehicle_blueprints()
        spawned = 0
        for tf in points:
            if spawned >= count:
                break
            if avoid_loc is not None:
                d = math.hypot(tf.location.x - avoid_loc.x,
                               tf.location.y - avoid_loc.y)
                if d < _MIN_EGO_CLEARANCE_M:
                    continue
            bp = self._rng.choice(bps)
            actor = self.world.spawn_actor(bp, tf)
            if actor is None:
                continue
            try:
                actor.set_autopilot(True, self.tm_port)
                # A little heterogeneity in following behaviour.
                self.tm.distance_to_leading_vehicle(
                    actor, self._rng.uniform(1.5, 3.5))
                self.tm.vehicle_percentage_speed_difference(
                    actor, self._rng.uniform(0.0, 25.0))
            except Exception as exc:
                log.debug("autopilot setup failed for %d: %s",
                          actor.id, exc)
            self.vehicles.append(actor)
            spawned += 1
        return spawned

    def _spawn_pedestrians(self, count: int) -> int:
        lib = self.world.blueprint_library
        try:
            walker_bps = list(lib.filter("walker.pedestrian.*"))
        except Exception:
            walker_bps = []
        try:
            ctrl_bp = lib.find("controller.ai.walker")
        except Exception:
            ctrl_bp = None
        if not walker_bps or ctrl_bp is None:
            log.warning("walker blueprints unavailable — no pedestrians")
            return 0

        c = require_carla()
        spawned = 0
        for _ in range(count * 2):            # allow some failed attempts
            if spawned >= count:
                break
            loc = self.world.random_location()
            if loc is None:
                continue
            tf = c.Transform(c.Location(loc.x, loc.y, loc.z + 0.5))
            walker = self.world.spawn_actor(self._rng.choice(walker_bps), tf)
            if walker is None:
                continue
            ctrl = self.world.spawn_actor(ctrl_bp, c.Transform(),
                                          attach_to=walker)
            if ctrl is None:
                try:
                    walker.destroy()
                except Exception:
                    pass
                continue
            try:
                ctrl.start()
                dest = self.world.random_location()
                if dest is not None:
                    ctrl.go_to_location(dest)
                ctrl.set_max_speed(1.0 + self._rng.uniform(0.0, 1.6))
            except Exception as exc:
                log.debug("walker controller init failed: %s", exc)
            self.walkers.append(walker)
            self.controllers.append(ctrl)
            spawned += 1
        return spawned

    def spawn_all(self, avoid_transform=None) -> dict:
        """Spawn the configured ambient fleet. Returns per-class counts."""
        sim = self.cfg.sim
        avoid_loc = avoid_transform.location if avoid_transform is not None \
            else None
        nv = self._spawn_vehicles(sim.traffic_count, avoid_loc)
        np_ = self._spawn_pedestrians(sim.pedestrian_count)
        log.info("traffic spawned: %d vehicles, %d pedestrians "
                 "(wanted %d/%d)", nv, np_,
                 sim.traffic_count, sim.pedestrian_count)
        return {"vehicles": nv, "pedestrians": np_}

    # ---------------------------------------------------------------- teardown

    @property
    def actor_count(self) -> int:
        return len(self.vehicles) + len(self.walkers) + len(self.controllers)

    def destroy_all(self) -> None:
        """Stop AI controllers then batch-destroy everything we spawned."""
        for ctrl in self.controllers:
            try:
                ctrl.stop()
            except Exception:
                pass
        c = None
        try:
            import carla as _c
            c = _c
        except Exception:
            pass
        actors = self.controllers + self.walkers + self.vehicles
        if c is not None and self.world.client is not None and actors:
            try:
                batch = [c.command.DestroyActor(a) for a in actors]
                for resp in self.world.client.apply_batch_sync(batch, True):
                    if resp.error:
                        log.debug("destroy actor error: %s", resp.error)
            except Exception as exc:  # async world should process instantly
                log.warning("batch destroy failed, trying per-actor: %s", exc)
                for a in actors:
                    try:
                        a.destroy()
                    except Exception:
                        pass
        else:
            for a in actors:
                try:
                    a.destroy()
                except Exception:
                    pass
        log.info("traffic destroyed: %d actors", self.actor_count)
        self.controllers.clear()
        self.walkers.clear()
        self.vehicles.clear()
