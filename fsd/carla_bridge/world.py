"""CARLA world wrapper — connection, map, synchronous ticking, cleanup.

The bridge owns a ``carla.Client`` and the ``carla.World`` it serves. Typical
lifecycle inside :class:`fsd.agents.autopilot.AutopilotAgent`::

    world = CarlaWorld(cfg)
    world.connect()                  # client + town load (async mode)
    ego.spawn()                      # spawn actors while async (no deadlock)
    traffic.spawn_all(avoid=ego.transform)
    world.set_synchronous(True)      # THEN flip server into fixed-step mode
    ...
    world.cleanup()                  # restore async settings

Spawning actors *before* enabling synchronous mode is deliberate: in sync
mode the server only drains its command queue during a frame, so blocking
spawn calls can deadlock unless another thread ticks.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

from fsd.core.config import Config
from fsd.core.logger import get

try:
    import carla
except Exception:  # pragma: no cover - carla egg/wheel absent
    carla = None  # type: ignore[assignment]

log = get("carla_bridge.world")


class CarlaUnavailableError(RuntimeError):
    """Raised when the CARLA python API cannot be imported or reached."""


def require_carla():
    """Return the ``carla`` module or raise a friendly error."""
    if carla is None:
        raise CarlaUnavailableError(
            "the 'carla' package is not importable. Install the CARLA wheel "
            "(pip install carla) or put the simulator's .egg on PYTHONPATH.")
    return carla


class CarlaWorld:
    """Thin owner around ``carla.Client``/``carla.World``."""

    #: Loading a big map (e.g. Town10HD) routinely exceeds the RPC timeout.
    _LOAD_TIMEOUT_S = 180.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client: Any = None
        self.world: Any = None
        self.synchronous = False
        self._prev_settings: Any = None
        self._connected = False

    # ------------------------------------------------------------ connection

    def connect(self) -> "CarlaWorld":
        """Connect to the server and load the configured town (async mode)."""
        c = require_carla()
        sim = self.cfg.sim
        log.info("connecting to CARLA at %s:%s (timeout %.0fs)",
                 sim.host, sim.port, sim.timeout_s)
        try:
            self.client = c.Client(sim.host, sim.port, worker_threads=2)
        except TypeError:  # older CARLA without worker_threads kwarg
            self.client = c.Client(sim.host, sim.port)
        self.client.set_timeout(sim.timeout_s)

        try:
            cv, sv = self.client.get_client_version(), self.client.get_server_version()
            log.info("carla client %s / server %s", cv, sv)
        except Exception as exc:
            raise CarlaUnavailableError(
                f"no CARLA server responding at {sim.host}:{sim.port} — {exc}")

        self._load_town(sim.town)
        self.world = self.client.get_world()
        self._connected = True
        log.info("world ready: map=%s", self.map_name)
        return self

    def _load_town(self, town: str) -> None:
        current = ""
        try:
            current = self.client.get_world().get_map().name
        except Exception:
            pass
        if town and town.split("/")[-1] not in current.split("/")[-1]:
            log.info("loading town %s (current %s) — can take a minute",
                     town, current or "?")
            try:
                self.client.set_timeout(self._LOAD_TIMEOUT_S)
                self.client.load_world(town)
            finally:
                self.client.set_timeout(self.cfg.sim.timeout_s)
            # Server is briefly unresponsive right after a map swap.
            deadline = time.time() + 30.0
            while time.time() < deadline:
                try:
                    if self.client.get_world().get_map().name.endswith(town):
                        break
                except Exception:
                    time.sleep(0.5)
            else:
                log.warning("map did not report %s after load — continuing", town)

    # -------------------------------------------------------------- settings

    def set_synchronous(self, enabled: bool = True,
                        fixed_delta_s: Optional[float] = None) -> None:
        """Toggle synchronous fixed-step mode (20 Hz by default)."""
        c = require_carla()
        if self.world is None:
            raise RuntimeError("connect() first")
        dt = self.cfg.sim.fixed_delta_s if fixed_delta_s is None else fixed_delta_s
        settings = self.world.get_settings()
        if enabled:
            self._prev_settings = settings
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = dt
            self.world.apply_settings(settings)
            self.synchronous = True
            log.info("synchronous mode ON (fixed_delta=%.3fs)", dt)
        else:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            self.world.apply_settings(settings)
            self.synchronous = False
            log.info("synchronous mode OFF")

    # ----------------------------------------------------------------- world

    def tick(self) -> int:
        """Advance one frame in sync mode; wait for next snapshot in async."""
        if self.world is None:
            raise RuntimeError("connect() first")
        if self.synchronous:
            return int(self.world.tick())
        snap = self.world.wait_for_tick(self.cfg.sim.timeout_s)
        return int(snap.frame)

    @property
    def snapshot(self):
        return self.world.get_snapshot() if self.world is not None else None

    @property
    def map_name(self) -> str:
        try:
            return self.world.get_map().name
        except Exception:
            return "?"

    @property
    def map(self):
        return self.world.get_map()

    @property
    def blueprint_library(self):
        return self.world.get_blueprint_library()

    @property
    def spawn_points(self) -> List[Any]:
        return self.map.get_spawn_points()

    def random_location(self):
        """Random drivable/nav-mesh location, or None."""
        try:
            return self.world.get_random_location_from_navigation()
        except Exception:
            return None

    def get_actors(self, type_filter: Optional[str] = None):
        actors = self.world.get_actors()
        return actors.filter(type_filter) if type_filter else actors

    def spawn_actor(self, blueprint, transform, attach_to: Any = None):
        """Spawn wrapper — blocking; call while async or pre-tick."""
        try:
            return self.world.spawn_actor(blueprint, transform,
                                          attach_to=attach_to)
        except RuntimeError as exc:
            log.debug("spawn_actor failed (%s): %s",
                      getattr(blueprint, "id", blueprint), exc)
            return None

    def try_spawn_actor(self, blueprint, transform):
        try:
            return self.world.try_spawn_actor(blueprint, transform)
        except RuntimeError:
            return None

    # --------------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Restore async mode + original settings. Actors are owned elsewhere."""
        if self.world is None:
            return
        try:
            if self.synchronous:
                self.set_synchronous(False)
            elif self._prev_settings is not None:
                self.world.apply_settings(self._prev_settings)
        except Exception as exc:
            log.warning("failed to restore world settings: %s", exc)
        self._connected = False
        log.info("world cleaned up")

    # Context manager -------------------------------------------------------

    def __enter__(self) -> "CarlaWorld":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.cleanup()
