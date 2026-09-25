"""Traffic light monitoring against CARLA actors.

Publishes the state of the most relevant traffic light ahead of the ego
vehicle plus the distance to its stop line. Works on ``carla.TrafficLight``
actors, plain dicts, or any object exposing ``state`` + a location — so it
stays testable without a simulator.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from fsd.core.logger import get
from fsd.core.types import LightState

try:
    import carla
except ImportError:
    carla = None

log = get("perception.traffic_light")

_STATE_MAP = {"red": LightState.RED, "yellow": LightState.YELLOW,
              "green": LightState.GREEN}


class TrafficLightMonitor:
    """Tracks the traffic light constraining the ego's forward path."""

    def __init__(self, search_radius: float = 60.0, ahead_dot: float = -0.2):
        self.search_radius = search_radius
        self.ahead_dot = ahead_dot          # dot(ego_fwd, dir_to_light) floor
        self.state = LightState.UNKNOWN     # last reported state
        self.stop_distance_m = math.inf     # distance to active stop line
        self.active_light_id: Optional[int] = None

    # ------------------------------------------------------------------ API
    def update(self, ego, world=None, traffic_lights=None) -> LightState:
        """Refresh and return the constraining light state.

        UNKNOWN  -> no light information was provided at all.
        GREEN    -> lights were queried but none constrains the ego (or green).
        """
        ex, ey, eyaw = self._ego_xy_yaw(ego)
        fwd = (math.cos(eyaw), math.sin(eyaw))

        # authoritative path: CARLA Vehicle knows the light it is under
        st = self._vehicle_api(ego, ex, ey)
        if st is not None:
            self.state = st
            return st

        lights = traffic_lights
        if lights is None and world is not None:
            try:
                lights = world.get_actors().filter("traffic.traffic_light")
            except Exception:
                lights = None
        if lights is None:
            self.state, self.stop_distance_m = LightState.UNKNOWN, math.inf
            self.active_light_id = None
            return self.state

        best_d, best_tl = math.inf, None
        for tl in lights:
            p = self._stop_point(tl) or self._location(tl)
            if p is None:
                continue
            dx, dy = p[0] - ex, p[1] - ey
            d = math.hypot(dx, dy)
            if d > self.search_radius:
                continue
            ahead = (dx * fwd[0] + dy * fwd[1]) / max(d, 1e-6)
            if ahead < self.ahead_dot and d > 6.0:
                continue
            if d < best_d:
                best_d, best_tl = d, tl

        if best_tl is None:
            self.state = LightState.GREEN   # queried, nothing constrains
            self.stop_distance_m = math.inf
            self.active_light_id = None
        else:
            self.state = self._map_state(self._raw_state(best_tl))
            self.stop_distance_m = best_d
            self.active_light_id = (best_tl.get("id") if isinstance(best_tl, dict)
                                    else getattr(best_tl, "id", None))
        return self.state

    def reset(self) -> None:
        self.state = LightState.UNKNOWN
        self.stop_distance_m = math.inf
        self.active_light_id = None

    # --------------------------------------------------------------- helpers
    def _vehicle_api(self, ego, ex: float, ey: float) -> Optional[LightState]:
        """carla.Vehicle exposes the light it is currently under — trust it."""
        try:
            if not (hasattr(ego, "is_at_traffic_light") and ego.is_at_traffic_light()):
                return None
            tl = ego.get_traffic_light()
            st = self._map_state(self._raw_state(tl) if tl is not None
                                 else ego.get_traffic_light_state())
            p = (self._stop_point(tl) or self._location(tl)) if tl is not None else None
            self.stop_distance_m = (math.hypot(p[0] - ex, p[1] - ey)
                                    if p is not None else math.inf)
            self.active_light_id = getattr(tl, "id", None)
            return st
        except Exception:
            return None

    @staticmethod
    def _map_state(st) -> LightState:
        if isinstance(st, LightState):
            return st
        name = getattr(st, "name", None) or str(st)
        return _STATE_MAP.get(name.lower(), LightState.UNKNOWN)

    @staticmethod
    def _raw_state(tl):
        if isinstance(tl, dict):
            return tl.get("state")
        st = getattr(tl, "state", None)
        if st is None and hasattr(tl, "get_state"):
            try:
                st = tl.get_state()
            except Exception:
                st = None
        return st

    def _stop_point(self, tl) -> Optional[Tuple[float, float]]:
        """Stop-line location: carla's stop waypoints, else trigger volume."""
        try:
            wps = tl.get_stop_waypoints()
            if wps:
                p = wps[0].transform.location
                return float(p.x), float(p.y)
        except Exception:
            pass
        tv = getattr(tl, "trigger_volume", None)
        if tv is not None and hasattr(tv, "location"):
            return float(tv.location.x), float(tv.location.y)
        return None

    @staticmethod
    def _location(tl) -> Optional[Tuple[float, float]]:
        try:
            loc = tl.get_location() if hasattr(tl, "get_location") else \
                tl.get_transform().location
            return float(loc.x), float(loc.y)
        except Exception:
            pass
        if all(hasattr(tl, k) for k in ("x", "y")):
            return float(tl.x), float(tl.y)
        if isinstance(tl, dict) and "x" in tl and "y" in tl:
            return float(tl["x"]), float(tl["y"])
        return None

    @staticmethod
    def _ego_xy_yaw(ego) -> Tuple[float, float, float]:
        if hasattr(ego, "get_transform"):
            t = ego.get_transform()
            return t.location.x, t.location.y, math.radians(t.rotation.yaw)
        if hasattr(ego, "rotation") and hasattr(ego, "location"):
            return ego.location.x, ego.location.y, math.radians(ego.rotation.yaw)
        if all(hasattr(ego, k) for k in ("x", "y", "yaw")):
            return float(ego.x), float(ego.y), float(ego.yaw)
        if isinstance(ego, dict):
            return float(ego["x"]), float(ego["y"]), float(ego.get("yaw", 0.0))
        return 0.0, 0.0, 0.0
