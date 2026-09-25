"""ObjectDetector CARLA ground-truth path — fake actors, no simulator.

Covers the wiring real runs depend on: world.get_actors() -> classified
DetectObjects with the ego vehicle excluded (by actor id, and by
proximity when no id is supplied).
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from fsd.perception.object_detector import ObjectDetector


class _Vec:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Tf:
    def __init__(self, x, y):
        self.location = _Vec(x, y)


class _BB:
    def __init__(self, ex, ey, ez):
        self.extent = _Vec(ex, ey, ez)


class _Actor:
    def __init__(self, aid, type_id, x, y, vx=0.0):
        self.id = aid
        self.type_id = type_id
        self._tf = _Tf(x, y)
        self._v = _Vec(vx, 0.0)
        self.bounding_box = _BB(2.2, 0.9, 0.8)

    def get_transform(self):
        return self._tf

    def get_velocity(self):
        return self._v


class _World:
    def __init__(self, actors):
        self._actors = actors

    def get_actors(self):
        return list(self._actors)


def _ego(x=0.0, y=0.0, yaw=0.0, aid=None):
    return {"x": x, "y": y, "yaw": yaw, "id": aid}


class TestCarlaPath(unittest.TestCase):
    def setUp(self):
        self.det = ObjectDetector(backend="none", max_range_m=120.0)

    def test_vehicles_and_walkers_classified(self):
        w = _World([
            _Actor(10, "vehicle.tesla.model3", 30.0, 0.0, vx=5.0),
            _Actor(11, "walker.pedestrian.0001", 20.0, 2.0),
            _Actor(12, "traffic.traffic_light", 40.0, -3.0),
            _Actor(13, "static.prop.box", 25.0, 5.0),
        ])
        objs = self.det.detect(world=w, ego=_ego(aid=99))
        by_id = {o.obj_id: o for o in objs}
        self.assertEqual(by_id[10].cls, "vehicle")
        self.assertAlmostEqual(by_id[10].velocity.x, 5.0)
        self.assertEqual(by_id[11].cls, "pedestrian")
        self.assertEqual(by_id[12].cls, "sign")
        self.assertEqual(by_id[13].cls, "misc")

    def test_ego_excluded_by_id(self):
        w = _World([_Actor(7, "vehicle.tesla.model3", 0.0, 0.0)])
        objs = self.det.detect(world=w, ego=_ego(aid=7))
        self.assertEqual(objs, [])

    def test_ego_excluded_by_proximity(self):
        # no id given — an actor sitting on the ego position is the ego
        w = _World([_Actor(7, "vehicle.tesla.model3", 0.3, 0.2)])
        objs = self.det.detect(world=w, ego=_ego(aid=None))
        self.assertEqual(objs, [])

    def test_range_filter(self):
        w = _World([
            _Actor(10, "vehicle.a", 50.0, 0.0),
            _Actor(11, "vehicle.b", 500.0, 0.0),
        ])
        objs = self.det.detect(world=w, ego=_ego(aid=99))
        self.assertEqual([o.obj_id for o in objs], [10])

    def test_bicycle_classified_as_cyclist(self):
        w = _World([_Actor(10, "vehicle.bh.crossbike", 10.0, 0.0)])
        objs = self.det.detect(world=w, ego=_ego(aid=99))
        self.assertEqual(objs[0].cls, "cyclist")


class TestTrafficLightMonitorActors(unittest.TestCase):
    """TL monitor over fake actor lists — states + stop distance."""

    def _tl(self, aid, x, y, state):
        # shape the monitor's duck-typing expects: .state + get_location()
        loc = _Vec(x, y)
        tl = SimpleNamespace(id=aid, state=SimpleNamespace(name=state),
                             get_location=lambda: loc,
                             get_stop_waypoints=lambda: [])
        return tl

    def test_nearest_ahead_wins(self):
        from fsd.perception.traffic_light import TrafficLightMonitor
        from fsd.core.types import LightState
        m = TrafficLightMonitor()
        lights = [self._tl(1, 30.0, 0.0, "green"),
                  self._tl(2, 15.0, 0.5, "red")]
        st = m.update({"x": 0, "y": 0, "yaw": 0.0}, traffic_lights=lights)
        self.assertEqual(st, LightState.RED)
        self.assertAlmostEqual(m.stop_distance_m, math.hypot(15.0, 0.5), places=1)

    def test_no_lights_is_green_not_unknown(self):
        from fsd.perception.traffic_light import TrafficLightMonitor
        from fsd.core.types import LightState
        m = TrafficLightMonitor()
        st = m.update({"x": 0, "y": 0, "yaw": 0.0}, traffic_lights=[])
        self.assertEqual(st, LightState.GREEN)


if __name__ == "__main__":
    unittest.main()
