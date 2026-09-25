"""Smoke tests for the planning stack on synthetic inputs.

BehaviorPlanner.decide(perception, ego) -> (state_name, target_speed)
TrajectoryPlanner.generate(route, decision, perception, ego) -> Trajectory
"""
from __future__ import annotations

import unittest

from fsd.core.types import (
    DetectedObject,
    LaneInfo,
    PerceptionOutput,
    Vec3,
    VehicleState,
    Waypoint,
)
from fsd.planning.behavior import BehaviorPlanner
from fsd.planning.trajectory import TrajectoryPlanner


def _state(speed: float = 8.0) -> VehicleState:
    return VehicleState(x=0.0, y=0.0, z=0.0, yaw=0.0, speed=speed,
                        accel=0.0, steer=0.0)


def _lane() -> LaneInfo:
    return LaneInfo(left_offset=1.75, right_offset=-1.75, center_offset=0.0,
                    heading_error=0.0, curvature=0.0)


def _perception(objects=None, free_space: float = 100.0) -> PerceptionOutput:
    return PerceptionOutput(objects=objects or [], lane=_lane(),
                            free_space_ahead=free_space)


def _route(n: int = 10) -> list:
    return [Waypoint(x=float(i) * 5.0, y=0.0) for i in range(n)]


def _lead_vehicle(distance_m: float = 12.0) -> DetectedObject:
    return DetectedObject(
        obj_id=1, cls="vehicle",
        position=Vec3(x=distance_m, y=0.0, z=0.0),
        velocity=Vec3(x=2.0, y=0.0, z=0.0),   # slow lead vehicle
        bbox_extent=Vec3(x=2.2, y=0.9, z=0.8),
        confidence=0.95,
    )


class TestBehaviorPlanner(unittest.TestCase):
    """The planner must always return a valid (state, speed) pair."""

    def setUp(self):
        self.planner = BehaviorPlanner()

    def test_nominal_scene_returns_valid_decision(self):
        name, speed = self.planner.decide(_perception(), _state())
        self.assertIn(name, BehaviorPlanner.STATES)
        self.assertGreaterEqual(speed, 0.0)
        self.assertEqual(name, BehaviorPlanner.LANE_KEEP)
        self.assertGreater(speed, 0.0)

    def test_blocked_lane_returns_valid_decision(self):
        # lead at 60m — comfortably outside TTC/follow-enter range
        far = _lead_vehicle(distance_m=60.0)
        name, speed = self.planner.decide(
            _perception(objects=[far]), _state())
        self.assertIn(name, BehaviorPlanner.STATES)
        self.assertIn(name, (BehaviorPlanner.FOLLOW_LEAD,
                             BehaviorPlanner.LANE_KEEP))
        self.assertGreaterEqual(speed, 0.0)

    def test_emergency_on_close_fast_approach(self):
        ego = _state(speed=15.0)
        near = _lead_vehicle(distance_m=5.0)
        near.velocity = Vec3(x=0.0, y=0.0, z=0.0)  # stationary obstacle
        name, speed = self.planner.decide(
            _perception(objects=[near]), ego)
        self.assertEqual(name, BehaviorPlanner.EMERGENCY_STOP)
        self.assertEqual(speed, 0.0)

    def test_junction_approach_tapers_speed(self):
        # 10m before a junction: factor = 0.55 + 0.45*(10/30) = 0.70
        p = _perception()
        p.junction_dist = 10.0
        name, speed = self.planner.decide(p, _state())
        self.assertEqual(name, BehaviorPlanner.LANE_KEEP)
        self.assertAlmostEqual(
            speed, self.planner.cruise_speed * 0.70, places=3)

    def test_inside_junction_uses_full_factor(self):
        p = _perception()
        p.junction_dist = 0.0
        _, speed = self.planner.decide(p, _state())
        self.assertAlmostEqual(
            speed, self.planner.cruise_speed
            * self.planner.junction_factor, places=3)

    def test_beyond_zone_no_cap(self):
        p = _perception()
        p.junction_dist = 60.0
        _, speed = self.planner.decide(p, _state())
        self.assertAlmostEqual(
            speed, self.planner.cruise_speed, places=3)

    def test_junction_dist_walks_waypoints(self):
        from fsd.agents.autopilot import AutopilotAgent

        class _Wp:
            def __init__(self, junction=False, ahead=None):
                self.is_junction = junction
                self._ahead = ahead or []

            def next(self, d):
                return self._ahead if d >= 4.0 else []

        # junction wp sits 8m ahead on the lane
        deep = _Wp(junction=True)
        wp = _Wp(ahead=[deep])
        self.assertEqual(AutopilotAgent._junction_dist(wp), 4.0)
        self.assertEqual(
            AutopilotAgent._junction_dist(_Wp(junction=True)), 0.0)
        self.assertEqual(
            AutopilotAgent._junction_dist(_Wp()), float("inf"))


class TestTrajectoryPlanner(unittest.TestCase):
    """Trajectory generation must produce forward waypoints or an empty
    trajectory — never garbage."""

    def setUp(self):
        self.planner = TrajectoryPlanner()

    def test_nominal_scene_produces_waypoints(self):
        traj = self.planner.generate(
            _route(), (BehaviorPlanner.LANE_KEEP, 13.9),
            _perception(), _state())
        self.assertGreater(len(traj.points), 0)
        for wp in traj.points:
            self.assertTrue(hasattr(wp, "x") and hasattr(wp, "y"))
            self.assertTrue(all(v == v and abs(v) < 1e6 for v in (wp.x, wp.y)))

    def test_target_speed_nonnegative(self):
        traj = self.planner.generate(
            _route(), (BehaviorPlanner.LANE_KEEP, 13.9),
            _perception(), _state())
        self.assertGreaterEqual(traj.target_speed, 0.0)

    def test_blocked_scene_still_returns_valid_trajectory(self):
        traj = self.planner.generate(
            _route(), (BehaviorPlanner.FOLLOW_LEAD, 5.0),
            _perception(objects=[_lead_vehicle()]), _state())
        self.assertIsNotNone(traj.points)
        self.assertGreaterEqual(traj.target_speed, 0.0)

    def test_emergency_stop_profile_decelerates(self):
        traj = self.planner.generate(
            _route(), (BehaviorPlanner.EMERGENCY_STOP, 0.0),
            _perception(free_space=30.0), _state(speed=15.0))
        speeds = [w.speed_limit for w in traj.points]
        self.assertTrue(all(s >= 0.0 for s in speeds))
        self.assertLessEqual(speeds[-1], speeds[0] + 1e-6)


if __name__ == "__main__":
    unittest.main()
