"""Contract tests for fsd.core.types — the shared data vocabulary.

These run with no simulator and no third-party deps; the invariants they pin
down (command bounds, vector math, trajectory emptiness) are what every
downstream module relies on.
"""
from __future__ import annotations

import math
import unittest

from fsd.core.types import (
    ControlCommand,
    Trajectory,
    Vec3,
    Waypoint,
)


class TestControlCommandClamp(unittest.TestCase):
    """ControlCommand.clamp() must always land values inside physical bounds."""

    def test_clamp_upper_bounds(self):
        cmd = ControlCommand(throttle=1.7, brake=2.0, steer=3.4).clamp()
        self.assertEqual(cmd.throttle, 1.0)
        self.assertEqual(cmd.brake, 1.0)
        self.assertEqual(cmd.steer, 1.0)

    def test_clamp_lower_bounds(self):
        cmd = ControlCommand(throttle=-0.5, brake=-1.0, steer=-2.0).clamp()
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.brake, 0.0)
        self.assertEqual(cmd.steer, -1.0)

    def test_clamp_preserves_in_range_values(self):
        cmd = ControlCommand(throttle=0.4, brake=0.1, steer=-0.25).clamp()
        self.assertAlmostEqual(cmd.throttle, 0.4)
        self.assertAlmostEqual(cmd.brake, 0.1)
        self.assertAlmostEqual(cmd.steer, -0.25)

    def test_clamp_is_in_place_and_returns_self(self):
        cmd = ControlCommand(throttle=5.0)
        self.assertIs(cmd.clamp(), cmd)
        self.assertEqual(cmd.throttle, 1.0)

    def test_defaults_are_safe(self):
        cmd = ControlCommand()
        self.assertEqual((cmd.throttle, cmd.brake, cmd.steer), (0.0, 0.0, 0.0))
        self.assertFalse(cmd.hand_brake)
        self.assertFalse(cmd.reverse)


class TestVec3(unittest.TestCase):
    def test_norm_zero(self):
        self.assertEqual(Vec3(0.0, 0.0, 0.0).norm(), 0.0)

    def test_norm_unit_axes(self):
        self.assertEqual(Vec3(1.0, 0.0, 0.0).norm(), 1.0)
        self.assertEqual(Vec3(0.0, -1.0, 0.0).norm(), 1.0)
        self.assertEqual(Vec3(0.0, 0.0, 1.0).norm(), 1.0)

    def test_norm_3_4_12(self):
        self.assertAlmostEqual(Vec3(3.0, 4.0, 12.0).norm(), 13.0)

    def test_z_defaults_to_zero(self):
        self.assertAlmostEqual(Vec3(3.0, 4.0).norm(), 5.0)

    def test_norm_is_nonnegative(self):
        v = Vec3(-2.0, -7.0, -11.0)
        self.assertGreater(v.norm(), 0.0)
        self.assertTrue(math.isfinite(v.norm()))


class TestTrajectory(unittest.TestCase):
    def test_empty_when_no_points(self):
        self.assertTrue(Trajectory(points=[], target_speed=0.0).empty)

    def test_not_empty_with_points(self):
        traj = Trajectory(points=[Waypoint(x=0.0, y=0.0), Waypoint(x=1.0, y=0.0)],
                          target_speed=5.0)
        self.assertFalse(traj.empty)

    def test_horizon_default(self):
        self.assertAlmostEqual(Trajectory(points=[], target_speed=0.0).horizon_s, 4.0)

    def test_empty_is_a_property(self):
        # .empty is a property, not a method — calling it must not be required.
        self.assertIsInstance(Trajectory(points=[], target_speed=0.0).empty, bool)


if __name__ == "__main__":
    unittest.main()
