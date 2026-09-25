"""C++/Python parity tests — does ``fsd_cpp`` agree with the reference stack?

Two tiers:

* ``TestAdapterContract`` always runs — it pins the ``fsd.compat`` adapter
  contract (methods exist, outputs are ``fsd.core.types``, fail-safe
  behavior) whichever backend is active. With no extension built it
  exercises the Python fallback path.
* ``TestCppParity`` runs only when ``fsd_cpp`` is importable — it drives the
  C++ backend and the Python reference through a battery of synthetic
  scenarios (nominal, TTC breach, speed violation, watchdog stall, lane
  departure) and asserts the arbitrated ``DriveMode`` matches, plus
  controller output agreement within tolerance.

Both skip cleanly — no C++ tree required for the suite to stay green.
"""
from __future__ import annotations

import time
import unittest

from fsd.compat import backend, has_cpp
from fsd.compat.control import CppVehicleController
from fsd.compat.safety import CppSafetyMonitor
from fsd.control.controller import VehicleController
from fsd.core.config import Config
from fsd.core.types import (
    ControlCommand,
    DetectedObject,
    DriveMode,
    LaneInfo,
    PerceptionOutput,
    Trajectory,
    Vec3,
    VehicleState,
    Waypoint,
)
from fsd.safety.monitor import SafetyMonitor


# ---------------------------------------------------------------------------
# Synthetic scene builders (mirrors tests/test_safety.py conventions)
# ---------------------------------------------------------------------------

def _state(speed: float = 5.0, stale: bool = False) -> VehicleState:
    ts = time.time() - (5.0 if stale else 0.0)
    return VehicleState(x=0.0, y=0.0, z=0.0, yaw=0.0, speed=speed,
                        accel=0.0, steer=0.0, timestamp=ts)


def _lane(center_offset: float = 0.0, detected: bool = True) -> LaneInfo:
    return LaneInfo(left_offset=1.75, right_offset=-1.75,
                    center_offset=center_offset, heading_error=0.0,
                    curvature=0.0, detected=detected)


def _perception(objects=None, free_space: float = 100.0,
                stale: bool = False, lane: LaneInfo | None = None
                ) -> PerceptionOutput:
    ts = time.time() - (5.0 if stale else 0.0)
    return PerceptionOutput(objects=objects or [],
                            lane=lane if lane is not None else _lane(),
                            free_space_ahead=free_space, timestamp=ts)


def _lead_vehicle(distance_m: float = 5.0, speed_mps: float = 0.0
                  ) -> DetectedObject:
    return DetectedObject(
        obj_id=1, cls="vehicle",
        position=Vec3(x=distance_m, y=0.0, z=0.0),
        velocity=Vec3(x=speed_mps, y=0.0, z=0.0),
        bbox_extent=Vec3(x=2.2, y=0.9, z=0.8),
        confidence=0.95)


def _trajectory(target_speed: float = 8.0, curve: float = 0.0) -> Trajectory:
    pts = []
    for i in range(1, 41):
        x = float(i) * 2.0
        y = curve * x * x * 0.01
        pts.append(Waypoint(x=x, y=y, yaw=0.0))
    return Trajectory(points=pts, target_speed=target_speed)


# ---------------------------------------------------------------------------
# Adapter contract — backend-agnostic, always runs
# ---------------------------------------------------------------------------

class TestAdapterContract(unittest.TestCase):
    """The adapter must be a faithful SafetyMonitor/VehicleController facade
    regardless of which backend it is serving."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()

    def test_monitor_constructs_and_reports_backend(self):
        mon = CppSafetyMonitor(self.cfg)
        self.assertIn(mon.backend, ("cpp", "python"))
        self.assertEqual(mon.backend, backend())

    def test_monitor_check_returns_drivemode(self):
        mon = CppSafetyMonitor(self.cfg)
        mon.heartbeat()
        mode = mon.check(_state(5.0), _perception(),
                         ControlCommand(throttle=0.2), True)
        self.assertIsInstance(mode, DriveMode)

    def test_monitor_safe_stop_shape(self):
        mon = CppSafetyMonitor(self.cfg)
        cmd = mon.engage_safe_stop()
        self.assertIsInstance(cmd, ControlCommand)
        self.assertGreaterEqual(cmd.brake, 0.9)
        self.assertLessEqual(cmd.throttle, 0.01)

    def test_monitor_enforce_returns_command(self):
        mon = CppSafetyMonitor(self.cfg)
        mon.heartbeat()
        mon.check(_state(5.0), _perception(), ControlCommand(), True)
        out = mon.enforce(ControlCommand(throttle=2.0, steer=5.0))
        self.assertIsInstance(out, ControlCommand)
        self.assertLessEqual(out.throttle, 1.0)
        self.assertLessEqual(out.steer, 1.0)

    def test_monitor_status_and_reset(self):
        mon = CppSafetyMonitor(self.cfg)
        st = mon.status()
        self.assertIn("mode", st)
        mon.reset()
        self.assertEqual(mon.mode, DriveMode.ENGAGED)

    def test_controller_constructs_and_computes(self):
        ctl = CppVehicleController(cfg=self.cfg)
        self.assertIn(ctl.backend, ("cpp", "python"))
        cmd = ctl.compute(_trajectory(8.0), _state(5.0), _perception())
        self.assertIsInstance(cmd, ControlCommand)
        for v, lo, hi in ((cmd.throttle, 0.0, 1.0), (cmd.brake, 0.0, 1.0),
                          (cmd.steer, -1.0, 1.0)):
            self.assertGreaterEqual(v, lo)
            self.assertLessEqual(v, hi)

    def test_controller_empty_trajectory_brakes(self):
        ctl = CppVehicleController(cfg=self.cfg)
        cmd = ctl.compute(Trajectory(points=[], target_speed=0.0),
                          _state(5.0), _perception())
        self.assertIsInstance(cmd, ControlCommand)
        self.assertGreaterEqual(cmd.brake, 0.5)
        self.assertLessEqual(cmd.throttle, 0.05)


# ---------------------------------------------------------------------------
# C++ parity — only when the extension is importable
# ---------------------------------------------------------------------------

@unittest.skipUnless(has_cpp(), "fsd_cpp extension not built — skipping parity")
class TestCppParity(unittest.TestCase):
    """The C++ backend must arbitrate the same DriveMode as the Python
    reference implementation on every scenario."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()

    def _monitor_pair(self):
        py = SafetyMonitor(self.cfg.safety)
        cpp = CppSafetyMonitor(self.cfg)
        # Guard against a silent ctor fallback: parity is meaningless if the
        # "cpp" side is secretly the python monitor.
        self.assertEqual(cpp.backend, "cpp",
                         "CppSafetyMonitor fell back to python — "
                         "fsd_cpp.SafetyMonitor unusable")
        return py, cpp

    def _controller_pair(self):
        py = VehicleController()
        cpp = CppVehicleController(cfg=self.cfg)
        self.assertEqual(cpp.backend, "cpp",
                         "CppVehicleController fell back to python — "
                         "fsd_cpp.VehicleController unusable")
        return py, cpp

    def _assert_mode_parity(self, ego, perception, cmd, alive=True):
        py, cpp = self._monitor_pair()
        py_mode = py.check(ego, perception, cmd, alive)
        cpp_mode = cpp.check(ego, perception, cmd, alive)
        self.assertIsInstance(cpp_mode, DriveMode)
        self.assertEqual(
            py_mode, cpp_mode,
            f"mode mismatch: python={py_mode.name} cpp={cpp_mode.name}")

    # --- monitor scenarios -------------------------------------------------

    def test_nominal_cruise_engaged(self):
        self._assert_mode_parity(_state(5.0), _perception(),
                                 ControlCommand(throttle=0.2))

    def test_ttc_breach_safe_stop(self):
        self._assert_mode_parity(
            _state(12.0),
            _perception(objects=[_lead_vehicle(5.0)], free_space=5.0),
            ControlCommand(throttle=0.4))

    def test_speed_warning_degraded(self):
        """Over the cap but under the hard factor -> warning/DEGRADED."""
        speed = self.cfg.safety.max_speed_mps + 2.0
        self._assert_mode_parity(_state(speed), _perception(),
                                 ControlCommand(throttle=0.5))

    def test_speed_hard_violation_safe_stop(self):
        speed = self.cfg.safety.max_speed_mps * 1.5
        self._assert_mode_parity(_state(speed), _perception(),
                                 ControlCommand(throttle=0.5))

    def test_watchdog_pipeline_dead(self):
        self._assert_mode_parity(_state(5.0), _perception(),
                                 ControlCommand(throttle=0.2),
                                 alive=False)

    def test_watchdog_stale_inputs(self):
        self._assert_mode_parity(_state(5.0, stale=True),
                                 _perception(stale=True),
                                 ControlCommand(throttle=0.2))

    def test_free_space_blocked(self):
        self._assert_mode_parity(_state(8.0), _perception(free_space=2.0),
                                 ControlCommand(throttle=0.5))

    def test_lane_departure(self):
        lane = _lane(center_offset=3.0)   # beyond half-width + exit margin
        self._assert_mode_parity(_state(10.0), _perception(lane=lane),
                                 ControlCommand(throttle=0.3))

    def test_engage_safe_stop_parity(self):
        py, cpp = self._monitor_pair()
        p_cmd = py.engage_safe_stop()
        c_cmd = cpp.engage_safe_stop()
        self.assertIsInstance(c_cmd, ControlCommand)
        self.assertAlmostEqual(p_cmd.brake, c_cmd.brake, delta=0.05)
        self.assertAlmostEqual(p_cmd.throttle, c_cmd.throttle, delta=0.05)

    # --- controller scenarios ------------------------------------------------

    def _assert_cmd_parity(self, traj, ego, perception, delta=0.25):
        py, cpp = self._controller_pair()
        p = py.compute(traj, ego, perception)
        c = cpp.compute(traj, ego, perception)
        self.assertIsInstance(c, ControlCommand)
        for name in ("throttle", "brake", "steer"):
            self.assertAlmostEqual(
                getattr(p, name), getattr(c, name), delta=delta,
                msg=f"{name} diverged: python={getattr(p, name):.3f} "
                    f"cpp={getattr(c, name):.3f}")
        self.assertEqual(p.hand_brake, c.hand_brake)
        self.assertEqual(p.reverse, c.reverse)
        return p, c

    def test_controller_cruise(self):
        self._assert_cmd_parity(_trajectory(10.0), _state(8.0), _perception())

    def test_controller_overspeed_brakes(self):
        self._assert_cmd_parity(_trajectory(4.0), _state(14.0), _perception())

    def test_controller_curve(self):
        self._assert_cmd_parity(_trajectory(8.0, curve=1.0), _state(8.0),
                                _perception())

    def test_controller_empty_trajectory(self):
        self._assert_cmd_parity(Trajectory(points=[], target_speed=0.0),
                                _state(5.0), _perception())


if __name__ == "__main__":
    unittest.main()
