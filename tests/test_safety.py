"""Safety-layer tests on synthetic states — no simulator required.

Two tiers:

* ``TestSafetyRules`` exercises ``fsd.safety.rules`` directly — the checks the
  monitor runs every cycle (TTC, speed cap, free space, watchdog).
* ``TestSafetyMonitor`` drives ``fsd.safety.monitor.SafetyMonitor`` through one
  gate cycle at a time and asserts on the arbitrated ``DriveMode`` plus the
  enforced ``ControlCommand``.

If ``fsd.safety.monitor`` has not landed yet, the monitor tier skips cleanly;
the rules tier runs as soon as ``fsd.safety.rules`` exists.
"""
from __future__ import annotations

import time
import unittest

from fsd.core.config import Config
from fsd.core.types import (
    ControlCommand,
    DetectedObject,
    DriveMode,
    LaneInfo,
    PerceptionOutput,
    Vec3,
    VehicleState,
)

try:
    from fsd.safety.monitor import SafetyMonitor  # type: ignore
except ImportError:  # module not landed yet (parallel development)
    SafetyMonitor = None

try:
    from fsd.safety.rules import (  # type: ignore
        FreeSpaceRule,
        SafetyContext,
        SpeedLimitRule,
        TTCRule,
        WatchdogRule,
        default_rules,
    )
    _HAS_RULES = True
except ImportError:
    _HAS_RULES = False

_CHECK_NAMES = ("check", "gate", "evaluate", "monitor", "step", "update")


def setUpModule():
    if SafetyMonitor is None and not _HAS_RULES:
        raise unittest.SkipTest("fsd.safety layer not implemented yet")


def _make_monitor(cfg: Config):
    """Construct a SafetyMonitor under the common constructor conventions."""
    assert SafetyMonitor is not None
    last_err: Exception | None = None
    for args in ((cfg.safety,), (cfg,), ()):  # SafetyConfig, Config, or no-arg
        try:
            return SafetyMonitor(*args)
        except TypeError as exc:
            last_err = exc
            continue
    raise AssertionError(f"SafetyMonitor did not accept any known ctor shape: {last_err}")


def _run_check(monitor, state, perception, cmd):
    """Run one arbitration cycle, whatever the monitor calls it.

    Signature order puts the real contract — (ego, perception, cmd) — first,
    then command-first and keyword variants for other conventions.
    """
    fn = None
    for name in _CHECK_NAMES:
        cand = getattr(monitor, name, None)
        if callable(cand):
            fn = cand
            break
    if fn is None:
        raise AssertionError(
            f"SafetyMonitor exposes none of {_CHECK_NAMES}; update the test contract.")
    signatures = (
        ((state, perception, cmd), {}),          # check(ego, perception, cmd)
        ((cmd, state, perception), {}),          # gate(cmd, state, perception)
        ((), {"ego": state, "perception": perception, "cmd": cmd}),
        ((), {"ego": state, "perception": perception, "command": cmd}),
        ((state, perception), {}),
    )
    last_err: Exception | None = None
    for args, kwargs in signatures:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_err = exc
            continue
    raise AssertionError(f"check call failed for all known signatures: {last_err}")


def _run_enforce(monitor, cmd, fallback):
    """Apply the monitor's enforcement phase if it is a separate step."""
    for name in ("enforce", "filter", "shape"):
        fn = getattr(monitor, name, None)
        if callable(fn):
            try:
                return fn(cmd)
            except TypeError:
                continue
    # engage_safe_stop produces the fallback command directly
    if getattr(monitor, "mode", None) is DriveMode.SAFE_STOP:
        stop_fn = getattr(monitor, "engage_safe_stop", None)
        if callable(stop_fn):
            return stop_fn()
    return fallback


def _extract_mode(result):
    if isinstance(result, DriveMode):
        return result
    if isinstance(result, str) and result.upper() in DriveMode.__members__:
        return result
    for attr in ("mode", "drive_mode"):
        val = getattr(result, attr, None)
        if isinstance(val, (DriveMode, str)):
            return val
    return None


def _extract_cmd(result, fallback):
    if isinstance(result, ControlCommand):
        return result
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, ControlCommand):
                return item
    for attr in ("command", "cmd", "control", "output"):
        val = getattr(result, attr, None)
        if isinstance(val, ControlCommand):
            return val
    return fallback


def _extract_allowed(result):
    if isinstance(result, bool):
        return result
    for attr in ("allowed", "ok", "passed", "safe"):
        val = getattr(result, attr, None)
        if isinstance(val, bool):
            return val
    return None


def _is_safe_stop(cmd, mode, allowed) -> bool:
    """True if the gate's answer is an unambiguous minimum-risk stop."""
    if allowed is False:
        return True
    name = mode.name if isinstance(mode, DriveMode) else str(mode or "").upper()
    if "SAFE_STOP" in name or "EMERGENCY" in name or "ESTOP" in name:
        return True
    return (
        cmd is not None
        and getattr(cmd, "brake", 0.0) >= 0.5
        and getattr(cmd, "throttle", 0.0) <= 0.05
    )


def _state(speed: float = 5.0, stale: bool = False) -> VehicleState:
    ts = time.time() - (5.0 if stale else 0.0)
    return VehicleState(x=0.0, y=0.0, z=0.0, yaw=0.0, speed=speed,
                        accel=0.0, steer=0.0, timestamp=ts)


def _lane() -> LaneInfo:
    return LaneInfo(left_offset=1.75, right_offset=-1.75, center_offset=0.0,
                    heading_error=0.0, curvature=0.0)


def _perception(objects=None, free_space: float = 100.0, stale: bool = False) -> PerceptionOutput:
    ts = time.time() - (5.0 if stale else 0.0)
    return PerceptionOutput(objects=objects or [], lane=_lane(),
                            free_space_ahead=free_space, timestamp=ts)


def _lead_vehicle(distance_m: float = 5.0, speed_mps: float = 0.0) -> DetectedObject:
    """Vehicle directly ahead on the +x lane, moving at ``speed_mps``."""
    return DetectedObject(
        obj_id=1, cls="vehicle",
        position=Vec3(x=distance_m, y=0.0, z=0.0),
        velocity=Vec3(x=speed_mps, y=0.0, z=0.0),
        bbox_extent=Vec3(x=2.2, y=0.9, z=0.8),
        confidence=0.95,
    )


@unittest.skipIf(SafetyMonitor is None,
                 "fsd.safety.monitor.SafetyMonitor not implemented yet")
class TestSafetyMonitor(unittest.TestCase):
    """Behavioral tests against the gate's documented safety envelope."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()

    def setUp(self):
        self.monitor = _make_monitor(self.cfg)

    def evaluate(self, cmd, state, perception):
        """One gate cycle -> (enforced_command, mode, allowed)."""
        result = _run_check(self.monitor, state, perception, cmd)
        mode = _extract_mode(result)
        allowed = _extract_allowed(result)
        out_cmd = _extract_cmd(result, cmd)
        out_cmd = _run_enforce(self.monitor, cmd, out_cmd)
        if mode is None:
            mode = getattr(self.monitor, "mode", None)
        return out_cmd, mode, allowed

    # --- nominal -----------------------------------------------------------

    def test_clean_command_passes_through(self):
        cmd = ControlCommand(throttle=0.3, steer=0.0)
        out_cmd, mode, allowed = self.evaluate(cmd, _state(speed=5.0), _perception())
        self.assertFalse(_is_safe_stop(out_cmd, mode, allowed),
                         "a clean request must not trigger the stop tier")
        if allowed is not None:
            self.assertTrue(allowed)

    def test_output_command_within_bounds(self):
        cmd = ControlCommand(throttle=2.0, brake=-1.0, steer=3.0)
        out_cmd, _mode, _allowed = self.evaluate(cmd, _state(speed=5.0), _perception())
        self.assertIsNotNone(out_cmd)
        self.assertLessEqual(out_cmd.throttle, 1.0)
        self.assertGreaterEqual(out_cmd.throttle, 0.0)
        self.assertGreaterEqual(out_cmd.brake, 0.0)
        self.assertLessEqual(out_cmd.brake, 1.0)
        self.assertGreaterEqual(out_cmd.steer, -1.0)
        self.assertLessEqual(out_cmd.steer, 1.0)

    # --- speed cap ---------------------------------------------------------

    def test_speeding_command_is_clamped(self):
        """At or above max_speed_mps a full-throttle request must be cut."""
        speed = self.cfg.safety.max_speed_mps + 4.0
        cmd = ControlCommand(throttle=1.0)
        out_cmd, mode, allowed = self.evaluate(cmd, _state(speed=speed), _perception())
        cut = out_cmd.throttle <= 0.05 or out_cmd.brake > 0.0
        self.assertTrue(
            cut or _is_safe_stop(out_cmd, mode, allowed) or allowed is False,
            f"gate left throttle={out_cmd.throttle} at speed {speed:.1f} m/s "
            f"(cap {self.cfg.safety.max_speed_mps})",
        )

    # --- TTC / free space --------------------------------------------------

    def test_ttc_breach_triggers_safe_stop(self):
        """Lead vehicle ~0.03 s bumper gap at 12 m/s — far under min_ttc_s."""
        cmd = ControlCommand(throttle=0.4)
        perception = _perception(objects=[_lead_vehicle(5.0)], free_space=5.0)
        out_cmd, mode, allowed = self.evaluate(cmd, _state(speed=12.0), perception)
        self.assertTrue(
            _is_safe_stop(out_cmd, mode, allowed),
            "sub-minimum TTC did not produce a safe stop",
        )

    def test_insufficient_free_space_blocks_motion(self):
        cmd = ControlCommand(throttle=0.5)
        out_cmd, mode, allowed = self.evaluate(
            cmd, _state(speed=8.0), _perception(free_space=2.0))
        intervened = (
            _is_safe_stop(out_cmd, mode, allowed)
            or out_cmd.brake > 0.0
            or out_cmd.throttle < 0.05
            or allowed is False
        )
        self.assertTrue(intervened, "gate allowed forward motion with 2 m free space")

    # --- watchdog ----------------------------------------------------------

    def test_stale_pipeline_triggers_safe_stop(self):
        """Timestamps older than watchdog_timeout_s (0.5 s) must not drive."""
        cmd = ControlCommand(throttle=0.3)
        out_cmd, mode, allowed = self.evaluate(
            cmd, _state(speed=5.0, stale=True), _perception(stale=True))
        self.assertTrue(
            _is_safe_stop(out_cmd, mode, allowed),
            "stale pipeline data did not produce a safe stop",
        )

    def test_fresh_data_does_not_trip_watchdog(self):
        cmd = ControlCommand(throttle=0.2)
        out_cmd, mode, allowed = self.evaluate(cmd, _state(), _perception())
        self.assertFalse(_is_safe_stop(out_cmd, mode, allowed),
                         "fresh inputs should never look like a stall")


@unittest.skipUnless(_HAS_RULES, "fsd.safety.rules not implemented yet")
class TestSafetyRules(unittest.TestCase):
    """Direct rule checks — the logic the monitor arbitrates each cycle."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()

    def ctx(self, state, perception, cmd, pipeline_alive=True, heartbeat=None):
        return SafetyContext(
            ego=state, perception=perception, cmd=cmd, applied=None,
            pipeline_alive=pipeline_alive, now=time.time(), dt=0.05,
            cfg=self.cfg.safety,
            last_heartbeat=heartbeat if heartbeat is not None else time.time(),
        )

    def test_default_rule_set_is_nonempty_and_ordered(self):
        rules = default_rules()
        names = [r.name for r in rules]
        self.assertIn("ttc", names)
        self.assertIn("watchdog", names)
        self.assertLess(names.index("speed_limit"), names.index("steer_rate"),
                        "accel caps must shape the command before steer slew")

    def test_ttc_rule_flags_critical_below_floor(self):
        ev = TTCRule().evaluate(
            self.ctx(_state(speed=12.0),
                     _perception(objects=[_lead_vehicle(5.0)]),
                     ControlCommand(throttle=0.4)))
        self.assertIsNotNone(ev, "TTC rule did not fire on a ~0.03 s gap")
        self.assertEqual(ev.level, "critical")

    def test_ttc_rule_quiet_when_lead_recedes(self):
        ev = TTCRule().evaluate(
            self.ctx(_state(speed=5.0),
                     _perception(objects=[_lead_vehicle(20.0, speed_mps=15.0)]),
                     ControlCommand(throttle=0.3)))
        self.assertIsNone(ev, "receding lead vehicle should not trip TTC")

    def test_speed_rule_enforce_cuts_throttle_over_cap(self):
        over = self.cfg.safety.max_speed_mps + 4.0
        shaped = SpeedLimitRule().enforce(
            ControlCommand(throttle=1.0),
            self.ctx(_state(speed=over), _perception(), ControlCommand()))
        self.assertEqual(shaped.throttle, 0.0)
        self.assertGreater(shaped.brake, 0.0)

    def test_speed_rule_enforce_bounds_accel_demand(self):
        shaped = SpeedLimitRule().enforce(
            ControlCommand(throttle=1.0),
            self.ctx(_state(speed=5.0), _perception(), ControlCommand()))
        self.assertLessEqual(shaped.throttle, 1.0)
        self.assertLess(shaped.throttle, 1.0,
                        "accel budget should bound the unitless demand")

    def test_free_space_rule_flags_critical(self):
        ev = FreeSpaceRule().evaluate(
            self.ctx(_state(speed=8.0), _perception(free_space=2.0),
                     ControlCommand()))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.level, "critical")

    def test_watchdog_flags_stale_inputs(self):
        ev = WatchdogRule().evaluate(
            self.ctx(_state(stale=True), _perception(stale=True),
                     ControlCommand()))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.level, "critical")

    def test_watchdog_flags_dead_pipeline_flag(self):
        ev = WatchdogRule().evaluate(
            self.ctx(_state(), _perception(), ControlCommand(),
                     pipeline_alive=False))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.level, "critical")


if __name__ == "__main__":
    unittest.main()
