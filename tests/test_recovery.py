"""Stuck-recovery path: powered-but-motionless -> reverse manoeuvre.

Drives the smoke-mode autopilot with the ego speed pinned to zero so the
wedge detector accumulates, then verifies the agent switches to a reverse
ControlCommand with counter-steer for the recovery window.
"""
from __future__ import annotations

import unittest

from fsd.core.config import Config
from fsd.core.types import VehicleState
from fsd.agents.autopilot import AutopilotAgent


def _frozen_state(v=0.0):
    import time
    return VehicleState(x=0.0, y=0.0, z=0.0, yaw=0.0, speed=v,
                        accel=0.0, steer=0.0, timestamp=time.time())


class TestStuckRecovery(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        self.agent = AutopilotAgent(cfg, smoke=True)
        self.agent.setup()
        # ego reports motionless forever — simulates being wedged
        self.agent._ego_state = lambda: _frozen_state()

    def tearDown(self):
        try:
            self.agent.cleanup()
        except Exception:
            pass

    def _countdown_ticks(self):
        return max(1, int(3.0 / self.agent.dt))

    def test_reverse_engages_after_wedge_window(self):
        saw_reverse = None
        for _ in range(self._countdown_ticks() + 5):
            r = self.agent.tick()
            cmd = r.get("cmd")
            if cmd is not None and getattr(cmd, "reverse", False):
                saw_reverse = cmd
                break
        self.assertIsNotNone(
            saw_reverse,
            "no reverse command within stuck window + recovery start")
        self.assertGreater(saw_reverse.throttle, 0.3)

    def test_stuck_counter_accumulates_and_resets(self):
        n = self._countdown_ticks()
        for _ in range(n - 1):
            self.agent.tick()
        self.assertGreater(self.agent._stuck_ticks, 0)
        self.agent.tick()          # triggers recovery -> counter cleared
        self.assertEqual(self.agent._stuck_ticks, 0)
        self.assertGreater(self.agent._recover_ticks, 0)

    def test_moving_ego_never_triggers_recovery(self):
        self.agent._ego_state = lambda: _frozen_state(v=3.0)
        for _ in range(self._countdown_ticks() + 10):
            self.agent.tick()
        self.assertEqual(self.agent._recover_ticks, 0)
        self.assertEqual(self.agent._stuck_ticks, 0)


if __name__ == "__main__":
    unittest.main()
