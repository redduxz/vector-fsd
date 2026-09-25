"""fsd.safety — the watchdog layer between the planner and the actuators.

The monitor arbitrates DriveMode from pluggable SafetyRules, clamps every
command that passes through it, and latches into SAFE_STOP on critical
violations. Diagnostics watches the watchers; FallbackController executes the
minimal-risk maneuver when the monitor says stop.
"""
from fsd.safety.diagnostics import Diagnostics, FaultInjectionError
from fsd.safety.fallback import FallbackController
from fsd.safety.monitor import SafetyMonitor
from fsd.safety.rules import (
    SEVERITY_ORDER,
    FreeSpaceRule,
    LaneDepartureRule,
    SafetyContext,
    SafetyRule,
    SpeedLimitRule,
    SteerRateRule,
    TTCRule,
    WatchdogRule,
    default_rules,
)

__all__ = [
    "Diagnostics",
    "FaultInjectionError",
    "FallbackController",
    "SafetyMonitor",
    "SafetyContext",
    "SafetyRule",
    "TTCRule",
    "SpeedLimitRule",
    "FreeSpaceRule",
    "WatchdogRule",
    "SteerRateRule",
    "LaneDepartureRule",
    "SEVERITY_ORDER",
    "default_rules",
]
