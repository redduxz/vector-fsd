"""Closed-loop scenario testing — scripted worlds + verdicts.

    from fsd.scenarios import build, SCENARIOS
    from fsd.scenarios.runner import ScenarioRunner

or from the shell::

    python -m fsd.scenarios.runner --all
"""
from fsd.scenarios.base import (
    Event,
    Scenario,
    ScenarioEvent,
    ScenarioScene,
    ScriptedActor,
    Verdict,
)
from fsd.scenarios.library import (
    JaywalkerPedestrian,
    LeadVehicleCutIn,
    PlannerStall,
    RedLightRunner,
    SCENARIOS,
    SensorDropout,
    SuddenBraking,
    build,
)

__all__ = [
    "Event",
    "Scenario",
    "ScenarioEvent",
    "ScenarioScene",
    "ScriptedActor",
    "Verdict",
    "JaywalkerPedestrian",
    "LeadVehicleCutIn",
    "PlannerStall",
    "RedLightRunner",
    "SensorDropout",
    "SuddenBraking",
    "SCENARIOS",
    "build",
]
