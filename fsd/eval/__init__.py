"""Closed-loop evaluation — metrics, fault injection, reporting.

    from fsd.eval.metrics import ClosedLoopMetrics, RunMetrics, TickRecord
    from fsd.eval.faults import DropSensor, DelayStage, SpoofObject, FreezeHeartbeat
    from fsd.eval import report
"""
from fsd.eval.metrics import ClosedLoopMetrics, RunMetrics, TickRecord
from fsd.eval.faults import (
    DelayStage,
    DropSensor,
    Fault,
    FreezeHeartbeat,
    HeartbeatLostError,
    SensorDropoutError,
    SpoofObject,
)

__all__ = [
    "ClosedLoopMetrics",
    "RunMetrics",
    "TickRecord",
    "Fault",
    "DropSensor",
    "DelayStage",
    "SpoofObject",
    "FreezeHeartbeat",
    "SensorDropoutError",
    "HeartbeatLostError",
]
