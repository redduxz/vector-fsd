"""CARLA bridge — the hardware-abstraction layer between the FSD stack and
the CARLA simulator. Safe to import without the ``carla`` package installed;
errors surface lazily at connect/spawn time as :class:`CarlaUnavailableError`.
"""
from fsd.carla_bridge.world import (
    CarlaUnavailableError,
    CarlaWorld,
    require_carla,
)
from fsd.carla_bridge.sensors import (
    EVENT_SENSORS,
    STREAM_SENSORS,
    SensorEvent,
    SensorReading,
    SensorSuite,
)
from fsd.carla_bridge.vehicle import EgoVehicle
from fsd.carla_bridge.traffic import TrafficManager

__all__ = [
    "CarlaUnavailableError",
    "CarlaWorld",
    "require_carla",
    "SensorSuite",
    "SensorReading",
    "SensorEvent",
    "STREAM_SENSORS",
    "EVENT_SENSORS",
    "EgoVehicle",
    "TrafficManager",
]
