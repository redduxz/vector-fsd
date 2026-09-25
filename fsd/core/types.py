"""Shared data contracts for the FSD stack. Every module talks in these types."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class LaneChangeState(Enum):
    KEEP = auto()
    LEFT = auto()
    RIGHT = auto()


class LightState(Enum):
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    UNKNOWN = auto()


class DriveMode(Enum):
    ENGAGED = auto()
    DEGRADED = auto()
    SAFE_STOP = auto()
    DISENGAGED = auto()


@dataclass
class Vec3:
    x: float
    y: float
    z: float = 0.0

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@dataclass
class VehicleState:
    """Ego-vehicle kinematic state, world frame."""
    x: float
    y: float
    z: float
    yaw: float            # radians
    speed: float          # m/s
    accel: float          # m/s^2
    steer: float          # normalized [-1, 1]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    speed_limit: float = 13.9   # m/s, ~50 km/h default


@dataclass
class Trajectory:
    points: List[Waypoint]
    target_speed: float
    horizon_s: float = 4.0

    @property
    def empty(self) -> bool:
        return not self.points


@dataclass
class DetectedObject:
    """A fused perception object."""
    obj_id: int
    cls: str                    # vehicle | pedestrian | cyclist | sign | misc
    position: Vec3
    velocity: Vec3
    bbox_extent: Vec3
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LaneInfo:
    left_offset: float          # m, + = left lane edge to the left of ego
    right_offset: float
    center_offset: float        # m, signed lateral error from lane center
    heading_error: float        # rad
    curvature: float            # 1/m
    lane_width: float = 3.5
    detected: bool = True


@dataclass
class ControlCommand:
    """Actuation demand sent to the vehicle."""
    throttle: float = 0.0       # [0, 1]
    brake: float = 0.0          # [0, 1]
    steer: float = 0.0          # [-1, 1]
    hand_brake: bool = False
    reverse: bool = False

    def clamp(self) -> "ControlCommand":
        self.throttle = min(max(self.throttle, 0.0), 1.0)
        self.brake = min(max(self.brake, 0.0), 1.0)
        self.steer = min(max(self.steer, -1.0), 1.0)
        return self


@dataclass
class SafetyEvent:
    level: str                  # info | warning | critical
    source: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerceptionOutput:
    objects: List[DetectedObject]
    lane: LaneInfo
    light: LightState = LightState.UNKNOWN
    free_space_ahead: float = 100.0   # m of clear path ahead
    stop_line_m: float = math.inf     # m to the constraining TL stop line
    timestamp: float = field(default_factory=time.time)
