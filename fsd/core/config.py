"""Config loader — YAML with safe defaults everywhere."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class SimConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 20.0
    fixed_delta_s: float = 0.05        # 20 Hz world tick
    town: str = "Town05"               # Town10HD_Opt stalls mid-range GPUs
    traffic_count: int = 20
    pedestrian_count: int = 10
    seed: int = 7


@dataclass
class SafetyConfig:
    min_ttc_s: float = 1.5             # time-to-collision floor
    max_speed_mps: float = 16.7        # 60 km/h cap
    max_accel_mps2: float = 3.0
    max_brake_mps2: float = 6.0
    watchdog_timeout_s: float = 0.5    # pipeline stall -> safe stop
    min_free_space_m: float = 8.0
    max_steer_rate: float = 0.4        # normalized units/s


@dataclass
class VehicleConfig:
    blueprint: str = "vehicle.tesla.model3"
    wheelbase_m: float = 2.875
    max_steer_deg: float = 60.0
    camera_hz: int = 20
    lidar_channels: int = 32           # 64ch parses ~1.3M pts/s per tick
    radar_hz: int = 20


@dataclass
class PerceptionConfig:
    object_backend: str = "none"       # 'none' | 'yolo' | 'onnx'
    object_model_path: str = ""
    object_conf_threshold: float = 0.35
    object_max_range_m: float = 120.0
    tl_search_radius_m: float = 60.0


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if path and os.path.exists(path):
            try:
                import yaml
                data = yaml.safe_load(open(path)) or {}
            except ImportError:
                data = {}
            cfg.raw = data
            for section, klass in (("sim", SimConfig), ("safety", SafetyConfig),
                                   ("vehicle", VehicleConfig), ("perception", PerceptionConfig)):
                if section in data:
                    setattr(cfg, section, klass(**{**asdict(getattr(cfg, section)), **data[section]}))
        return cfg

    def dump(self) -> Dict[str, Any]:
        return {"sim": asdict(self.sim), "safety": asdict(self.safety),
                "vehicle": asdict(self.vehicle),
                "perception": asdict(self.perception)}
