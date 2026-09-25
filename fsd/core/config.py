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
    town: str = "Town10HD_Opt"
    traffic_count: int = 40
    pedestrian_count: int = 20
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
    lidar_channels: int = 64
    radar_hz: int = 20


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
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
            for section, klass in (("sim", SimConfig), ("safety", SafetyConfig), ("vehicle", VehicleConfig)):
                if section in data:
                    setattr(cfg, section, klass(**{**asdict(getattr(cfg, section)), **data[section]}))
        return cfg

    def dump(self) -> Dict[str, Any]:
        return {"sim": asdict(self.sim), "safety": asdict(self.safety), "vehicle": asdict(self.vehicle)}
