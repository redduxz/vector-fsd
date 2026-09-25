"""Attribute-by-attribute conversion between ``fsd.core.types`` and ``fsd_cpp``.

The C++ extension mirrors the Python dataclasses 1:1 (same attribute names),
but we cannot know exactly how the pybind11 classes were bound — kwargs ctor,
positional ctor, or default-construct + ``def_readwrite`` members. So every
conversion is *generic*: read the documented attributes off the Python object,
then instantiate the C++ class trying each binding style in turn.

If a mirrored class is absent from the extension we degrade to a plain dict —
pybind11 signatures taking ``py::object``/``py::dict`` can still consume it,
and callers wrap the whole crossing in a fail-safe try/except anyway.

Return direction (C++ -> Python) never trusts the incoming object: anything
with the right attributes (or a dict, or our own dataclass) is re-built as a
real ``fsd.core.types`` instance so downstream code always sees Python types.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import fields as dc_fields, is_dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from fsd.core.types import (
    ControlCommand,
    DetectedObject,
    DriveMode,
    LaneInfo,
    LightState,
    PerceptionOutput,
    Trajectory,
    Vec3,
    VehicleState,
    Waypoint,
)

from fsd.compat.backend import cpp_module

# ---------------------------------------------------------------------------
# Field specs — (attribute names, in declared ctor order) per mirrored type.
# Order doubles as the positional-args fallback sequence.
# ---------------------------------------------------------------------------

VEC3_FIELDS: Tuple[str, ...] = ("x", "y", "z")
VEHICLE_STATE_FIELDS: Tuple[str, ...] = (
    "x", "y", "z", "yaw", "speed", "accel", "steer", "timestamp")
WAYPOINT_FIELDS: Tuple[str, ...] = ("x", "y", "z", "yaw", "speed_limit")
TRAJECTORY_FIELDS: Tuple[str, ...] = ("points", "target_speed", "horizon_s")
DETECTED_OBJECT_FIELDS: Tuple[str, ...] = (
    "obj_id", "cls", "position", "velocity", "bbox_extent", "confidence",
    "timestamp")
LANE_INFO_FIELDS: Tuple[str, ...] = (
    "left_offset", "right_offset", "center_offset", "heading_error",
    "curvature", "lane_width", "detected")
CONTROL_COMMAND_FIELDS: Tuple[str, ...] = (
    "throttle", "brake", "steer", "hand_brake", "reverse")
PERCEPTION_OUTPUT_FIELDS: Tuple[str, ...] = (
    "objects", "lane", "light", "free_space_ahead", "timestamp")


def _cpp():
    return cpp_module()


# ---------------------------------------------------------------------------
# Config -> dict (the C++ constructors take a plain cfg_dict)
# ---------------------------------------------------------------------------

_SAFETY_ATTRS: Tuple[str, ...] = (
    "min_ttc_s", "max_speed_mps", "max_accel_mps2", "max_brake_mps2",
    "watchdog_timeout_s", "min_free_space_m", "max_steer_rate",
)

_SECTIONS: Tuple[str, ...] = ("sim", "safety", "vehicle")


def _section_dict(obj: Any) -> Dict[str, Any]:
    """Dataclass / dict / plain object -> {attr: value}; {} if unreadable."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        if is_dataclass(obj):
            return {f.name: getattr(obj, f.name) for f in dc_fields(obj)}
    except Exception:
        pass
    try:
        return dict(vars(obj))
    except TypeError:
        return {}


def cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """Normalize any accepted config object into a plain ``cfg_dict``.

    Accepts ``Config`` (uses .dump()), ``SafetyConfig``, a duck-typed object
    exposing the safety attrs, a raw dict (passed through), or None (defaults).
    The result always carries the standard sections so the C++ side can read
    either ``cfg["safety"]["min_ttc_s"]`` or a top-level key.
    """
    from fsd.core.config import Config, SafetyConfig  # local: keep import light

    if cfg is None:
        return Config().dump()
    if isinstance(cfg, Config):
        return cfg.dump()
    if isinstance(cfg, SafetyConfig):
        return {"safety": _section_dict(cfg)}
    if isinstance(cfg, dict):
        return dict(cfg)
    # Duck-typed safety config (test stubs etc.)
    if all(hasattr(cfg, a) for a in _SAFETY_ATTRS):
        return {"safety": {a: getattr(cfg, a) for a in _SAFETY_ATTRS}}
    # Object carrying named sections (Config-like but not a Config)
    out: Dict[str, Any] = {}
    for section in _SECTIONS:
        sub = _section_dict(getattr(cfg, section, None))
        if sub:
            out[section] = sub
    if out:
        return out
    return Config().dump()


def cpp_class(name: str):
    """``fsd_cpp.<name>`` if the extension exposes it, else None."""
    mod = _cpp()
    return getattr(mod, name, None) if mod is not None else None


def _read_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``obj.name`` — attribute first, then dict key, then default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _read_attrs(obj: Any, names: Iterable[str]) -> Dict[str, Any]:
    return {n: _read_attr(obj, n) for n in names}


def _instantiate(cls_name: str, attrs: Dict[str, Any],
                 order: Iterable[str]) -> Any:
    """Build ``fsd_cpp.<cls_name>`` from ``attrs`` trying every binding style.

    Attempts, in order: keyword ctor, positional ctor (in ``order``),
    default ctor + attribute assignment. Falls back to a plain dict if the
    class is missing or every style fails — the caller decides whether that
    is acceptable.
    """
    cls = cpp_class(cls_name)
    if cls is None:
        return dict(attrs)
    try:
        return cls(**attrs)
    except Exception:
        pass
    seq = [attrs[k] for k in order if k in attrs]
    try:
        return cls(*seq)
    except Exception:
        pass
    try:
        inst = cls()
        for k in order:
            if k in attrs:
                setattr(inst, k, attrs[k])
        return inst
    except Exception:
        return dict(attrs)


# ---------------------------------------------------------------------------
# Python -> C++
# ---------------------------------------------------------------------------

def vec3_to_cpp(v: Any) -> Any:
    attrs = _read_attrs(v, VEC3_FIELDS)
    return _instantiate("Vec3", attrs, VEC3_FIELDS)


def vehicle_state_to_cpp(ego: Any) -> Any:
    attrs = _read_attrs(ego, VEHICLE_STATE_FIELDS)
    return _instantiate("VehicleState", attrs, VEHICLE_STATE_FIELDS)


def waypoint_to_cpp(wp: Any) -> Any:
    attrs = _read_attrs(wp, WAYPOINT_FIELDS)
    return _instantiate("Waypoint", attrs, WAYPOINT_FIELDS)


def detected_object_to_cpp(obj: Any) -> Any:
    attrs = _read_attrs(obj, DETECTED_OBJECT_FIELDS)
    attrs["position"] = vec3_to_cpp(attrs["position"])
    attrs["velocity"] = vec3_to_cpp(attrs["velocity"])
    attrs["bbox_extent"] = vec3_to_cpp(attrs["bbox_extent"])
    return _instantiate("DetectedObject", attrs, DETECTED_OBJECT_FIELDS)


def lane_info_to_cpp(lane: Any) -> Any:
    attrs = _read_attrs(lane, LANE_INFO_FIELDS)
    return _instantiate("LaneInfo", attrs, LANE_INFO_FIELDS)


def control_command_to_cpp(cmd: Any) -> Any:
    attrs = _read_attrs(cmd, CONTROL_COMMAND_FIELDS)
    attrs["hand_brake"] = bool(attrs.get("hand_brake") or False)
    attrs["reverse"] = bool(attrs.get("reverse") or False)
    return _instantiate("ControlCommand", attrs, CONTROL_COMMAND_FIELDS)


def enum_to_cpp(value: Any, cls_name: str) -> Any:
    """Convert a Python Enum to the mirrored ``fsd_cpp`` enum if it exists.

    Falls back to the member name (a string) — pybind11 signatures typed as
    ``std::string`` consume it directly, and numeric signatures can often
    take the enum's int value which we try second.
    """
    cls = cpp_class(cls_name)
    name = getattr(value, "name", None) or str(value)
    if cls is not None:
        member = getattr(cls, name, None)
        if member is not None:
            return member
        for cand in (name.upper(), name.lower()):
            member = getattr(cls, cand, None)
            if member is not None:
                return member
        intval = getattr(value, "value", None)
        if intval is not None:
            try:
                return cls(int(intval))
            except Exception:
                pass
    return name


def perception_to_cpp(perception: Any) -> Any:
    attrs = _read_attrs(perception, PERCEPTION_OUTPUT_FIELDS)
    objects = attrs.get("objects") or []
    attrs["objects"] = [detected_object_to_cpp(o) for o in objects]
    attrs["lane"] = lane_info_to_cpp(attrs.get("lane"))
    attrs["light"] = enum_to_cpp(attrs.get("light") or LightState.UNKNOWN,
                                 "LightState")
    return _instantiate("PerceptionOutput", attrs, PERCEPTION_OUTPUT_FIELDS)


def trajectory_to_cpp(traj: Any) -> Any:
    attrs = _read_attrs(traj, TRAJECTORY_FIELDS)
    points = attrs.get("points") or []
    attrs["points"] = [waypoint_to_cpp(p) for p in points]
    return _instantiate("Trajectory", attrs, TRAJECTORY_FIELDS)


# ---------------------------------------------------------------------------
# C++ -> Python
# ---------------------------------------------------------------------------

def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def command_from_cpp(obj: Any) -> ControlCommand:
    """Rebuild a ``fsd.core.types.ControlCommand`` from anything cmd-shaped."""
    if obj is None:
        return ControlCommand()
    if isinstance(obj, ControlCommand):
        return obj
    try:
        return ControlCommand(
            throttle=_num(_read_attr(obj, "throttle")),
            brake=_num(_read_attr(obj, "brake")),
            steer=_num(_read_attr(obj, "steer")),
            hand_brake=bool(_read_attr(obj, "hand_brake", False)),
            reverse=bool(_read_attr(obj, "reverse", False)),
        ).clamp()
    except Exception:
        # A command we cannot even read must not reach the actuators.
        return ControlCommand(throttle=0.0, brake=1.0)


# DriveMode decoding: the C++ side returns "a str mode" per contract, but be
# liberal — accept DriveMode, pybind enums (.name), ints (ordinal) and spellings
# like "SafeStop"/"safe-stop"/"ESTOP".
_MODE_ALIASES = {
    "ENGAGED": DriveMode.ENGAGED, "ACTIVE": DriveMode.ENGAGED,
    "NORMAL": DriveMode.ENGAGED, "OK": DriveMode.ENGAGED,
    "DEGRADED": DriveMode.DEGRADED, "LIMITED": DriveMode.DEGRADED,
    "WARN": DriveMode.DEGRADED, "WARNING": DriveMode.DEGRADED,
    "SAFESTOP": DriveMode.SAFE_STOP, "ESTOP": DriveMode.SAFE_STOP,
    "EMERGENCY": DriveMode.SAFE_STOP, "EMERGENCYSTOP": DriveMode.SAFE_STOP,
    "STOP": DriveMode.SAFE_STOP, "CRITICAL": DriveMode.SAFE_STOP,
    "DISENGAGED": DriveMode.DISENGAGED, "OFF": DriveMode.DISENGAGED,
    "MANUAL": DriveMode.DISENGAGED, "STANDBY": DriveMode.DISENGAGED,
}


def mode_from_cpp(value: Any, default: DriveMode = DriveMode.SAFE_STOP
                  ) -> DriveMode:
    """Coerce a backend check() result into ``fsd.core.types.DriveMode``.

    ``default`` is SAFE_STOP — an unreadable answer must bias toward stop,
    never toward silently resuming engagement.
    """
    if value is None:
        return default
    if isinstance(value, DriveMode):
        return value
    if isinstance(value, str):
        key = re.sub(r"[^A-Z0-9]", "", value.upper())
        for member in DriveMode:
            if key == member.name.replace("_", ""):
                return member
        return _MODE_ALIASES.get(key, default)
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return mode_from_cpp(name, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        members = list(DriveMode)
        if 0 <= value < len(members):
            return members[value]
    return default


def vec3_from_cpp(obj: Any) -> Vec3:
    return Vec3(x=_num(_read_attr(obj, "x")),
                y=_num(_read_attr(obj, "y")),
                z=_num(_read_attr(obj, "z")))


def vehicle_state_from_cpp(obj: Any) -> VehicleState:
    return VehicleState(
        x=_num(_read_attr(obj, "x")), y=_num(_read_attr(obj, "y")),
        z=_num(_read_attr(obj, "z")), yaw=_num(_read_attr(obj, "yaw")),
        speed=_num(_read_attr(obj, "speed")),
        accel=_num(_read_attr(obj, "accel")),
        steer=_num(_read_attr(obj, "steer")),
        timestamp=_num(_read_attr(obj, "timestamp"), time.time()))


def waypoint_from_cpp(obj: Any) -> Waypoint:
    return Waypoint(
        x=_num(_read_attr(obj, "x")), y=_num(_read_attr(obj, "y")),
        z=_num(_read_attr(obj, "z")), yaw=_num(_read_attr(obj, "yaw")),
        speed_limit=_num(_read_attr(obj, "speed_limit"), 13.9))


def lane_info_from_cpp(obj: Any) -> LaneInfo:
    if obj is None:
        return LaneInfo(0.0, 0.0, 0.0, 0.0, 0.0, detected=False)
    return LaneInfo(
        left_offset=_num(_read_attr(obj, "left_offset")),
        right_offset=_num(_read_attr(obj, "right_offset")),
        center_offset=_num(_read_attr(obj, "center_offset")),
        heading_error=_num(_read_attr(obj, "heading_error")),
        curvature=_num(_read_attr(obj, "curvature")),
        lane_width=_num(_read_attr(obj, "lane_width"), 3.5),
        detected=bool(_read_attr(obj, "detected", True)))


def detected_object_from_cpp(obj: Any) -> DetectedObject:
    return DetectedObject(
        obj_id=int(_num(_read_attr(obj, "obj_id"))),
        cls=str(_read_attr(obj, "cls", "misc")),
        position=vec3_from_cpp(_read_attr(obj, "position")),
        velocity=vec3_from_cpp(_read_attr(obj, "velocity")),
        bbox_extent=vec3_from_cpp(_read_attr(obj, "bbox_extent")),
        confidence=_num(_read_attr(obj, "confidence")),
        timestamp=_num(_read_attr(obj, "timestamp"), time.time()))


def light_from_cpp(value: Any) -> LightState:
    if isinstance(value, LightState):
        return value
    name = getattr(value, "name", None)
    text = name if isinstance(name, str) else str(value or "")
    key = re.sub(r"[^A-Z0-9]", "", text.upper())
    for member in LightState:
        if key == member.name.replace("_", ""):
            return member
    return LightState.UNKNOWN


def perception_from_cpp(obj: Any) -> PerceptionOutput:
    return PerceptionOutput(
        objects=[detected_object_from_cpp(o)
                 for o in (_read_attr(obj, "objects") or [])],
        lane=lane_info_from_cpp(_read_attr(obj, "lane")),
        light=light_from_cpp(_read_attr(obj, "light")),
        free_space_ahead=_num(_read_attr(obj, "free_space_ahead"), 0.0),
        timestamp=_num(_read_attr(obj, "timestamp"), time.time()))


__all__ = [
    "cfg_to_dict",
    "cpp_class",
    "vec3_to_cpp",
    "vehicle_state_to_cpp",
    "waypoint_to_cpp",
    "detected_object_to_cpp",
    "lane_info_to_cpp",
    "control_command_to_cpp",
    "perception_to_cpp",
    "trajectory_to_cpp",
    "enum_to_cpp",
    "command_from_cpp",
    "mode_from_cpp",
    "vec3_from_cpp",
    "vehicle_state_from_cpp",
    "waypoint_from_cpp",
    "lane_info_from_cpp",
    "detected_object_from_cpp",
    "light_from_cpp",
    "perception_from_cpp",
]
