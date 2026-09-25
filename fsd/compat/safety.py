"""CppSafetyMonitor — SafetyMonitor-compatible facade over ``fsd_cpp``.

Same public API as ``fsd.safety.monitor.SafetyMonitor``; per call it converts
``fsd.core.types`` inputs to the extension's mirrored types, invokes
``fsd_cpp.SafetyMonitor.check(...)``, and coerces the returned mode string
back to a ``DriveMode``.

When ``fsd_cpp`` is absent (or its construction fails) the adapter transparently
wraps the pure-Python ``SafetyMonitor`` — call sites never need to know which
backend is serving them. Every crossing is guarded: any fault in conversion or
inside the extension fails safe to ``DriveMode.SAFE_STOP``, matching the
monitor's own "degrade toward stop" contract.
"""
from __future__ import annotations

import copy
import math
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence

from fsd.core.logger import get
from fsd.core.types import (
    ControlCommand,
    DriveMode,
    LaneInfo,
    LightState,
    PerceptionOutput,
    SafetyEvent,
    VehicleState,
)

from fsd.compat.backend import cpp_module, has_cpp
from fsd.compat.convert import (
    cfg_to_dict,
    command_from_cpp,
    control_command_to_cpp,
    mode_from_cpp,
    perception_to_cpp,
    vehicle_state_to_cpp,
)

# The Python reference implementation — used as the fallback backend and kept
# importable even while the C++ tree is under construction.
from fsd.safety.monitor import SafetyMonitor as _PySafetyMonitor


class CppSafetyMonitor:
    """Drop-in replacement for ``SafetyMonitor`` backed by ``fsd_cpp``.

    Construction mirrors ``SafetyMonitor(cfg, rules=None, *, ...)``; ``rules``
    are a Python-layer concept so they are only honored by the fallback
    backend. The C++ monitor is configured from ``cfg_dict`` derived via
    :func:`fsd.compat.convert.cfg_to_dict`.
    """

    def __init__(
        self,
        cfg=None,
        rules: Optional[Sequence] = None,
        *,
        latch_critical: bool = True,
        warn_clear_s: float = 0.5,
        critical_clear_s: float = 3.0,
        default_dt: float = 0.05,
        diagnostics=None,
        event_history: int = 4096,
    ) -> None:
        self.log = get("compat.safety")
        self._impl = None          # backend implementation object
        self._is_cpp = False
        self._local_rules: List = list(rules) if rules else []

        self.latch_critical = bool(latch_critical)
        self.warn_clear_s = float(warn_clear_s)
        self.critical_clear_s = float(critical_clear_s)
        self.default_dt = float(default_dt)
        self.diagnostics = diagnostics
        self._orig_cfg = cfg
        self.cfg_dict = cfg_to_dict(cfg)
        # Surface the monitor knobs to the extension without demanding a schema.
        mon = self.cfg_dict.get("monitor")
        if not isinstance(mon, dict):
            mon = {}
        mon.update(latch_critical=self.latch_critical,
                   warn_clear_s=self.warn_clear_s,
                   critical_clear_s=self.critical_clear_s,
                   default_dt=self.default_dt)
        self.cfg_dict["monitor"] = mon

        # Local mirrors — keep the adapter truthful even if the extension
        # exposes none of SafetyMonitor's bookkeeping attributes.
        self._mode = DriveMode.ENGAGED
        self._mode_since = time.time()
        self._disengaged = False
        self._latched = False
        self._hazards = False
        self._applied: Optional[ControlCommand] = None
        self._last_ego: Optional[VehicleState] = None
        self._last_heartbeat = time.time()
        self._events: deque = deque(maxlen=event_history)
        self._counts: Counter = Counter()

        self._impl = self._build_cpp() if has_cpp() else None
        if self._impl is None:
            self._impl = self._build_python(rules)
        self._is_cpp = not isinstance(self._impl, _PySafetyMonitor)
        self.log.info("CppSafetyMonitor online — backend=%s",
                      "cpp" if self._is_cpp else "python")

    # ------------------------------------------------------------- backends

    def _build_cpp(self):
        """Instantiate ``fsd_cpp.SafetyMonitor`` over plausible ctor shapes."""
        cls = getattr(cpp_module(), "SafetyMonitor", None)
        if cls is None:
            self.log.warning("fsd_cpp has no SafetyMonitor — using python")
            return None
        safety_section = self.cfg_dict.get("safety")
        attempts = [(self.cfg_dict,)]
        if isinstance(safety_section, dict):
            attempts.append((safety_section,))
        attempts.append(())
        for args in attempts:
            try:
                return cls(*args)
            except Exception as exc:
                self.log.debug("fsd_cpp.SafetyMonitor%r rejected: %s",
                               args and ("cfg_dict",) or (), exc)
        self.log.warning("fsd_cpp.SafetyMonitor ctor failed — using python")
        return None

    def _build_python(self, rules):
        cfg = self._orig_cfg
        if isinstance(cfg, dict):
            cfg = self._cfg_from_dict()  # python monitor rejects raw dicts
        return _PySafetyMonitor(
            cfg,
            rules=rules,
            latch_critical=self.latch_critical,
            warn_clear_s=self.warn_clear_s,
            critical_clear_s=self.critical_clear_s,
            default_dt=self.default_dt,
            diagnostics=self.diagnostics,
            event_history=self._events.maxlen or 4096,
        )

    def _cfg_from_dict(self):
        """Rebuild a Config for the Python monitor from a raw cfg_dict."""
        from fsd.core.config import Config, SafetyConfig
        cfg = Config()
        safety = self.cfg_dict.get("safety")
        if isinstance(safety, dict):
            try:
                known = {k: v for k, v in safety.items()
                         if k in SafetyConfig.__dataclass_fields__}
                cfg.safety = SafetyConfig(**{**vars(cfg.safety), **known})
            except Exception:
                pass
        return cfg

    # ------------------------------------------------------------------ API

    def check(
        self,
        ego: Optional[VehicleState],
        perception: Optional[PerceptionOutput],
        cmd: Optional[ControlCommand],
        pipeline_alive: bool = True,
    ) -> DriveMode:
        """One arbitration cycle; never raises, faults degrade to SAFE_STOP."""
        if not self._is_cpp:
            mode = self._impl.check(ego, perception, cmd, pipeline_alive)
            self._mode = mode
            self._sync()
            return mode
        try:
            if self.diagnostics is not None:
                with self.diagnostics.stage("safety.check"):
                    raw = self._check_cpp(ego, perception, cmd, pipeline_alive)
            else:
                raw = self._check_cpp(ego, perception, cmd, pipeline_alive)
            mode = mode_from_cpp(raw)
            self._set_mode(mode)
            self._sync()
            return self._mode
        except Exception:
            self.log.exception("cpp safety.check fault — fail-safe SAFE_STOP")
            self._record("critical", "compat.safety",
                         "extension check() raised — forcing SAFE_STOP")
            self._set_mode(DriveMode.SAFE_STOP)
            self._hazards = True
            return DriveMode.SAFE_STOP

    def _check_cpp(self, ego, perception, cmd, pipeline_alive):
        """Preflight (mirroring the python monitor) then cross the boundary."""
        if ego is None:
            self._record("critical", "compat.preflight",
                         "ego state missing — assuming stationary")
            ego = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               timestamp=time.time())
        if perception is None:
            self._record("critical", "compat.preflight",
                         "perception unavailable — treating world as obstructed")
            perception = PerceptionOutput(
                objects=[],
                lane=LaneInfo(0.0, 0.0, 0.0, 0.0, 0.0, detected=False),
                light=LightState.UNKNOWN,
                free_space_ahead=0.0,
                timestamp=time.time())
        if not isinstance(cmd, ControlCommand):
            if cmd is not None:
                self._record("warning", "compat.preflight",
                             f"command of type {type(cmd).__name__} "
                             "not a ControlCommand — zeroing demand")
            cmd = ControlCommand()
        self._last_ego = ego
        return self._impl.check(
            vehicle_state_to_cpp(ego),
            perception_to_cpp(perception),
            control_command_to_cpp(cmd),
            bool(pipeline_alive))

    def enforce(self, cmd: Optional[ControlCommand]) -> ControlCommand:
        """Return a command legal to send now — backend enforce or local."""
        fn = getattr(self._impl, "enforce", None)
        if callable(fn):
            try:
                out = fn(control_command_to_cpp(cmd)) if self._is_cpp \
                    else fn(cmd)
                res = command_from_cpp(out)
                self._applied = res
                return res
            except Exception:
                self.log.exception("backend enforce() fault — local fallback")
                self._record("critical", "compat.safety",
                             "enforce() raised on backend — minimal enforcement")
        if self.mode is DriveMode.SAFE_STOP:
            return self.engage_safe_stop()
        c = copy.copy(cmd) if isinstance(cmd, ControlCommand) \
            else ControlCommand()
        self._applied = c
        return c.clamp()

    def engage_safe_stop(self) -> ControlCommand:
        """Minimal-risk command: full brake, hold steer, hazards on."""
        self._hazards = True
        fn = getattr(self._impl, "engage_safe_stop", None)
        if callable(fn):
            try:
                cmd = command_from_cpp(fn())
                # The contract is a minimal-risk command — enforce it even if
                # the backend returned something weaker.
                cmd.brake = max(cmd.brake, 1.0)
                cmd.throttle = 0.0
                cmd.reverse = False
                self._applied = cmd
                self._sync()
                return cmd
            except Exception:
                self.log.exception("engage_safe_stop fault on backend — "
                                   "synthesizing locally")
        # Local mirror of SafetyMonitor.engage_safe_stop semantics.
        steer = 0.0
        if self._applied is not None and math.isfinite(self._applied.steer):
            steer = self._applied.steer
        speed = getattr(self._last_ego, "speed", math.inf)
        stopped = not math.isfinite(speed) or speed < 0.3
        cmd = ControlCommand(throttle=0.0, brake=1.0, steer=steer,
                             hand_brake=stopped, reverse=False)
        self._applied = cmd
        return cmd

    def heartbeat(self) -> None:
        """Upstream pipeline calls this every healthy tick."""
        self._last_heartbeat = time.time()
        fn = getattr(self._impl, "heartbeat", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                self.log.exception("backend heartbeat() fault")

    def disengage(self) -> None:
        """Hand control to the human. Sticky until reset()."""
        self._disengaged = True
        self._set_mode(DriveMode.DISENGAGED)
        fn = getattr(self._impl, "disengage", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                self.log.exception("backend disengage() fault")
        self._record("info", "monitor", "disengaged by operator")

    def reset(self) -> None:
        """Clear latches and return to ENGAGED on both layers."""
        self._mode = DriveMode.ENGAGED
        self._mode_since = time.time()
        self._disengaged = False
        self._latched = False
        self._hazards = False
        self._applied = None
        self._last_ego = None
        self._last_heartbeat = time.time()
        fn = getattr(self._impl, "reset", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                self.log.exception("backend reset() fault")
        self._record("info", "monitor", "monitor reset to ENGAGED")

    # ------------------------------------------------------------ rule admin

    def add_rule(self, rule) -> None:
        fn = getattr(self._impl, "add_rule", None)
        if callable(fn):
            fn(rule)
            return
        self._local_rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        fn = getattr(self._impl, "remove_rule", None)
        if callable(fn):
            try:
                return bool(fn(name))
            except Exception:
                return False
        for i, r in enumerate(self._local_rules):
            if getattr(r, "name", None) == name:
                del self._local_rules[i]
                return True
        return False

    def set_rule_enabled(self, name: str, enabled: bool) -> bool:
        fn = getattr(self._impl, "set_rule_enabled", None)
        if callable(fn):
            try:
                return bool(fn(name, enabled))
            except Exception:
                return False
        for r in self._local_rules:
            if getattr(r, "name", None) == name:
                r.enabled = bool(enabled)
                return True
        return False

    def get_rule(self, name: str):
        fn = getattr(self._impl, "get_rule", None)
        if callable(fn):
            try:
                return fn(name)
            except Exception:
                return None
        for r in self._local_rules:
            if getattr(r, "name", None) == name:
                return r
        return None

    @property
    def rules(self) -> List:
        r = getattr(self._impl, "rules", None)
        return r if r is not None else self._local_rules

    # -------------------------------------------------------------- introspect

    @property
    def backend(self) -> str:
        return "cpp" if self._is_cpp else "python"

    @property
    def impl(self):
        """The wrapped backend object (extension or python monitor)."""
        return self._impl

    @property
    def mode(self) -> DriveMode:
        m = getattr(self._impl, "mode", None)
        return mode_from_cpp(m, self._mode) if m is not None else self._mode

    @property
    def latched(self) -> bool:
        v = getattr(self._impl, "latched", None)
        return self._latched if v is None else (bool(v) or self._latched)

    @property
    def hazards_requested(self) -> bool:
        v = getattr(self._impl, "hazards_requested", None)
        return self._hazards if v is None else (bool(v) or self._hazards)

    @property
    def last_command(self) -> Optional[ControlCommand]:
        v = getattr(self._impl, "last_command", None)
        if v is None:
            v = getattr(self._impl, "applied", None)
        return command_from_cpp(v) if v is not None else self._applied

    @property
    def event_log(self) -> List[SafetyEvent]:
        v = getattr(self._impl, "event_log", None)
        if v is None and hasattr(self._impl, "events"):
            try:
                v = self._impl.events()
            except Exception:
                v = None
        return list(v) if v is not None else list(self._events)

    @property
    def event_counts(self) -> Dict[str, int]:
        v = getattr(self._impl, "event_counts", None)
        if v is not None:
            try:
                return dict(v)
            except Exception:
                pass
        return dict(self._counts)

    def recent_events(self, n: int = 20) -> List[SafetyEvent]:
        return self.event_log[-n:]

    @property
    def healthy(self) -> bool:
        return self.mode is DriveMode.ENGAGED

    def status(self) -> Dict:
        fn = getattr(self._impl, "status", None)
        base: Dict = {}
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, dict):
                    base = v
            except Exception:
                pass
        base.update({
            "backend": self.backend,
            "mode": self.mode.name,
            "latched": self.latched,
            "hazards": self.hazards_requested,
            "heartbeat_age_s": round(time.time() - self._last_heartbeat, 3),
        })
        return base

    # -------------------------------------------------------------- internals

    def __getattr__(self, name):
        """Forward anything unimplemented to the wrapped backend object."""
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(name)
        try:
            return getattr(impl, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r} "
                f"(backend={self.backend})") from None

    def _set_mode(self, mode: DriveMode) -> None:
        if mode is self._mode:
            return
        prev, self._mode = self._mode, mode
        self._mode_since = time.time()
        self._record("info", "monitor",
                     f"drive mode {prev.name} -> {self._mode.name}")
        if self._mode is DriveMode.SAFE_STOP:
            self._hazards = True
            if self.latch_critical:
                self._latched = True

    def _sync(self) -> None:
        """Pull bookkeeping state off the backend after each call.

        ``hazards``/``latched`` are latch-true until ``reset()`` — merge the
        backend's report with the adapter's own tracking instead of letting a
        backend that doesn't track them clobber our flags.
        """
        impl = self._impl
        h = getattr(impl, "hazards_requested", None)
        if h is not None:
            self._hazards = self._hazards or bool(h)
        l = getattr(impl, "latched", None)
        if l is not None:
            self._latched = self._latched or bool(l)
        lc = getattr(impl, "last_command", getattr(impl, "applied", None))
        if lc is not None:
            self._applied = command_from_cpp(lc)

    def _record(self, level: str, source: str, message: str) -> None:
        self._events.append(SafetyEvent(level, source, message))
        self._counts[level] += 1


__all__ = ["CppSafetyMonitor"]
