"""SafetyMonitor — the hard gate between the planner and the actuators.

Nothing reaches the vehicle unless it went through ``check()`` + ``enforce()``
(or ``engage_safe_stop()``). Design principles:

* **Fail-safe everywhere.** Any internal error, malformed input, stale sensor
  or faulting rule degrades toward SAFE_STOP, never toward silence.
* **Escalate instantly, de-escalate slowly.** Warnings hold for
  ``warn_clear_s`` of clean cycles; criticals latch by default and require an
  explicit ``reset()`` — the same reason real emergency stops latch.
* **Enforcement is separate from detection.** Rules flag events; the monitor
  arbitrates a DriveMode, then runs every rule's ``enforce()`` on a private
  copy of the command so the planner's object is never mutated.
"""
from __future__ import annotations

import copy
import math
import time
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Sequence

from fsd.core.config import Config, SafetyConfig
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
from fsd.safety.rules import (
    SEVERITY_ORDER,
    SafetyContext,
    SafetyRule,
    default_rules,
)

_SAFETY_ATTRS = (
    "min_ttc_s", "max_speed_mps", "max_accel_mps2", "max_brake_mps2",
    "watchdog_timeout_s", "min_free_space_m", "max_steer_rate",
)

_MODE_PRIORITY = {
    DriveMode.ENGAGED: 0,
    DriveMode.DEGRADED: 1,
    DriveMode.SAFE_STOP: 2,
    DriveMode.DISENGAGED: 3,  # sticky: only reset() leaves it
}


def _resolve_safety_cfg(cfg) -> SafetyConfig:
    """Accept Config, SafetyConfig, a duck-typed equivalent, or None."""
    if cfg is None:
        return SafetyConfig()
    if isinstance(cfg, SafetyConfig):
        return cfg
    if isinstance(cfg, Config):
        return cfg.safety
    if all(hasattr(cfg, a) for a in _SAFETY_ATTRS):
        return cfg  # duck-typed safety config (e.g. a test stub)
    raise TypeError(
        f"cannot interpret {type(cfg).__name__} as a safety config; "
        "expected fsd.core.config.Config or SafetyConfig")


class SafetyMonitor:
    """Evaluates rules each cycle and arbitrates the DriveMode.

    Typical use, once per control tick::

        mode = monitor.check(ego, perception, demand_cmd, pipeline_alive)
        if mode is DriveMode.SAFE_STOP:
            actuate(monitor.engage_safe_stop())
        else:
            actuate(monitor.enforce(demand_cmd))
    """

    def __init__(
        self,
        cfg=None,
        rules: Optional[Sequence[SafetyRule]] = None,
        *,
        latch_critical: bool = True,
        warn_clear_s: float = 0.5,
        critical_clear_s: float = 3.0,
        default_dt: float = 0.05,
        diagnostics=None,
        event_history: int = 4096,
    ) -> None:
        self.cfg: SafetyConfig = _resolve_safety_cfg(cfg)
        self.rules: List[SafetyRule] = list(rules) if rules else default_rules()
        self.latch_critical = bool(latch_critical)
        self.warn_clear_s = float(warn_clear_s)
        self.critical_clear_s = float(critical_clear_s)
        self.default_dt = float(default_dt)
        self.diagnostics = diagnostics

        self.log = get("safety.monitor")
        self._mode = DriveMode.ENGAGED
        self._mode_since = time.time()
        self._disengaged = False
        self._latched = False
        self.hazards_requested = False

        self._last_check: Optional[float] = None
        self._last_heartbeat = time.time()  # grace period == watchdog_timeout_s
        self._last_ctx: Optional[SafetyContext] = None
        self._applied: Optional[ControlCommand] = None
        self._last_warn_t = -math.inf
        self._last_critical_t = -math.inf
        self._safe_stop_announced = False

        self._events: Deque[SafetyEvent] = deque(maxlen=event_history)
        self._counts: Counter = Counter()
        self._rule_faults: Counter = Counter()

    # ------------------------------------------------------------------ API

    def check(
        self,
        ego: Optional[VehicleState],
        perception: Optional[PerceptionOutput],
        cmd: Optional[ControlCommand],
        pipeline_alive: bool = True,
    ) -> DriveMode:
        """Run one arbitration cycle. Never raises — internal faults fail safe."""
        try:
            return self._check_inner(ego, perception, cmd, pipeline_alive)
        except Exception:  # pragma: no cover - defensive net
            self.log.exception("internal monitor fault — fail-safe SAFE_STOP")
            self._mode = DriveMode.SAFE_STOP
            if self.latch_critical:
                self._latched = True
            self.hazards_requested = True
            return DriveMode.SAFE_STOP

    def enforce(self, cmd: Optional[ControlCommand]) -> ControlCommand:
        """Return a command that is legal to send right now.

        Applies input sanitation, per-rule shaping, and the current drive
        mode's override. Never mutates the caller's object.
        """
        c = copy.copy(cmd) if isinstance(cmd, ControlCommand) else ControlCommand()
        c = self._sanitize_cmd(c)
        if self._mode is DriveMode.SAFE_STOP:
            return self.engage_safe_stop()
        if self._mode is DriveMode.DISENGAGED:
            return c.clamp()  # human owns the car; still range-clamp the pass-through
        ctx = self._last_ctx
        if ctx is not None:
            for rule in self.rules:
                if not rule.enabled:
                    continue
                try:
                    c = rule.enforce(c, ctx)
                except Exception as exc:  # pragma: no cover - defensive net
                    self._rule_faults[rule.name] += 1
                    self._log_event(SafetyEvent(
                        "critical", f"rule.{rule.name}",
                        f"enforce() fault: {exc!r} — skipping this shaper"))
        return c.clamp()

    def engage_safe_stop(self) -> ControlCommand:
        """The minimal-risk command: full brake, no throttle, hold steer.

        Also raises ``hazards_requested`` — the vehicle bridge should map that
        to the hazard lamps, since ControlCommand carries no light channel.
        """
        self.hazards_requested = True
        steer = 0.0
        if self._applied is not None and math.isfinite(self._applied.steer):
            steer = self._applied.steer
        speed = self._last_ctx.ego.speed if self._last_ctx else math.inf
        stopped = not math.isfinite(speed) or speed < 0.3
        if not self._safe_stop_announced:
            self._safe_stop_announced = True
            self.log.warning("SAFE STOP engaged — full brake, hazard lamps requested")
            self._log_event(SafetyEvent(
                "info", "monitor", "safe stop engaged; hazard lamps requested"))
        return ControlCommand(
            throttle=0.0,
            brake=1.0,
            steer=steer,
            hand_brake=stopped,   # hold the car once it has actually stopped
            reverse=False,
        )

    def heartbeat(self) -> None:
        """Upstream pipeline calls this every healthy tick."""
        self._last_heartbeat = time.time()

    def disengage(self) -> None:
        """Hand control to the human. Sticky until reset()."""
        if not self._disengaged:
            self._disengaged = True
            prev, self._mode = self._mode, DriveMode.DISENGAGED
            self._mode_since = time.time()
            self.log.warning("drive mode %s -> DISENGAGED (operator)", prev.name)
            self._log_event(SafetyEvent("info", "monitor", "disengaged by operator"))

    def reset(self) -> None:
        """Clear latches and return to ENGAGED. Records a fresh heartbeat."""
        self._mode = DriveMode.ENGAGED
        self._mode_since = time.time()
        self._disengaged = False
        self._latched = False
        self.hazards_requested = False
        self._safe_stop_announced = False
        self._applied = None
        self._last_ctx = None
        self._last_check = None
        self._last_warn_t = -math.inf
        self._last_critical_t = -math.inf
        self._last_heartbeat = time.time()
        for rule in self.rules:
            try:
                rule.reset()
            except Exception:  # pragma: no cover - defensive net
                self.log.exception("rule %s failed to reset", rule.name)
        self.log.info("monitor reset -> ENGAGED")
        self._log_event(SafetyEvent("info", "monitor", "monitor reset to ENGAGED"))

    # ------------------------------------------------------------ rule admin

    def add_rule(self, rule: SafetyRule) -> None:
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                return True
        return False

    def set_rule_enabled(self, name: str, enabled: bool) -> bool:
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = enabled
                return True
        return False

    def get_rule(self, name: str) -> Optional[SafetyRule]:
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    # -------------------------------------------------------------- introspect

    @property
    def mode(self) -> DriveMode:
        return self._mode

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def last_command(self) -> Optional[ControlCommand]:
        """The command the monitor last produced (post-enforcement)."""
        return self._applied

    @property
    def event_log(self) -> List[SafetyEvent]:
        return list(self._events)

    @property
    def event_counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def recent_events(self, n: int = 20) -> List[SafetyEvent]:
        return list(self._events)[-n:]

    @property
    def healthy(self) -> bool:
        return self._mode is DriveMode.ENGAGED

    def status(self) -> Dict:
        now = time.time()
        return {
            "mode": self._mode.name,
            "mode_for_s": round(now - self._mode_since, 3),
            "latched": self._latched,
            "hazards": self.hazards_requested,
            "heartbeat_age_s": round(now - self._last_heartbeat, 3),
            "rules": [{"name": r.name, "enabled": r.enabled} for r in self.rules],
            "rule_faults": dict(self._rule_faults),
            "event_counts": dict(self._counts),
            "events_logged": len(self._events),
        }

    # -------------------------------------------------------------- internals

    def _check_inner(self, ego, perception, cmd, pipeline_alive) -> DriveMode:
        if self.diagnostics is not None:
            with self.diagnostics.stage("safety.check"):
                return self._evaluate_cycle(ego, perception, cmd, pipeline_alive)
        return self._evaluate_cycle(ego, perception, cmd, pipeline_alive)

    def _evaluate_cycle(self, ego, perception, cmd, pipeline_alive) -> DriveMode:
        now = time.time()
        if self._last_check is None:
            dt = self.default_dt
        else:
            dt = min(max(now - self._last_check, 1e-4), 5.0)
        self._last_check = now

        events: List[SafetyEvent] = []
        ego = self._preflight_ego(ego, now, events)
        if perception is None:
            events.append(SafetyEvent(
                "critical", "monitor.preflight",
                "perception unavailable — treating world as obstructed"))
            perception = self._blind_perception(now)
        if not isinstance(cmd, ControlCommand):
            if cmd is not None:
                events.append(SafetyEvent(
                    "warning", "monitor.preflight",
                    f"command of type {type(cmd).__name__} not a ControlCommand — zeroing demand"))
            cmd = ControlCommand()

        ctx = SafetyContext(
            ego=ego, perception=perception, cmd=cmd, applied=self._applied,
            pipeline_alive=bool(pipeline_alive), now=now, dt=dt,
            cfg=self.cfg, last_heartbeat=self._last_heartbeat,
        )

        for rule in self.rules:
            if not rule.enabled:
                continue
            try:
                ev = rule.evaluate(ctx)
            except Exception as exc:
                self._rule_faults[rule.name] += 1
                ev = SafetyEvent(
                    "critical", f"rule.{rule.name}",
                    f"evaluate() fault: {exc!r} — rule assumed compromised")
            if ev is not None:
                events.append(ev)

        for ev in events:
            self._log_event(ev)
            if SEVERITY_ORDER.get(ev.level, 0) >= SEVERITY_ORDER["critical"]:
                self._last_critical_t = now
            elif ev.level == "warning":
                self._last_warn_t = now

        self._update_mode(self._arbitrate(events), now)
        self._last_ctx = ctx
        self._applied = self.enforce(cmd)
        self.hazards_requested = self.hazards_requested or self._mode is DriveMode.SAFE_STOP
        return self._mode

    def _preflight_ego(self, ego, now: float, events: List[SafetyEvent]) -> VehicleState:
        """Guarantee a finite VehicleState; substitute a parked car if needed."""
        if ego is None:
            events.append(SafetyEvent(
                "critical", "monitor.preflight", "ego state missing — assuming stationary"))
            return VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp=now)
        bad = [f for f, v in (("x", ego.x), ("y", ego.y), ("yaw", ego.yaw),
                              ("speed", ego.speed), ("accel", ego.accel),
                              ("steer", ego.steer)) if not math.isfinite(v)]
        if bad:
            events.append(SafetyEvent(
                "critical", "monitor.preflight",
                f"ego telemetry non-finite fields: {', '.join(bad)} — substituting zeros"))
            for f in bad:
                setattr(ego, f, 0.0)
        elif ego.speed < -0.5:
            events.append(SafetyEvent(
                "warning", "monitor.preflight",
                f"negative speed {ego.speed:.1f}m/s — sensor glitch or reversing"))
        return ego

    def _sanitize_cmd(self, c: ControlCommand) -> ControlCommand:
        dirty = False
        for f in ("throttle", "brake", "steer"):
            if not math.isfinite(getattr(c, f)):
                setattr(c, f, 0.0)
                dirty = True
        if dirty:
            self._log_event(SafetyEvent(
                "warning", "monitor.sanitize",
                "non-finite command fields zeroed before enforcement"))
        c.hand_brake = bool(c.hand_brake)
        c.reverse = bool(c.reverse)
        return c

    @staticmethod
    def _blind_perception(now: float) -> PerceptionOutput:
        """What the monitor assumes when perception is absent: worst case."""
        return PerceptionOutput(
            objects=[],
            lane=LaneInfo(0.0, 0.0, 0.0, 0.0, 0.0, detected=False),
            light=LightState.UNKNOWN,
            free_space_ahead=0.0,   # blind == obstructed
            timestamp=now,
        )

    @staticmethod
    def _arbitrate(events: Sequence[SafetyEvent]) -> DriveMode:
        worst = max((SEVERITY_ORDER.get(e.level, 0) for e in events), default=0)
        if worst >= SEVERITY_ORDER["critical"]:
            return DriveMode.SAFE_STOP
        if worst >= SEVERITY_ORDER["warning"]:
            return DriveMode.DEGRADED
        return DriveMode.ENGAGED

    def _update_mode(self, raw: DriveMode, now: float) -> None:
        if self._disengaged:
            return  # operator owns the car; still recording events
        prev = self._mode
        if _MODE_PRIORITY[raw] > _MODE_PRIORITY[prev]:
            self._mode = raw
        elif _MODE_PRIORITY[raw] < _MODE_PRIORITY[prev]:
            if not self._latched:
                if prev is DriveMode.SAFE_STOP and now - self._last_critical_t > self.critical_clear_s:
                    self._mode = (DriveMode.DEGRADED
                                  if now - self._last_warn_t <= self.warn_clear_s
                                  else DriveMode.ENGAGED)
                elif prev is DriveMode.DEGRADED and now - self._last_warn_t > self.warn_clear_s:
                    self._mode = DriveMode.ENGAGED
        if self._mode is DriveMode.SAFE_STOP and self.latch_critical:
            self._latched = True
        if self._mode is not prev:
            self._mode_since = now
            self.log.info("drive mode %s -> %s", prev.name, self._mode.name)

    def _log_event(self, ev: SafetyEvent) -> None:
        self._events.append(ev)
        self._counts[ev.level] += 1
        msg = f"[{ev.source}] {ev.message}"
        if ev.level == "critical":
            self.log.error("%s", msg)   # critical *event*; keep logger levels calm
        elif ev.level == "warning":
            self.log.warning("%s", msg)
        else:
            self.log.info("%s", msg)
