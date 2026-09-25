"""Closed-loop metrics — per-run aggregation for scenario evaluation.

``ClosedLoopMetrics.observe(res, t)`` digests one ``agent.tick()`` result
dict into a :class:`TickRecord` and accumulates running aggregates;
``finalize()`` rolls the run up into an immutable :class:`RunMetrics` used by
scenarios (pass/fail) and by :mod:`fsd.eval.report` (tables / markdown).

Metrics computed per run
------------------------

* ``min_ttc_s``        smallest time-to-collision over the ego corridor
                       (same geometry as ``TTCRule``: corridor half-width
                       = ego half-width + object half-width; TTC = bumper
                       gap / closing speed).
* ``min_gap_m``        smallest bumper gap to any in-corridor object;
                       ``collided`` when it went negative.
* ``rule_hits``        safety-rule firings by rule name and severity,
                       harvested from ``SafetyMonitor.event_log``.
* ``safe_stops`` / ``disengagements`` / ``degraded_entries``
                     DriveMode transition counters.
* ``distance_m``       path length integrated from ego positions.
* ``rms_jerk_mps3`` / ``max_jerk_mps3``
                     comfort metric — d(accel)/dt through the run.
* ``lane_rmse_m``      RMSE of ``lane.center_offset`` over detected ticks.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fsd.core.types import DriveMode

EGO_HALF_LENGTH_M = 2.4
EGO_HALF_WIDTH_M = 1.0
MIN_CLOSING_MPS = 0.1          # mirrors TTCRule.MIN_CLOSING_MPS


@dataclass
class TickRecord:
    """One digested agent.tick() result."""
    t: float
    tick: int
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    accel: float = 0.0
    ego_steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    mode: str = "DISENGAGED"
    perception_ok: bool = True
    planning_ok: bool = True
    n_objects: int = 0
    free_space: float = math.inf
    light: str = "UNKNOWN"
    lane_detected: bool = False
    lane_offset: Optional[float] = None
    ttc: Optional[float] = None          # seconds; None when no corridor threat
    min_gap: Optional[float] = None      # bumper gap, metres
    crashed: bool = False


@dataclass
class RunMetrics:
    """Per-run aggregates consumed by the report and verdict logic."""
    ticks: int = 0
    duration_s: float = 0.0
    distance_m: float = 0.0
    avg_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    min_ttc_s: Optional[float] = None
    min_gap_m: Optional[float] = None
    collided: bool = False
    rms_jerk_mps3: float = 0.0
    max_jerk_mps3: float = 0.0
    max_abs_accel_mps2: float = 0.0
    lane_rmse_m: Optional[float] = None
    lane_samples: int = 0
    safe_stops: int = 0
    disengagements: int = 0
    degraded_entries: int = 0
    crashed_ticks: int = 0
    perception_failures: int = 0
    planning_failures: int = 0
    mode_ticks: Dict[str, int] = field(default_factory=dict)
    rule_hits: Dict[str, Dict[str, int]] = field(default_factory=dict)
    notable_events: List[str] = field(default_factory=list)

    def mode_seconds(self, dt: float) -> Dict[str, float]:
        return {k: v * dt for k, v in self.mode_ticks.items()}


class ClosedLoopMetrics:
    """Streaming aggregator — call :meth:`observe` once per agent tick."""

    def __init__(self, dt: float = 0.05,
                 corridor_base_half_width: float = EGO_HALF_WIDTH_M):
        self.dt = float(dt)
        self.corridor_base = float(corridor_base_half_width)
        self.records: List[TickRecord] = []

        self._prev_x = self._prev_y = None
        self._prev_accel = None
        self._prev_mode: Optional[str] = None

        self._distance = 0.0
        self._speed_sum = 0.0
        self._max_speed = 0.0
        self._max_abs_accel = 0.0
        self._min_ttc = math.inf
        self._min_gap = math.inf
        self._jerk_sq_sum = 0.0
        self._jerk_n = 0
        self._max_jerk = 0.0
        self._lane_sq_sum = 0.0
        self._lane_n = 0
        self._mode_ticks: Counter = Counter()
        self._safe_stops = 0
        self._disengagements = 0
        self._degraded = 0
        self._crashed = 0
        self._perc_fail = 0
        self._plan_fail = 0
        self._rule_hits: Dict[str, Counter] = {}
        self._events_seen: set = set()
        self._notable: List[str] = []

    # ------------------------------------------------------------------ feed

    @staticmethod
    def _mode_name(mode) -> str:
        if isinstance(mode, DriveMode):
            return mode.name
        return str(mode) if mode is not None else "DISENGAGED"

    def _corridor_scan(self, ego, objects) -> Tuple[float, float]:
        """(min_ttc, min_bumper_gap) over objects inside the ego corridor."""
        cy, sy = math.cos(ego.yaw), math.sin(ego.yaw)
        best_ttc = math.inf
        best_gap = math.inf
        for o in objects or []:
            try:
                dx = o.position.x - ego.x
                dy = o.position.y - ego.y
            except AttributeError:
                continue
            lon = dx * cy + dy * sy
            lat = -dx * sy + dy * cy
            if lon < -1.0:
                continue
            corridor = self.corridor_base + max(0.5, abs(o.bbox_extent.y))
            if abs(lat) > corridor:
                continue
            gap = lon - EGO_HALF_LENGTH_M - abs(o.bbox_extent.x)
            best_gap = min(best_gap, gap)
            obj_fwd = o.velocity.x * cy + o.velocity.y * sy
            closing = ego.speed - obj_fwd
            if gap <= 0.0:
                best_ttc = 0.0
            elif closing > MIN_CLOSING_MPS:
                best_ttc = min(best_ttc, gap / closing)
        return best_ttc, best_gap

    def observe(self, res: dict, t: float) -> TickRecord:
        """Digest one tick result dict into a record + running aggregates."""
        ego = res.get("ego")
        cmd = res.get("cmd")
        perc = res.get("perception")
        rec = TickRecord(t=t, tick=int(res.get("tick", len(self.records))))
        if ego is not None:
            rec.x, rec.y, rec.yaw = ego.x, ego.y, ego.yaw
            rec.speed, rec.accel, rec.ego_steer = (
                ego.speed, ego.accel, ego.steer)
        if cmd is not None:
            rec.throttle, rec.brake, rec.steer = (
                cmd.throttle, cmd.brake, cmd.steer)
        rec.mode = self._mode_name(res.get("mode"))
        rec.perception_ok = bool(res.get("perception_ok", True))
        rec.planning_ok = bool(res.get("planning_ok", True))
        rec.crashed = bool(res.get("crashed", False))
        if perc is not None:
            rec.n_objects = len(perc.objects or [])
            rec.free_space = perc.free_space_ahead
            rec.light = getattr(perc.light, "name", str(perc.light))
            lane = getattr(perc, "lane", None)
            if lane is not None:
                rec.lane_detected = bool(lane.detected)
                rec.lane_offset = (float(lane.center_offset)
                                   if lane.detected else None)
        if ego is not None and perc is not None:
            ttc, gap = self._corridor_scan(ego, perc.objects)
            rec.ttc = None if ttc is math.inf else ttc
            rec.min_gap = None if gap is math.inf else gap
            self._min_ttc = min(self._min_ttc, ttc)
            self._min_gap = min(self._min_gap, gap)

        # ---- aggregates -------------------------------------------------- #
        if self._prev_x is not None and ego is not None:
            self._distance += math.hypot(rec.x - self._prev_x,
                                         rec.y - self._prev_y)
        if self._prev_accel is not None and not rec.crashed:
            jerk = (rec.accel - self._prev_accel) / self.dt
            self._jerk_sq_sum += jerk * jerk
            self._jerk_n += 1
            self._max_jerk = max(self._max_jerk, abs(jerk))
        self._prev_x, self._prev_y = rec.x, rec.y
        self._prev_accel = rec.accel
        self._speed_sum += rec.speed
        self._max_speed = max(self._max_speed, rec.speed)
        self._max_abs_accel = max(self._max_abs_accel, abs(rec.accel))
        if rec.lane_offset is not None:
            self._lane_sq_sum += rec.lane_offset ** 2
            self._lane_n += 1
        self._mode_ticks[rec.mode] += 1
        if self._prev_mode is not None and rec.mode != self._prev_mode:
            if rec.mode == "SAFE_STOP":
                self._safe_stops += 1
            elif rec.mode == "DISENGAGED":
                self._disengagements += 1
            elif rec.mode == "DEGRADED":
                self._degraded += 1
        self._prev_mode = rec.mode
        if rec.crashed:
            self._crashed += 1
        if not rec.perception_ok:
            self._perc_fail += 1
        if not rec.planning_ok:
            self._plan_fail += 1
        self.records.append(rec)
        return rec

    def ingest_safety_events(self, events) -> None:
        """Harvest rule hits + notable messages from ``monitor.event_log``."""
        for ev in events or []:
            src = getattr(ev, "source", "?")
            level = getattr(ev, "level", "info")
            name = src[len("rule."):] if src.startswith("rule.") else src
            self._rule_hits.setdefault(name, Counter())[level] += 1
            key = (src, getattr(ev, "message", ""))
            if level in ("warning", "critical") and key not in self._events_seen:
                self._events_seen.add(key)
                self._notable.append(f"[{level}] {src}: {ev.message}")

    # ------------------------------------------------------------------ out

    def finalize(self) -> RunMetrics:
        n = len(self.records)
        m = RunMetrics()
        m.ticks = n
        m.duration_s = n * self.dt
        m.distance_m = self._distance
        m.avg_speed_mps = self._speed_sum / n if n else 0.0
        m.max_speed_mps = self._max_speed
        m.min_ttc_s = None if self._min_ttc is math.inf else self._min_ttc
        m.min_gap_m = None if self._min_gap is math.inf else self._min_gap
        m.collided = bool(m.min_gap_m is not None and m.min_gap_m < 0.0)
        m.rms_jerk_mps3 = (math.sqrt(self._jerk_sq_sum / self._jerk_n)
                           if self._jerk_n else 0.0)
        m.max_jerk_mps3 = self._max_jerk
        m.max_abs_accel_mps2 = self._max_abs_accel
        m.lane_rmse_m = (math.sqrt(self._lane_sq_sum / self._lane_n)
                         if self._lane_n else None)
        m.lane_samples = self._lane_n
        m.safe_stops = self._safe_stops
        m.disengagements = self._disengagements
        m.degraded_entries = self._degraded
        m.crashed_ticks = self._crashed
        m.perception_failures = self._perc_fail
        m.planning_failures = self._plan_fail
        m.mode_ticks = dict(self._mode_ticks)
        m.rule_hits = {k: dict(v) for k, v in self._rule_hits.items()}
        m.notable_events = list(self._notable)
        return m


__all__ = ["TickRecord", "RunMetrics", "ClosedLoopMetrics"]
