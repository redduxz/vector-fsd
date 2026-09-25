"""Diagnostics — the safety layer's eyes on the rest of the stack.

Three jobs:

* **Latency tracking** — per-stage ring buffers with hz/mean/p95/max stats,
  fed either explicitly (``record_latency``) or via the ``stage()`` context
  manager (which also counts body exceptions).
* **Sensor staleness** — ``observe()`` marks data arrival; ``is_stale()``
  tells whether a source has gone quiet beyond its budget.
* **Fault injection** — armed faults either raise ``FaultInjectionError``
  through ``maybe_raise()`` or corrupt a value through ``corrupt()``, so test
  rigs can prove the watchdog actually fires.

Thread-safe for the realistic case of a control thread + a heartbeat thread.
"""
from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterator, Optional


class FaultInjectionError(RuntimeError):
    """Raised through Diagnostics.maybe_raise() when a 'raise' fault is armed."""


@dataclass
class _Stage:
    samples: Deque[float] = field(default_factory=deque)  # seconds, windowed
    count: int = 0
    total: float = 0.0
    last: float = 0.0
    worst: float = 0.0
    last_end: float = 0.0
    errors: int = 0


class Diagnostics:
    """Collects timing/staleness/health data. Intended to be a shared singleton."""

    def __init__(
        self,
        window: int = 512,
        stage_budget_ms: float = 100.0,
        stale_budget_s: float = 1.0,
    ) -> None:
        self.window = int(window)
        self.stage_budget_ms = float(stage_budget_ms)
        self.stale_budget_s = float(stale_budget_s)
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._stages: Dict[str, _Stage] = {}
        self._sensors: Dict[str, float] = {}
        self._counters: Counter = Counter()
        self._faults: Dict[str, Any] = {}  # name -> "raise" | callable(value)->value

    # ----------------------------------------------------------- latency

    def record_latency(self, stage: str, seconds: float, at: Optional[float] = None) -> None:
        """Record one latency sample (seconds) for a named pipeline stage."""
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return
        with self._lock:
            st = self._stages.get(stage)
            if st is None:
                st = self._stages[stage] = _Stage(samples=deque(maxlen=self.window))
            st.samples.append(float(seconds))
            st.count += 1
            st.total += seconds
            st.last = seconds
            st.worst = max(st.worst, seconds)
            st.last_end = time.time() if at is None else at

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block; latency is recorded even if the body raises."""
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            with self._lock:
                st = self._stages.get(name)
                if st is not None:
                    st.errors += 1
                self._counters[f"stage_errors.{name}"] += 1
            raise
        finally:
            self.record_latency(name, time.perf_counter() - t0)

    # ----------------------------------------------------------- staleness

    def observe(self, sensor: str, timestamp: Optional[float] = None) -> None:
        """Mark that fresh data arrived from ``sensor`` (wall-clock ``timestamp``)."""
        with self._lock:
            self._sensors[sensor] = time.time() if timestamp is None else float(timestamp)

    def staleness(self, sensor: str, now: Optional[float] = None) -> float:
        """Seconds since the sensor was observed; inf if never seen."""
        with self._lock:
            ts = self._sensors.get(sensor)
        if ts is None:
            return float("inf")
        return (time.time() if now is None else now) - ts

    def is_stale(self, sensor: str, max_age_s: Optional[float] = None,
                 now: Optional[float] = None) -> bool:
        return self.staleness(sensor, now) > (max_age_s if max_age_s is not None
                                              else self.stale_budget_s)

    # ----------------------------------------------------------- counters

    def inc(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counters[key] += n

    def counter(self, key: str) -> int:
        return self._counters.get(key, 0)

    # ------------------------------------------------------ fault injection

    def arm_fault(self, name: str, effect: Any = "raise") -> None:
        """Arm a fault. ``effect`` is 'raise' or a callable corrupting a value."""
        if effect != "raise" and not callable(effect):
            raise ValueError("fault effect must be 'raise' or callable")
        with self._lock:
            self._faults[name] = effect

    def clear_fault(self, name: str) -> bool:
        with self._lock:
            return self._faults.pop(name, None) is not None

    def clear_faults(self) -> None:
        with self._lock:
            self._faults.clear()

    def fault_armed(self, name: str) -> bool:
        with self._lock:
            return name in self._faults

    def armed_faults(self) -> list:
        with self._lock:
            return sorted(self._faults)

    def maybe_raise(self, stage: str) -> None:
        """Raise FaultInjectionError if a 'raise' fault is armed for ``stage``."""
        with self._lock:
            effect = self._faults.get(stage)
        if effect == "raise":
            self.inc(f"fault_fired.{stage}")
            raise FaultInjectionError(f"injected fault at stage '{stage}'")

    def corrupt(self, stage: str, value: Any) -> Any:
        """If a callable fault is armed for ``stage``, return effect(value)."""
        with self._lock:
            effect = self._faults.get(stage)
        if callable(effect):
            self.inc(f"fault_fired.{stage}")
            return effect(value)
        return value

    # ----------------------------------------------------------- report

    def stage_stats(self, name: str) -> Optional[Dict[str, float]]:
        with self._lock:
            st = self._stages.get(name)
            if st is None:
                return None
            samples = sorted(st.samples)
            count, total, last, worst, errors = st.count, st.total, st.last, st.worst, st.errors
        p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))] if samples else 0.0
        uptime = max(time.time() - self._t0, 1e-9)
        return {
            "count": count,
            "hz": round(count / uptime, 2),
            "mean_ms": round(1000.0 * total / count, 2) if count else 0.0,
            "p95_ms": round(1000.0 * p95, 2),
            "max_ms": round(1000.0 * worst, 2),
            "last_ms": round(1000.0 * last, 2),
            "errors": errors,
        }

    def health_report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Full status dict: 'ok' | 'degraded' | 'fault'."""
        now = time.time() if now is None else now
        with self._lock:
            sensors = dict(self._sensors)
            counters = dict(self._counters)
            faults = sorted(self._faults)
            stage_names = list(self._stages)

        stages = {n: self.stage_stats(n) for n in stage_names}
        stale = {s: round(now - ts, 3) for s, ts in sensors.items()}
        stale_over = [s for s, age in stale.items() if age > self.stale_budget_s]
        badly_stale = [s for s, age in stale.items() if age > 3.0 * self.stale_budget_s]

        over_budget = [n for n, s in stages.items()
                       if s and s["p95_ms"] > self.stage_budget_ms]
        way_over = [n for n, s in stages.items()
                    if s and s["p95_ms"] > 2.0 * self.stage_budget_ms]
        error_stages = [n for n, s in stages.items() if s and s["errors"] > 0]

        if badly_stale or way_over:
            status = "fault"
        elif stale_over or over_budget or error_stages or faults:
            status = "degraded"
        else:
            status = "ok"

        return {
            "status": status,
            "uptime_s": round(now - self._t0, 3),
            "stages": stages,
            "sensors": stale,
            "stale_sensors": stale_over,
            "counters": counters,
            "faults_armed": faults,
        }

    def reset(self) -> None:
        """Clear all collected data (faults stay armed — disarm explicitly)."""
        with self._lock:
            self._stages.clear()
            self._sensors.clear()
            self._counters.clear()
            self._t0 = time.time()
