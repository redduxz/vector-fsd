"""Evaluation report — console table + markdown artifact under docs/eval/.

Input is a sequence of duck-typed run results (``name``, ``verdict``,
``metrics`` (:class:`~fsd.eval.metrics.RunMetrics`), ``events``, optional
``wall_s`` / ``note``) as produced by ``fsd.scenarios.runner.ScenarioResult``.

Usage::

    from fsd.eval import report
    print(report.console_table(results))
    path = report.write_markdown(results)          # docs/eval/<ts>.md
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import List, Optional, Sequence

from fsd.core.logger import get

log = get("eval.report")

#: fsd/eval/report.py -> repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "eval"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _f(x, fmt="{:.2f}", na="-"):
    if x is None:
        return na
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return na


def _rule_hits_str(metrics) -> str:
    """Compact 'ttc:c3,w1; lane_departure:w12' summary of rule hits."""
    parts = []
    for name in sorted(metrics.rule_hits):
        hits = metrics.rule_hits[name]
        frag = ",".join(f"{lvl[0]}{n}" for lvl, n in sorted(hits.items()))
        parts.append(f"{name}:{frag}")
    return "; ".join(parts) or "-"


def _verdict_cell(v) -> str:
    name = getattr(v, "name", str(v))
    return {"PASS": "PASS", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT"}.get(
        name, name)


_COLUMNS = (
    ("Scenario", lambda r: r.name, 22),
    ("Verdict", lambda r: _verdict_cell(r.verdict), 11),
    ("Ticks", lambda r: str(r.metrics.ticks), 6),
    ("Dist m", lambda r: _f(r.metrics.distance_m, "{:.1f}"), 8),
    ("v_max", lambda r: _f(r.metrics.max_speed_mps, "{:.1f}"), 6),
    ("min TTC", lambda r: _f(r.metrics.min_ttc_s), 8),
    ("min gap", lambda r: _f(r.metrics.min_gap_m, "{:.1f}"), 8),
    ("jerk RMS", lambda r: _f(r.metrics.rms_jerk_mps3, "{:.2f}"), 9),
    ("lane RMSE", lambda r: _f(r.metrics.lane_rmse_m, "{:.2f}"), 10),
    ("safe stops", lambda r: str(r.metrics.safe_stops), 11),
    ("rule hits", lambda r: _rule_hits_str(r.metrics), 0),
)


def console_table(results: Sequence) -> str:
    """Render a fixed-width results table for the terminal."""
    if not results:
        return "(no results)"
    headers = [c[0] for c in _COLUMNS]
    rows: List[List[str]] = []
    for r in results:
        rows.append([fn(r) for _h, fn, _w in _COLUMNS])
    widths = []
    for i, (h, _fn, minw) in enumerate(_COLUMNS):
        w = max([len(h), minw] + [len(row[i]) for row in rows])
        widths.append(w)
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line,
           "|" + "|".join(f" {h:<{widths[i]}} " for i, h in enumerate(headers))
           + "|",
           line.replace("-", "=")]
    for row in rows:
        out.append("|" + "|".join(
            f" {c:<{widths[i]}} " for i, c in enumerate(row)) + "|")
    out.append(line)
    verdicts = [getattr(r.verdict, "name", str(r.verdict)) for r in results]
    out.append(
        f"totals: {verdicts.count('PASS')} PASS / "
        f"{verdicts.count('FAIL')} FAIL / {verdicts.count('TIMEOUT')} TIMEOUT")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------

def _md_metrics(metrics) -> str:
    dt = metrics.duration_s / max(1, metrics.ticks)
    mode_s = ", ".join(f"{k} {v * dt:.1f}s"
                       for k, v in sorted(metrics.mode_ticks.items())) or "-"
    return (
        f"| ticks | {metrics.ticks} |\n"
        f"| sim time | {metrics.duration_s:.1f} s |\n"
        f"| distance | {metrics.distance_m:.1f} m |\n"
        f"| speed avg/max | {metrics.avg_speed_mps:.1f} / "
        f"{metrics.max_speed_mps:.1f} m/s |\n"
        f"| min TTC | {_f(metrics.min_ttc_s, '{:.2f} s')} |\n"
        f"| min bumper gap | {_f(metrics.min_gap_m, '{:.2f} m')} |\n"
        f"| collision | {'yes' if metrics.collided else 'no'} |\n"
        f"| jerk rms/max | {metrics.rms_jerk_mps3:.2f} / "
        f"{metrics.max_jerk_mps3:.2f} m/s^3 |\n"
        f"| max |accel| | {metrics.max_abs_accel_mps2:.2f} m/s^2 |\n"
        f"| lane-center RMSE | {_f(metrics.lane_rmse_m, '{:.3f} m')} "
        f"({metrics.lane_samples} samples) |\n"
        f"| safe stops / degraded / disengaged | {metrics.safe_stops} / "
        f"{metrics.degraded_entries} / {metrics.disengagements} |\n"
        f"| crashed ticks | {metrics.crashed_ticks} |\n"
        f"| perception/planning failures | {metrics.perception_failures} / "
        f"{metrics.planning_failures} |\n"
        f"| time in mode | {mode_s} |\n"
        f"| rule hits | {_rule_hits_str(metrics)} |")


def write_markdown(results: Sequence,
                   out_dir: Optional[os.PathLike] = None,
                   title: str = "Closed-Loop Scenario Evaluation") -> Path:
    """Write the markdown report under ``docs/eval/`` and return its path.

    A timestamped file is written for history, plus ``latest.md`` is
    refreshed for quick diffing / linking.
    """
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")

    lines: List[str] = [
        f"# {title}",
        "",
        f"_Generated {ts.strftime('%Y-%m-%d %H:%M:%S')} — "
        "fsd.scenarios.runner (smoke world, no CARLA)_",
        "",
        "## Summary",
        "",
        "| Scenario | Verdict | Ticks | Dist (m) | min TTC (s) | "
        "min gap (m) | jerk RMS | lane RMSE | safe stops |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.metrics
        v = getattr(r.verdict, "name", str(r.verdict))
        lines.append(
            f"| {r.name} | {v} | {m.ticks} | {m.distance_m:.1f} | "
            f"{_f(m.min_ttc_s)} | {_f(m.min_gap_m, '{:.1f}')} | "
            f"{m.rms_jerk_mps3:.2f} | {_f(m.lane_rmse_m, '{:.3f}')} | "
            f"{m.safe_stops} |")
    verdicts = [getattr(r.verdict, "name", str(r.verdict)) for r in results]
    lines += [
        "",
        f"**Totals:** {verdicts.count('PASS')} PASS / "
        f"{verdicts.count('FAIL')} FAIL / {verdicts.count('TIMEOUT')} TIMEOUT",
        "",
        "## Details",
        "",
    ]
    for r in results:
        v = getattr(r.verdict, "name", str(r.verdict))
        lines += [f"### {r.name} — {v}", ""]
        if getattr(r, "description", ""):
            lines += [f"> {r.description}", ""]
        if getattr(r, "note", ""):
            lines += [f"**note:** {r.note}", ""]
        lines += ["| metric | value |", "|---|---|",
                  _md_metrics(r.metrics), ""]
        if r.events:
            lines += ["**Scenario events**", ""]
            lines += [f"- {e}" for e in r.events]
            lines += [""]
        if r.metrics.notable_events:
            lines += ["**Safety events (deduped)**", ""]
            for ev in r.metrics.notable_events[:15]:
                lines.append(f"- `{ev}`")
            if len(r.metrics.notable_events) > 15:
                lines.append(f"- … +{len(r.metrics.notable_events) - 15} more")
            lines += [""]
    text = "\n".join(lines) + "\n"

    stamped = out / f"scenario_eval_{stamp}.md"
    latest = out / "latest.md"
    stamped.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    log.info("wrote eval report: %s (+ latest.md)", stamped)
    return stamped


__all__ = ["console_table", "write_markdown", "DEFAULT_OUT_DIR"]
