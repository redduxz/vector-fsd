"""ScenarioRunner — executes scripted scenarios against the autopilot loop.

Flow per scenario::

    agent = AutopilotAgent(cfg, smoke=True); agent.setup()
    scene = ScenarioScene(agent).install()
    scenario.setup(scene)
    while t < timeout and not scenario.done(t):
        ev = scenario.tick(t)        # perturb world / inject faults
        res = agent.tick()           # full stack + safety veto runs
        metrics.observe(res, t)
    verdict = scenario.evaluate(metrics)

The runner collects per-tick logs (``ClosedLoopMetrics``), scenario events,
safety-rule firings from ``SafetyMonitor.event_log`` and emits a verdict of
PASS / FAIL / TIMEOUT. CARLA is never required — smoke mode only.

CLI::

    python -m fsd.scenarios.runner --all
    python -m fsd.scenarios.runner -s lead_vehicle_cut_in -s jaywalker_pedestrian
    python -m fsd.scenarios.runner --list
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Allow `python fsd/scenarios/runner.py` in addition to `-m` usage.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

from fsd.core.config import Config
from fsd.core.logger import get
from fsd.core.types import ControlCommand, DriveMode
from fsd.agents.autopilot import AutopilotAgent
from fsd.eval.metrics import ClosedLoopMetrics, RunMetrics
from fsd.eval import report as eval_report
from fsd.scenarios.base import Scenario, ScenarioEvent, ScenarioScene, Verdict
from fsd.scenarios.library import SCENARIOS, build

log = get("scenarios.runner")


@dataclass
class ScenarioResult:
    """One scenario run: verdict + aggregated metrics + event log."""
    name: str
    verdict: Verdict
    metrics: RunMetrics
    events: List[ScenarioEvent] = field(default_factory=list)
    description: str = ""
    note: str = ""
    wall_s: float = 0.0


class ScenarioRunner:
    """Runs scenarios in smoke mode and produces ScenarioResults."""

    def __init__(self, cfg: Optional[Config] = None,
                 timeout_s: Optional[float] = None,
                 verbose: bool = False) -> None:
        self.cfg = cfg or Config.load("configs/default.yaml")
        self.timeout_s = timeout_s
        self.verbose = verbose

    # ------------------------------------------------------------------ core

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """Execute one scenario and return its verdict + metrics."""
        agent = AutopilotAgent(self.cfg, smoke=True)
        t0 = time.monotonic()
        events: List[ScenarioEvent] = []
        metrics = ClosedLoopMetrics(dt=agent.dt)
        note = ""
        verdict = Verdict.TIMEOUT
        scene: Optional[ScenarioScene] = None
        try:
            agent.setup()
            scene = ScenarioScene(agent).install()
            scenario.setup(scene)
            timeout = self.timeout_s or scenario.timeout_s
            t = 0.0
            done = False
            while t < timeout:
                ev = scenario.tick(t)
                if ev is not None:
                    events.append(ev)
                    log.info("%s | %s", scenario.name, ev)
                try:
                    res = agent.tick()
                except Exception as exc:  # a tick that escapes the loop
                    log.exception("%s: tick raised — emergency record", 
                                  scenario.name)
                    res = dict(tick=agent._tick_idx, ego=agent._ego_state(),
                               cmd=None, mode=DriveMode.SAFE_STOP,
                               perception=getattr(agent, "_last_perception",
                                                  None),
                               perception_ok=False, planning_ok=False,
                               crashed=True, events={})
                    try:
                        agent._apply(ControlCommand(brake=1.0))
                    except Exception:
                        pass
                metrics.observe(res, t)
                if scenario.done(t):
                    done = True
                    break
                t += agent.dt

            # harvest safety-rule firings recorded by the monitor
            safety = getattr(agent, "safety", None)
            elog = getattr(safety, "event_log", None)
            if elog:
                metrics.ingest_safety_events(elog)
            fails = getattr(agent, "_fail_counts", {})
            if fails:
                events.append(ScenarioEvent(
                    t, "note",
                    "stage failures: " + ", ".join(
                        f"{k}x{v}" for k, v in fails.items())))
            m = metrics.finalize()
            outcome = scenario.evaluate(m)
            if outcome is True:
                verdict = Verdict.PASS
            elif outcome is False:
                verdict = Verdict.FAIL
            elif done:
                verdict = Verdict.FAIL      # done but oracle undecided
            else:
                verdict = Verdict.TIMEOUT
            if not done and verdict is not Verdict.TIMEOUT:
                # loop ended by timeout yet oracle decided — keep verdict,
                # annotate so the report shows it
                note = f"oracle decided at timeout boundary ({timeout:.0f}s)"
        except Exception as exc:
            log.exception("scenario %s aborted", scenario.name)
            verdict = Verdict.FAIL
            note = f"runner exception: {exc!r}"
            m = metrics.finalize()
        finally:
            if scene is not None:
                try:
                    scene.teardown()
                except Exception:
                    pass
            try:
                agent.cleanup()
            except Exception:
                pass
        return ScenarioResult(
            name=scenario.name, verdict=verdict, metrics=m,
            events=events, description=scenario.description, note=note,
            wall_s=time.monotonic() - t0)

    def run_all(self, names: Optional[List[str]] = None
                ) -> List[ScenarioResult]:
        """Run the registry (or a subset) and return results in order."""
        order = names if names else list(SCENARIOS)
        results: List[ScenarioResult] = []
        for name in order:
            scenario = build(name)
            log.info("=== scenario %s starting (timeout %.0fs) ===",
                     name, self.timeout_s or scenario.timeout_s)
            res = self.run_scenario(scenario)
            log.info("=== %s -> %s (%.1fs wall, %d ticks) ===",
                     name, res.verdict.name, res.wall_s, res.metrics.ticks)
            results.append(res)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="fsd.scenarios.runner",
        description="Closed-loop scenario tests against the smoke-mode "
                    "autopilot (no CARLA required).")
    p.add_argument("--all", action="store_true",
                   help="run every registered scenario")
    p.add_argument("-s", "--scenario", action="append", default=[],
                   help="scenario name (repeatable)")
    p.add_argument("--list", action="store_true",
                   help="list registered scenarios and exit")
    p.add_argument("--timeout", type=float, default=None,
                   help="override per-scenario timeout (sim seconds)")
    p.add_argument("--config", default="configs/default.yaml",
                   help="YAML config path (missing -> defaults)")
    p.add_argument("--report", dest="report", action="store_true",
                   default=True, help="write markdown report (default)")
    p.add_argument("--no-report", dest="report", action="store_false")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="INFO-level pipeline logging (default: WARNING+ "
                        "suppressed except errors)")
    args = p.parse_args(argv)

    if not args.verbose:
        # Fault-injection scenarios raise inside stages on purpose — the
        # agent logs each as an ERROR traceback. Quiet mode drops to CRITICAL;
        # failures still surface via metrics/fail_counts.
        logging.getLogger("fsd").setLevel(logging.CRITICAL)

    if args.list:
        for name, cls in SCENARIOS.items():
            print(f"{name:24s} {cls.description}")
        return 0

    names = list(args.scenario)
    if args.all:
        names = list(SCENARIOS)
    if not names:
        p.error("nothing to run — use --all or -s NAME")

    cfg = Config.load(args.config)
    runner = ScenarioRunner(cfg, timeout_s=args.timeout,
                            verbose=args.verbose)
    results = runner.run_all(names)

    print()
    print(eval_report.console_table(results))
    if args.report:
        path = eval_report.write_markdown(results)
        print(f"\nmarkdown report: {path}")
    return 0 if all(r.verdict is Verdict.PASS for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
