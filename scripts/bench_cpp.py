"""Benchmark the safety/control hot path across backends.

Runs N ticks of ``SafetyMonitor.check()`` and ``VehicleController.compute()``
on synthetic inputs — once through the pure-Python implementations and once
through the ``fsd.compat`` adapters (which route to ``fsd_cpp`` when the C++
extension is built). Prints per-call microseconds and the speedup.

Standalone usage::

    python scripts/bench_cpp.py                # auto-detect backend
    python scripts/bench_cpp.py --ticks 20000 --warmup 500

With no ``fsd_cpp`` built, the adapter runs on the Python fallback — the
report then measures the conversion/forwarding overhead of the adapter
itself (a useful floor: real C++ deployments only get faster from there).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fsd.compat import diagnostics as backend_diagnostics, has_cpp  # noqa: E402
from fsd.compat.control import CppVehicleController  # noqa: E402
from fsd.compat.safety import CppSafetyMonitor  # noqa: E402
from fsd.control.controller import VehicleController  # noqa: E402
from fsd.core.config import Config  # noqa: E402
from fsd.core.types import (  # noqa: E402
    ControlCommand,
    DetectedObject,
    LaneInfo,
    PerceptionOutput,
    Trajectory,
    Vec3,
    VehicleState,
    Waypoint,
)
from fsd.safety.monitor import SafetyMonitor  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic scene — nominal driving, exercises the real hot path (TTC scan,
# speed cap, steer-rate limiter, longitudinal PID + lateral Stanley) without
# tripping into latched SAFE_STOP where the work would be trivially short.
# ---------------------------------------------------------------------------

def make_inputs(cfg: Config):
    ego = VehicleState(x=0.0, y=0.0, z=0.0, yaw=0.0, speed=9.0,
                       accel=0.0, steer=0.0, timestamp=time.time())
    lead = DetectedObject(
        obj_id=1, cls="vehicle",
        position=Vec3(x=30.0, y=0.0, z=0.0),
        velocity=Vec3(x=8.5, y=0.0, z=0.0),
        bbox_extent=Vec3(x=2.2, y=0.9, z=0.75),
        confidence=0.95, timestamp=time.time())
    perception = PerceptionOutput(
        objects=[lead],
        lane=LaneInfo(left_offset=1.75, right_offset=-1.75,
                      center_offset=0.0, heading_error=0.0,
                      curvature=0.0),
        free_space_ahead=60.0,
        timestamp=time.time())
    cmd = ControlCommand(throttle=0.3, brake=0.0, steer=0.0)
    traj = Trajectory(
        points=[Waypoint(x=float(i) * 2.0, y=0.0, yaw=0.0)
                for i in range(1, 61)],
        target_speed=10.0)
    return ego, perception, cmd, traj, lead


def _time(fn: Callable[[int], None], ticks: int, warmup: int) -> float:
    """Mean seconds-per-call over ``ticks`` calls after ``warmup``."""
    for i in range(warmup):
        fn(i)
    t0 = time.perf_counter()
    for i in range(ticks):
        fn(i)
    return (time.perf_counter() - t0) / max(ticks, 1)


def bench_monitor(monitor, inputs, ticks: int, warmup: int) -> float:
    ego, perception, cmd, _traj, _lead = inputs
    heartbeat = getattr(monitor, "heartbeat", None)

    def tick(i: int) -> None:
        now = time.time()
        ego.timestamp = now
        perception.timestamp = now
        ego.speed = 9.0 + math.sin(i * 0.02)          # mild variation
        ego.x = i * 0.05
        if callable(heartbeat):
            heartbeat()
        monitor.check(ego, perception, cmd, True)

    return _time(tick, ticks, warmup)


def bench_controller(controller, inputs, ticks: int, warmup: int) -> float:
    ego, perception, _cmd, traj, _lead = inputs

    def tick(i: int) -> None:
        ego.timestamp = time.time()
        ego.speed = 8.0 + math.sin(i * 0.02)
        controller.compute(traj, ego, perception)

    return _time(tick, ticks, warmup)


def _fmt_row(name: str, py_s: float, ad_s: float) -> str:
    py_us, ad_us = py_s * 1e6, ad_s * 1e6
    if has_cpp():
        ratio = py_us / ad_us if ad_us > 0 else float("inf")
        verdict = f"speedup {ratio:6.2f}x" if ratio >= 1.0 else \
                  f"slowdown {ad_us / py_us:6.2f}x"
    else:
        overhead = (ad_us / py_us) if py_us > 0 else float("nan")
        verdict = f"adapter overhead {overhead:5.2f}x (python fallback)"
    return (f"  {name:<24} python {py_us:9.1f} us   "
            f"adapter {ad_us:9.1f} us   {verdict}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", "-n", type=int, default=3000,
                    help="timed iterations per measurement (default 3000)")
    ap.add_argument("--warmup", type=int, default=300,
                    help="unmeasured warm-up iterations (default 300)")
    args = ap.parse_args(argv)

    cfg = Config()
    diag = backend_diagnostics()
    print("=" * 70)
    print("fsd compat benchmark — safety/control hot path")
    print("=" * 70)
    print(f"backend       : {diag['backend']}")
    if diag["module"]:
        print(f"fsd_cpp       : {diag['module']}")
    elif diag["import_error"]:
        print(f"fsd_cpp       : unavailable ({diag['import_error']})")
    print(f"ticks/warmup  : {args.ticks}/{args.warmup}")
    print("-" * 70)

    py_monitor = SafetyMonitor(cfg.safety)
    ad_monitor = CppSafetyMonitor(cfg)
    inputs = make_inputs(cfg)

    mon_py = bench_monitor(py_monitor, inputs, args.ticks, args.warmup)
    mon_ad = bench_monitor(ad_monitor, inputs, args.ticks, args.warmup)

    py_ctrl = VehicleController()
    ad_ctrl = CppVehicleController(cfg=cfg)
    ctl_py = bench_controller(py_ctrl, inputs, args.ticks, args.warmup)
    ctl_ad = bench_controller(ad_ctrl, inputs, args.ticks, args.warmup)

    print("per-call latency (mean):")
    print(_fmt_row("monitor.check()", mon_py, mon_ad))
    print(_fmt_row("controller.compute()", ctl_py, ctl_ad))
    print("-" * 70)
    print(f"monitor backend   : {ad_monitor.backend}")
    print(f"controller backend: {ad_ctrl.backend}")
    if not has_cpp():
        print("note: fsd_cpp not built — adapter timings show pure-Python")
        print("      fallback plus conversion/forwarding shim overhead.")
        print("      build with scripts/build_cpp.ps1 and re-run.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
