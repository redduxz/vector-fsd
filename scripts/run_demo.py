#!/usr/bin/env python3
"""One-command city demo: CARLA up-check -> optional server launch -> dashboard.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --carla-exe "C:\\CARLA\\CarlaUE4.exe" --town Town05
    python scripts/run_demo.py --no-carla            # synthetic smoke mode

Steps it performs:
  1. probe 127.0.0.1:2000 for a live CARLA server
  2. if absent and --carla-exe given, spawn it (DX11 — D3D12 asserts on
     RTX-class cards in 0.9.16) and wait for the RPC port
  3. exec ui/dashboard.py which runs the autopilot in-process and serves
     the web UI (default http://127.0.0.1:8085)
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), 2):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="vector-fsd city demo")
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rpc-port", type=int, default=2000)
    ap.add_argument("--carla-exe", default="",
                    help="path to CarlaUE4.exe — spawned if server is down")
    ap.add_argument("--town", default="", help="CARLA map path, e.g. "
                    "/Game/Carla/Maps/Town05 (server boot only)")
    ap.add_argument("--dx11", action="store_true", default=True,
                    help="force DirectX 11 (avoids the 0.9.16 D3D12 assert)")
    ap.add_argument("--no-dx11", dest="dx11", action="store_false")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--no-carla", action="store_true")
    ap.add_argument("--wait", type=float, default=120.0,
                    help="seconds to wait for the server")
    a = ap.parse_args()

    if a.no_carla:
        cmd = [sys.executable, os.path.join(ROOT, "ui", "dashboard.py"),
               "--config", a.config, "--no-carla", "--port", str(a.port)]
        print("[demo] smoke mode:", " ".join(cmd))
        os.chdir(ROOT)
        return subprocess.call(cmd)

    # 1. is a server already up?
    up = _wait_port(a.host, a.rpc_port, 3.0)

    # 2. spawn one if we were given the binary
    server = None
    if not up:
        if not a.carla_exe:
            print(f"[demo] no CARLA on {a.host}:{a.rpc_port} and no "
                  f"--carla-exe given.\n       start the simulator first "
                  f"or pass --carla-exe <path to CarlaUE4.exe>")
            return 2
        if not os.path.exists(a.carla_exe):
            print(f"[demo] carla exe not found: {a.carla_exe}")
            return 2
        res_x, res_y = a.res.split("x")
        cmd = [a.carla_exe]
        if a.town:
            cmd.append(a.town)
        cmd += ["-windowed", f"-ResX={res_x}", f"-ResY={res_y}"]
        if a.dx11:
            cmd.append("-dx11")
        print("[demo] launching CARLA:", " ".join(cmd))
        server = subprocess.Popen(
            cmd, cwd=os.path.dirname(a.carla_exe),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[demo] waiting for {a.host}:{a.rpc_port} "
              f"(first DX11 boot compiles shaders — can take minutes)")
        if not _wait_port(a.host, a.rpc_port, a.wait):
            print("[demo] server never came up — check the CARLA window")
            server.terminate()
            return 3
        print("[demo] CARLA is up")

    # 3. hand over to the dashboard (autopilot inside)
    print(f"[demo] starting dashboard on http://127.0.0.1:{a.port}")
    os.chdir(ROOT)
    cmd = [sys.executable, os.path.join("ui", "dashboard.py"),
           "--config", a.config, "--port", str(a.port)]
    try:
        return subprocess.call(cmd)
    finally:
        if server is not None:
            print("[demo] stopping CARLA we spawned")
            server.terminate()


if __name__ == "__main__":
    sys.exit(main())
