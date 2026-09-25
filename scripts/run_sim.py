"""Convenience launcher: ensure a CARLA server is running, then start autopilot.

Usage:
    python scripts/run_sim.py --config configs/default.yaml
    python scripts/run_sim.py --config configs/highway.yaml --launch-server
    python scripts/run_sim.py --config configs/default.yaml -- --verbose

Anything after ``--`` (or any unrecognized arguments) is forwarded verbatim to
``python -m fsd.agents.autopilot``.
"""
from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SERVER_BINARIES = {
    "Windows": ("CarlaUE4.exe",),
    "Linux": ("CarlaUE4.sh",),
    "Darwin": ("CarlaUE4.sh",),
}


def _sim_endpoint(config_path: str) -> tuple[str, int]:
    """Read sim.host / sim.port out of the YAML config, with sane fallbacks."""
    host, port = "127.0.0.1", 2000
    try:
        import yaml  # optional at launch time — fall back to defaults without it

        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        sim = data.get("sim") or {}
        host = str(sim.get("host", host))
        port = int(sim.get("port", port))
    except (OSError, ValueError, TypeError):
        pass  # missing file or malformed YAML: the autopilot will complain later
    return host, port


def _server_up(host: str, port: int, timeout: float = 1.5) -> bool:
    """True if a TCP connection to the CARLA RPC port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _find_server_binary() -> str | None:
    """Locate the CARLA server executable under $CARLA_ROOT."""
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root or not os.path.isdir(carla_root):
        return None
    names = _SERVER_BINARIES.get(platform.system(), ())
    for root, _dirs, files in os.walk(carla_root):
        for name in names:
            if name in files:
                return os.path.join(root, name)
    return None


def _launch_server(port: int) -> subprocess.Popen | None:
    """Spawn a CARLA server process. Returns None if no binary was found."""
    binary = _find_server_binary()
    if binary is None:
        return None
    args = [binary, f"-carla-rpc-port={port}", "-quality-level=Low"]
    if platform.system() == "Windows":
        args.append("-dx11")
    return subprocess.Popen(args, cwd=os.path.dirname(binary))


def _wait_for_server(host: str, port: int, timeout_s: float) -> bool:
    """Poll the RPC port until it accepts connections or the deadline hits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _server_up(host, port):
            return True
        time.sleep(1.0)
    return False


def _run_autopilot(config: str, extra: list[str]) -> int:
    """Run ``python -m fsd.agents.autopilot`` from the repo root."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "fsd.agents.autopilot", "--config", config, *extra]
    print(f"[run_sim] exec: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wait for a CARLA server, then run the fsd autopilot.",
        epilog="Unrecognized arguments are forwarded to fsd.agents.autopilot.",
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config passed to the autopilot.")
    parser.add_argument("--host", default=None,
                        help="CARLA RPC host (default: sim.host from --config).")
    parser.add_argument("--port", type=int, default=None,
                        help="CARLA RPC port (default: sim.port from --config).")
    parser.add_argument("--launch-server", action="store_true",
                        help="Start a CARLA server from $CARLA_ROOT if none is listening.")
    parser.add_argument("--wait-timeout", type=float, default=120.0,
                        help="Seconds to wait for the server (default: 120).")
    known, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    config = known.config
    if not os.path.isabs(config):
        config = os.path.join(REPO_ROOT, config)

    cfg_host, cfg_port = _sim_endpoint(config)
    host = known.host or cfg_host
    port = known.port or cfg_port

    if not _server_up(host, port):
        print(f"[run_sim] no CARLA server on {host}:{port}")
        if known.launch_server:
            server = _launch_server(port)
            if server is None:
                print("[run_sim] $CARLA_ROOT not set or no server binary found — "
                      "start CarlaUE4 manually and retry.")
                return 2
            print(f"[run_sim] launched server (pid {server.pid}); waiting up to "
                  f"{known.wait_timeout:.0f}s for {host}:{port} ...")
        else:
            print(f"[run_sim] waiting up to {known.wait_timeout:.0f}s "
                  "(pass --launch-server to auto-start from $CARLA_ROOT) ...")
        if not _wait_for_server(host, port, known.wait_timeout):
            print(f"[run_sim] timed out waiting for {host}:{port} — aborting.")
            return 2

    print(f"[run_sim] CARLA server reachable on {host}:{port}")
    return _run_autopilot(config, extra)


if __name__ == "__main__":
    sys.exit(main())
