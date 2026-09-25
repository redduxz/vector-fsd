#!/usr/bin/env python3
"""vector-fsd live dashboard — annotated camera + bird's-eye + HUD.

Usage:
    python ui/dashboard.py --config configs/default.yaml
    python ui/dashboard.py --no-carla            # synthetic smoke mode
    then open http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from fsd.core.config import Config
from fsd.core.types import DriveMode

app = Flask(__name__)

# Shared state written by the driving thread, read by HTTP handlers.
STATE = {"running": False, "agent": None, "latest": {}}
LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# rendering                                                                   #
# --------------------------------------------------------------------------- #

def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def _overlay_cam(frame: np.ndarray, st: dict) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    mode = st.get("mode", "—")
    color = {"ENGAGED": (60, 220, 120), "DEGRADED": (0, 200, 255),
             "SAFE_STOP": (60, 60, 240), "DISENGAGED": (180, 180, 180)}.get(
        mode, (200, 200, 200))
    cv2.rectangle(img, (0, 0), (w, 46), (10, 14, 18), -1)
    cv2.putText(img, f"{mode}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(img, f"{st.get('speed_kph', 0):5.1f} km/h", (w - 250, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 255), 2)
    lane = st.get("lane", {})
    if lane.get("detected"):
        off = lane.get("center_offset", 0.0)
        bar_w = 260
        cx = w // 2
        y = h - 36
        cv2.rectangle(img, (cx - bar_w // 2, y), (cx + bar_w // 2, y + 10),
                      (40, 44, 52), -1)
        px = int(np.clip(off / 2.0, -1, 1) * bar_w / 2)
        cv2.rectangle(img, (cx + px - 4, y - 4), (cx + px + 4, y + 14),
                      (80, 220, 255), -1)
        cv2.putText(img, f"lane off {off:+.2f} m", (cx - bar_w // 2, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 160, 170), 1)
    return img


def _bev(st: dict, size=560, ppm=6.0) -> np.ndarray:
    """Top-down: ego, objects, trajectory, free-space wedge, lane edges."""
    img = np.zeros((size, size, 3), np.uint8)
    cx, cy = size // 2, size - 80          # ego sits near the bottom
    ego_yaw = st.get("yaw", 0.0)

    def to_px(wx, wy):
        dx, dy = wx - st.get("x", 0.0), wy - st.get("y", 0.0)
        # rotate into ego frame (forward = -y on image)
        fx = dx * np.cos(ego_yaw) + dy * np.sin(ego_yaw)
        fy = -dx * np.sin(ego_yaw) + dy * np.cos(ego_yaw)
        return int(cx + fy * ppm), int(cy - fx * ppm)

    # grid
    for g in range(-10, 61, 10):
        cv2.line(img, (0, cy - g * ppm), (size, cy - g * ppm), (22, 26, 32), 1)
        cv2.putText(img, f"{g}m", (4, cy - g * ppm - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 66, 74), 1)

    # lane edges
    lane = st.get("lane", {})
    if lane.get("detected"):
        w2 = lane.get("lane_width", 3.5) / 2.0
        for off, col in ((w2, (70, 120, 140)), (-w2, (70, 120, 140))):
            pts = [to_px(st.get("x", 0) + f * np.cos(ego_yaw) -
                         off * np.sin(ego_yaw),
                         st.get("y", 0) + f * np.sin(ego_yaw) +
                         off * np.cos(ego_yaw))
                   for f in range(0, 61, 4)]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], col, 2)

    # free-space wedge
    fs = min(st.get("free_space", 0.0), 60.0)
    cv2.line(img, (cx, cy), (cx, int(cy - fs * ppm)), (30, 90, 60), 12)

    # trajectory
    for p in st.get("traj", [])[:45]:
        cv2.circle(img, to_px(p["x"], p["y"]), 3, (200, 160, 60), -1)

    # objects
    for o in st.get("objects", []):
        px, py = to_px(o["x"], o["y"])
        ex = max(4, int(o.get("ex", 2.2) * ppm))
        ey = max(4, int(o.get("ey", 0.9) * ppm))
        col = (60, 60, 220) if o.get("cls") == "vehicle" else (60, 180, 220)
        cv2.rectangle(img, (px - ey, py - ex), (px + ey, py + ex), col, 2)
        cv2.putText(img, o.get("cls", "?")[:8], (px - ey, py - ex - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    # ego
    cv2.rectangle(img, (cx - 10, cy - 22), (cx + 10, cy + 22), (80, 220, 255), 2)
    cv2.line(img, (cx, cy), (cx, cy - 30), (80, 220, 255), 2)
    cv2.putText(img, "EGO", (cx - 16, cy + 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (80, 220, 255), 1)
    return img


def _snapshot(agent):
    """Pull the newest sensor frames wherever they live."""
    try:
        if getattr(agent, "smoke", False) and getattr(agent, "_scene", None):
            ego = agent._synthetic_ego.state()
            return ego, agent._scene.sensors(ego, agent._tick_idx)
        if getattr(agent, "vehicle", None) is not None:
            return agent.vehicle.state(), agent.vehicle.sensors.snapshot()
    except Exception:
        pass
    return None, {}


def _mjpeg(which):
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            agent = STATE["agent"]
            st = STATE["latest"]
            out = None
            if agent is not None:
                try:
                    if which == "cam":
                        _, sensors = _snapshot(agent)
                        rd = sensors.get("camera_rgb")
                        if rd is not None:
                            out = _overlay_cam(np.asarray(rd.data), st)
                    else:
                        out = _bev(st)
                except Exception:
                    out = None
            if out is None:
                out = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(out, "waiting for pipeline...", (90, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2)
            yield boundary + _jpeg(out) + b"\r\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------- #
# routes                                                                      #
# --------------------------------------------------------------------------- #

@app.get("/stream/cam")
def stream_cam():
    return _mjpeg("cam")


@app.get("/stream/bev")
def stream_bev():
    return _mjpeg("bev")


@app.get("/state")
def state():
    return jsonify(STATE["latest"])


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>vector-fsd · live</title>
<style>
  * { box-sizing: border-box; }
  body { background:#0b0e13; color:#c9d1d9; font-family:Consolas,monospace;
         margin:0; padding:18px; }
  .grid { display:grid; grid-template-columns:1.5fr 1fr; gap:14px;
          max-width:1500px; margin:0 auto; }
  .card { background:#10151c; border:1px solid #1e2530; border-radius:10px;
          overflow:hidden; }
  .card img { width:100%; display:block; }
  .h { padding:8px 14px; font-size:12px; letter-spacing:.15em;
       color:#36bcf7; border-bottom:1px solid #1e2530; }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
           margin-bottom:14px; }
  .stat { background:#10151c; border:1px solid #1e2530; border-radius:10px;
          padding:10px 14px; }
  .stat .k { font-size:10px; color:#5b6570; letter-spacing:.1em; }
  .stat .v { font-size:22px; color:#eaf2f8; margin-top:2px; }
  #mode { font-size:26px; font-weight:bold; }
  .ENGAGED { color:#3ddc78; } .DEGRADED { color:#ffa657; }
  .SAFE_STOP { color:#f85149; } .DISENGAGED { color:#8b949e; }
  #events { font-size:11px; color:#8b949e; max-height:120px;
            overflow-y:auto; padding:8px 14px; }
  #events div { padding:2px 0; border-bottom:1px solid #161b22; }
  .bar { height:8px; background:#1e2530; border-radius:4px; margin-top:6px; }
  .bar > div { height:8px; border-radius:4px; }
</style></head><body>
<div class="grid">
  <div>
    <div class="stats">
      <div class="stat"><div class="k">DRIVE MODE</div><div id="mode" class="v">—</div></div>
      <div class="stat"><div class="k">SPEED</div><div class="v"><span id="spd">0</span> km/h</div></div>
      <div class="stat"><div class="k">FREE SPACE</div><div class="v"><span id="fs">—</span> m</div></div>
      <div class="stat"><div class="k">OBJECTS</div><div class="v" id="obj">0</div></div>
      <div class="stat"><div class="k">THROTTLE</div><div class="v" id="thr">0%</div><div class="bar"><div id="thrb" style="background:#3ddc78;width:0%"></div></div></div>
      <div class="stat"><div class="k">BRAKE</div><div class="v" id="brk">0%</div><div class="bar"><div id="brkb" style="background:#f85149;width:0%"></div></div></div>
      <div class="stat"><div class="k">STEER</div><div class="v" id="str">0</div><div class="bar"><div id="strb" style="background:#36bcf7;width:50%"></div></div></div>
      <div class="stat"><div class="k">TRAFFIC LIGHT</div><div class="v" id="tl">—</div></div>
    </div>
    <div class="card"><div class="h">CAMERA · DRIVER VIEW</div><img src="/stream/cam"></div>
  </div>
  <div>
    <div class="card"><div class="h">BIRD'S-EYE · WORLD STATE</div><img src="/stream/bev"></div>
    <div class="card" style="margin-top:14px"><div class="h">SAFETY LOG</div><div id="events"></div></div>
  </div>
</div>
<script>
setInterval(async () => {
  try {
    const s = await (await fetch('/state')).json();
    document.getElementById('mode').textContent = s.mode || '—';
    document.getElementById('mode').className = 'v ' + (s.mode || '');
    document.getElementById('spd').textContent = (s.speed_kph||0).toFixed(0);
    document.getElementById('fs').textContent = (s.free_space||0).toFixed(0);
    document.getElementById('obj').textContent = s.object_count||0;
    document.getElementById('tl').textContent = s.light||'—';
    const c = s.cmd||{};
    document.getElementById('thr').textContent = Math.round((c.throttle||0)*100)+'%';
    document.getElementById('brk').textContent = Math.round((c.brake||0)*100)+'%';
    document.getElementById('str').textContent = (c.steer||0).toFixed(2);
    document.getElementById('thrb').style.width = (c.throttle||0)*100+'%';
    document.getElementById('brkb').style.width = (c.brake||0)*100+'%';
    document.getElementById('strb').style.width = ((c.steer||0)*50+50)+'%';
    document.getElementById('strb').style.marginLeft = '-4px';
    const el = document.getElementById('events');
    el.innerHTML = (s.events||[]).slice(-14).reverse()
      .map(e=>`<div>[${e.level}] ${e.source}: ${e.message}</div>`).join('');
  } catch(e) {}
}, 200);
</script></body></html>"""


# --------------------------------------------------------------------------- #
# driving thread                                                              #
# --------------------------------------------------------------------------- #

def drive(cfg_path: str, smoke: bool):
    from fsd.agents.autopilot import AutopilotAgent
    agent = AutopilotAgent(Config.load(cfg_path), smoke=smoke)
    STATE["agent"] = agent
    agent.setup()
    while STATE["running"]:
        try:
            r = agent.tick()
        except Exception:
            continue
        ego = r.get("ego")
        cmd = r.get("cmd")
        perc = r.get("perception")
        st = {
            "mode": getattr(r.get("mode"), "name", str(r.get("mode"))),
            "tick": agent._tick_idx,
            "speed_kph": (ego.speed * 3.6) if ego else 0.0,
            "x": ego.x if ego else 0.0, "y": ego.y if ego else 0.0,
            "yaw": ego.yaw if ego else 0.0,
            "cmd": {"throttle": cmd.throttle, "brake": cmd.brake,
                    "steer": cmd.steer} if cmd else {},
            "objects": [{"x": o.position.x, "y": o.position.y, "cls": o.cls,
                         "ex": o.bbox_extent.x, "ey": o.bbox_extent.y}
                        for o in (perc.objects if perc else [])],
            "object_count": len(perc.objects) if perc else 0,
            "lane": {"detected": perc.lane.detected,
                     "center_offset": perc.lane.center_offset,
                     "lane_width": perc.lane.lane_width} if perc else {},
            "light": getattr(perc.light, "name", "—") if perc else "—",
            "free_space": perc.free_space_ahead if perc else 0.0,
            "traj": [{"x": p.x, "y": p.y}
                     for p in getattr(agent, "_last_traj", [])] if
                    getattr(agent, "_last_traj", None) else [],
            "events": [{"level": e.level, "source": e.source,
                        "message": e.message}
                       for e in getattr(getattr(agent, "safety", None),
                                        "event_log", [])[-30:]],
        }
        with LOCK:
            STATE["latest"] = st
        time.sleep(max(0.0, agent.dt - 0.0))


def main():
    ap = argparse.ArgumentParser(description="vector-fsd live dashboard")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-carla", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    STATE["running"] = True
    t = threading.Thread(target=drive, args=(a.config, a.no_carla), daemon=True)
    t.start()
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
