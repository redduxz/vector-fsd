# Running the city demo

What works today, and what bit us getting here.

## TL;DR

```bash
# 1. start CARLA (DX11, Town05 — see "hard-won notes" below)
"C:\Users\R\CARLA\CARLA_0.9.16_extract\CarlaUE4.exe" /Game/Carla/Maps/Town05 -windowed -ResX=1280 -ResY=720 -dx11

# 2. autopilot + web UI (repo root)
python ui/dashboard.py --config configs/demo.yaml --port 8085

# or the one-command version (it can spawn the server too)
python scripts/run_demo.py --carla-exe "C:\Users\R\CARLA\CARLA_0.9.16_extract\CarlaUE4.exe" --town /Game/Carla/Maps/Town05
```

Open `http://127.0.0.1:8085`. Camera feed (driver view) on the left,
bird's-eye world state + semantic perception feed on the right, safety
log underneath.

## What you'll see

- Tesla Model 3 driving Town05 under lane-level A* routing, replanned
  every tick — junctions, traffic lights, lead-vehicle following,
  posted speed limits, junction-approach speed taper
- Projected detection boxes + traffic-light banner on the camera
- BEV: occupancy corridor, trajectory polyline, class-coloured actors,
  STOP marker at the active stop line
- Drive mode: ENGAGED / DEGRADED / SAFE_STOP with the safety log
  explaining every transition
- Stuck recovery: wedged ~3s -> reverse manoeuvre -> relocate to a
  fresh spawn point (max 3 per run)
- Agent faults surface as a red banner on the page, not a frozen UI

## Data + evaluation flags (headless autopilot)

```bash
# log (obs, act) .npz episodes for the imitation track
python -m fsd.agents.autopilot --config configs/demo.yaml --record runs/ep1

# stream the run through ClosedLoopMetrics -> JSON report
python -m fsd.agents.autopilot --config configs/demo.yaml --metrics-out runs/ep1-metrics.json
```

Metrics include distance, min TTC/gap, RMS jerk, mode breakdown, and
per-rule safety hits — same numbers `fsd/eval/report.py` renders for
the scenario suite.

## Hard-won notes (this machine)

- **DX11, not D3D12.** CARLA 0.9.16 `LowLevelFatalError`s on the RTX
  4060 under D3D12. `-dx11` fixes it; the first boot recompiles all
  shaders (GPU pegged, RPC starved for minutes — wait it out).
- **Town05, not Town10HD.** Town10HD at 720p+ stalled ticks to ~2 s
  (safety-cycle overruns, DEGRADED flapping). Town05 is far lighter and
  still a real city.
- **Sensor load.** 64-ch LiDAR = 1.28 M pts/s parsed in NumPy per tick.
  `configs/demo.yaml` uses 16 ch — occupancy still fed, ticks stay
  real-time.
- **Traffic.** 40 vehicles + 20 pedestrians killed the first run;
  12 + 6 is comfortable.
- **Port 8080** is squatted by a stray `http.server` process on this
  box — the dashboard defaults were moved to 8085.

## Known limitations (honest list)

- Object detection runs on CARLA ground-truth actors (`backend: none`),
  not camera inference — swap `perception.object_backend` to `yolo` or
  `onnx` with a model to change that.
- Lane geometry comes from map waypoints (authoritative), vision is
  the fallback path only.
- The route goal is "200 m ahead", replanned per tick — deliberate
  endless driving for demos, not navigation to a destination.
- `monitor.check()` through the pybind adapter is ~1.9x slower than
  pure Python (dict conversion overhead) while `controller.compute()`
  is ~20x faster. Net win; the monitor conversion layer is the
  documented optimisation target.
