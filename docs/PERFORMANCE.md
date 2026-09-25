# Performance programme

Notes on where the Python implementation spends its time, and the path to a
production-grade latency budget. This reflects the current research position —
it will evolve as the stack matures.

## Why Python alone is not enough

Every production autonomous-driving stack of note — Autoware, Apollo, and the
NVIDIA DRIVE stack itself — runs C++ on the hot path. Python is appropriate for
orchestration, configuration, and research iteration; it is not appropriate for
a hard 20 Hz control loop once real perception models enter the pipeline.

Indicative figures from the literature:

- Autoware (ROS 2 / FastDDS) sensor-data hop: ~3.8 ms per message
- Apollo (CyberRT, shared memory): ~1.7 µs — three orders of magnitude lower
- TensorRT (INT8) network inference on embedded silicon: ~20 ms end-to-end
- A production target for full E2E latency (sensor → actuator): **< 30 ms**

Our current Python loop comfortably meets 50 ms on synthetic scenes, but camera +
LiDAR inference will blow the budget immediately. The migration plan below
front-loads the components that are both latency-critical and numerically small.

## Migration plan

### Phase 1 — C++ core (highest leverage, smallest surface)

Port to C++17 first, behind pybind11 bindings so the Python stack keeps working:

1. **`fsd.safety.monitor`** — the safety gate is pure arithmetic (TTC, limits,
   watchdog comparisons). Tiny, critical, and the cheapest possible port.
2. **`fsd.control`** — PID / Stanley / MPC-lite are all closed-form math; trivial
   in C++, and control benefits most from deterministic timing.
3. **`fsd.perception.fusion`** — Kalman association is O(n²) bookkeeping; the
   first genuinely hot module once objects scale past tens.

Build system: CMake + pybind11, `fsd_cpp` extension module. The Python classes
become thin adapters, so existing tests and the smoke loop continue to pass
against either backend.

### Phase 2 — inference acceleration

- Export detection/segmentation networks to **ONNX**, run through
  **TensorRT** (FP16 then INT8 calibration) — the `fsd.ml.models` registry
  already abstracts backends, so this is a drop-in.
- Reserve CUDA streams; pin memory for camera frames.

### Phase 3 — middleware (optional, longer-term)

If the stack ever targets real hardware or multi-process deployment, the
industry answer is **ROS 2** (Autoware) or **CyberRT** (Apollo). CyberRT's
shared-memory transport is the performance leader; ROS 2 has the ecosystem. We
defer this — a single-process pipeline with a hard gate is the right complexity
level for a research stack — but `fsd.core.types` deliberately resembles message
schemas so a DDS mapping stays mechanical.

### Phase 4 — end-to-end experiment

A parallel track under `fsd.ml`: a single network mapping camera → trajectory
(VAD-style, trained on recorded CARLA runs via `fsd.ml.data.recorder`). Kept
strictly behind the safety gate — an end-to-end policy that cannot veto itself
is not one we drive with.

## Budget

| Stage | Budget at 20 Hz | Notes |
| --- | --- | --- |
| Sensor ingest | 5 ms | CARLA callback → typed frame |
| Perception | 15 ms | TensorRT engine, batched |
| Fusion + occupancy | 5 ms | C++ core |
| Planning | 10 ms | A* amortised; trajectory each tick |
| Control + safety gate | 5 ms | C++ core |
| **Total** | **40 ms** | leaves 10 ms headroom per tick |

## Non-goals

- Real-time OS / kernel tuning — unnecessary inside a simulator loop.
- ROS 2 migration for its own sake — only when distribution is actually needed.
- Rust — a fine language, but the automotive C++ ecosystem (AUTOSAR idioms,
  compiler qualification, LibCarla) is the reason to pick C++ specifically.
