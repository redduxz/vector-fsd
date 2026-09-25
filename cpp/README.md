# fsd C++ core (`fsd_core` / `fsd_cpp`)

Phase 1 of the C++ migration described in `docs/PERFORMANCE.md`: the
latency-critical, numerically small parts of the stack re-implemented in
C++17 with no external dependencies beyond (optional) pybind11.

| Piece | Ported from | Files |
| --- | --- | --- |
| Contracts | `fsd/core/types.py`, `SafetyConfig` from `fsd/core/config.py` | `include/fsd/types.hpp` |
| Safety gate | `fsd/safety/monitor.py`, `fsd/safety/rules.py` | `include/fsd/safety.hpp`, `src/safety.cpp` |
| Control | `fsd/control/{pid,lateral,longitudinal,mpc,controller}.py` | `include/fsd/control.hpp`, `src/control.cpp` |
| Occupancy | `fsd/perception/occupancy.py` | `include/fsd/occupancy.hpp`, `src/occupancy.cpp` |
| Bindings | — | `bindings/pybind_module.cpp` |
| Smoke test | — | `src/smoke_main.cpp` |

The port mirrors the Python semantics one-for-one: same rule set and
severities, same latch/de-escalation logic, same PID/Stanley/pure-pursuit/
MPC-lite math, same log-odds occupancy grid.

## Build

```sh
cmake -B build -S cpp
cmake --build build --config Release
ctest --test-dir build          # runs fsd_smoke
```

`fsd_smoke` runs a deterministic 200-tick synthetic scenario — ego
accelerating toward a stationary lead vehicle — and prints the arbitrated
`DriveMode` per tick (ENGAGED -> DEGRADED -> SAFE_STOP latch -> stopped with
hand brake), then self-checks and exits non-zero on failure.

## Python extension module (`fsd_cpp`)

```sh
pip install pybind11                      # provides the CMake package
cmake -B build -S cpp -DFSD_PYBIND=ON
cmake --build build --config Release
```

The module lands at `build/<config>/fsd_cpp.pyd` on Windows (or
`build/fsd_cpp.<abi>.so` on Linux/macOS). If `find_package(pybind11)` does
not locate it automatically, CMake falls back to `python -m pybind11
--cmakedir`; `-Dpybind11_DIR=...` also works.

### Consuming it from the Python stack

The Python classes become thin adapters — tests keep passing against either
backend (per `docs/PERFORMANCE.md`):

```python
import sys
sys.path.insert(0, "cpp/build/Release")   # wherever fsd_cpp.pyd landed
import fsd_cpp

monitor = fsd_cpp.SafetyMonitor()         # same ctor/attrs as the Python one
mode = monitor.check(ego, perception, cmd, pipeline_alive=True)
if mode is fsd_cpp.DriveMode.SAFE_STOP:
    actuate(monitor.engage_safe_stop())
else:
    actuate(monitor.enforce(cmd))
```

Everything is mirrored: `Vec3`, `VehicleState`, `Waypoint`, `Trajectory`,
`DetectedObject`, `LaneInfo`, `ControlCommand`, `SafetyEvent`,
`PerceptionOutput`, `LightState`, `DriveMode`, `LaneChangeState`,
`SafetyConfig`, `SafetyContext`, the rule classes (`TTCRule`,
`SpeedLimitRule`, `FreeSpaceRule`, `WatchdogRule`, `SteerRateRule`,
`LaneDepartureRule`, `default_rules()`), `PIDController`, `PurePursuit`,
`StanleyLateral`, `LateralController`, `LongitudinalController`, `MPCLite`,
`VehicleController`, and `OccupancyGrid`.

`Optional[...]` Python arguments map to `None` (`check(ego, perception, cmd)`
accepts `None` for any of the three). `OccupancyGrid.grid` is exposed as a
numpy copy; `as_uint8()` returns a numpy `(h, w)` image.

Two C++-only affordances useful for tests:

```python
monitor.set_clock(lambda: sim_time)          # deterministic watchdogs
monitor.set_logger(lambda lvl, msg: print(lvl, msg))
```

## Layout notes for the bridge

- `SafetyMonitor.check()` never throws and never blocks — safe for the
  20 Hz loop. `enforce()` never mutates the caller's command.
- `hazards_requested` maps to the vehicle's hazard lamps.
- The monitor clock defaults to wall time (`std::chrono::system_clock`,
  same domain as `time.time()`); inject `set_clock` in sims/tests.
