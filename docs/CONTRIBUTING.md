# Contributing

Thanks for helping build `fsd`. This document covers the ground rules that keep
a multi-module autonomy codebase readable and safe.

## Setup

```bash
pip install -r requirements.txt
python -m compileall fsd            # byte-compile check
python -m unittest discover tests   # test suite (no pytest needed)
```

## Code style

- **Python 3.11**, PEP 8, 4-space indents, `from __future__ import annotations`.
- **Type hints** on all public signatures; `dataclasses` for every data
  container that crosses a module boundary.
- **Units in field names** — `m`, `mps`, `mps2`, `rad`, `s` (see
  `SafetyConfig` for the convention). Bare numbers are a review blocker.
- **Logging** via `fsd.core.logger.get("module_name")`. Never `print`, and
  never configure logging inside a library module.
- **Docstrings** on every public class/function: one line summary, then
  args/returns where non-obvious. Comments explain *why*, not *what*.
- **No hidden global state.** Config comes from `Config.load`, state lives in
  the module's class instances.
- **Only `fsd/carla_bridge/` may `import carla`.** Downstream modules must stay
  runnable without the simulator so `tests/` works headless.
- Keep dependencies minimal — `numpy` and `pyyaml` are the core set. Anything
  else (torch, onnxruntime, flask) is optional and must be imported lazily
  inside the code path that needs it.

## Module ownership

Each module owns its contract and its tests. If you change a contract in
`fsd/core/types.py`, you own the fallout in every consumer.

| Path | Owns | Talk to before changing |
| --- | --- | --- |
| `fsd/core/` | types, config, logger | everyone — contracts are load-bearing |
| `fsd/carla_bridge/` | world, vehicle, sensors, traffic | perception (input shape) |
| `fsd/perception/` | detectors, fusion, occupancy | planning, safety (`PerceptionOutput`) |
| `fsd/planning/` | behavior, route, trajectory, costmap | control, safety |
| `fsd/control/` | pid, lateral, longitudinal, mpc | safety (command shape) |
| `fsd/safety/` | monitor, rules, diagnostics, fallback | **safety review required** |
| `fsd/agents/` | autopilot loop, keyboard override | all of the above |
| `fsd/ml/` | models, infer, train, recorder | perception |
| `configs/`, `docs/`, `tests/` | this layer | — |

## PR checklist

Before requesting review:

- [ ] `python -m compileall fsd` is clean
- [ ] `python -m unittest discover tests` passes (new features → new tests)
- [ ] Public functions typed and docstringed
- [ ] No new hard dependency (optional deps are lazy-imported)
- [ ] Config changes update the matching `configs/*.yaml` and `core/config.py`
- [ ] Docs updated (`README.md`, `docs/ARCHITECTURE.md`, `docs/SAFETY.md`)
- [ ] Nothing bypasses the safety gate — if the PR touches `fsd/safety/` or the
      command path, say so loudly in the description

## Safety-related changes

PRs touching `fsd/safety/`, `ControlCommand`, or the autopilot actuation path
get extra scrutiny:

1. State which rule/invariant is affected and why the change is safe.
2. Add the fault-injection test that would have caught a regression.
3. Never weaken a threshold to silence a test — fix the code that trips it.

## Commit / PR hygiene

- Small, focused commits; imperative subject lines ("add TTC rule", not "stuff").
- Reference the module in the subject where helpful: `safety: latch SAFE_STOP`.
- A PR that only moves code should not change behavior; a PR that changes
  behavior should not move code. Keeping the two apart halves review time.
