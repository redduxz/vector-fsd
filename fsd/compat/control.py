"""CppVehicleController — VehicleController-compatible facade over ``fsd_cpp``.

Same public API as ``fsd.control.controller.VehicleController``; per call it
converts ``Trajectory``/``VehicleState``/``PerceptionOutput`` into the
extension's mirrored types, invokes ``fsd_cpp.VehicleController.compute(...)``
and rebuilds a real ``fsd.core.types.ControlCommand`` from whatever comes back.

When ``fsd_cpp`` is absent the adapter wraps the pure-Python
``VehicleController`` — identical signature, identical semantics. Any fault on
the extension path returns a firm-brake command rather than raising into the
control loop: a controller that crashes is itself a driving hazard.
"""
from __future__ import annotations

from typing import Optional

from fsd.core.logger import get
from fsd.core.types import (
    ControlCommand,
    PerceptionOutput,
    Trajectory,
    VehicleState,
)

from fsd.compat.backend import cpp_module, has_cpp
from fsd.compat.convert import (
    _section_dict,
    command_from_cpp,
    perception_to_cpp,
    trajectory_to_cpp,
    vehicle_state_to_cpp,
)

from fsd.control.controller import VehicleController as _PyVehicleController


class CppVehicleController:
    """Drop-in replacement for ``VehicleController`` backed by ``fsd_cpp``.

    Signature mirrors the Python controller exactly, plus an optional
    keyword-only ``cfg`` (``Config`` or dict) whose ``vehicle``/``safety``
    sections are merged into the ``cfg_dict`` handed to the extension.
    A ``Config`` passed positionally as the first argument (the ``_build``
    helper convention in ``fsd.agents.autopilot``) is detected and treated as
    ``cfg`` rather than as a wheelbase.
    """

    def __init__(
        self,
        wheelbase_m: float = 2.875,
        max_steer_deg: float = 60.0,
        max_steer_rate: float = 0.4,
        max_accel_mps2: float = 3.0,
        max_brake_mps2: float = 6.0,
        lookahead_m: float = 8.0,
        use_mpc: bool = True,
        mpc_blend: float = 0.35,
        mpc_min_speed: float = 1.0,
        *,
        cfg=None,
    ) -> None:
        self.log = get("compat.control")

        # A Config/dict parked in the first positional slot means the caller
        # follows the ``cls(cfg)`` convention — absorb it.
        if cfg is None and not isinstance(wheelbase_m, (int, float)):
            cfg, wheelbase_m = wheelbase_m, 2.875
            v = _section_dict(getattr(cfg, "vehicle", None))
            wheelbase_m = float(v.get("wheelbase_m", wheelbase_m) or wheelbase_m)

        self.params = {
            "wheelbase_m": float(wheelbase_m),
            "max_steer_deg": float(max_steer_deg),
            "max_steer_rate": float(max_steer_rate),
            "max_accel_mps2": float(max_accel_mps2),
            "max_brake_mps2": float(max_brake_mps2),
            "lookahead_m": float(lookahead_m),
            "use_mpc": bool(use_mpc),
            "mpc_blend": float(mpc_blend),
            "mpc_min_speed": float(mpc_min_speed),
        }

        self.cfg_dict = self._build_cfg_dict(cfg)
        self._prev_steer = 0.0
        self._prev_ts: Optional[float] = None
        self._impl = self._build_cpp() if has_cpp() else None
        if self._impl is None:
            self._impl = _PyVehicleController(**self.params)
        self._is_cpp = not isinstance(self._impl, _PyVehicleController)
        self.log.info("CppVehicleController online — backend=%s",
                      "cpp" if self._is_cpp else "python")

    # ------------------------------------------------------------- backends

    def _build_cfg_dict(self, cfg) -> dict:
        """Control params flat + under 'control', plus vehicle/safety sections.

        Laying keys out both ways means the C++ ctor works whether it reads
        ``cfg["lookahead_m"]`` or ``cfg["control"]["lookahead_m"]``.
        """
        d = dict(self.params)
        d["control"] = dict(self.params)
        if cfg is not None:
            from fsd.compat.convert import cfg_to_dict
            if isinstance(cfg, dict):
                base = cfg
            else:
                base = cfg_to_dict(cfg)
            for section in ("vehicle", "safety", "sim"):
                sub = base.get(section)
                if isinstance(sub, dict):
                    d[section] = sub
        return d

    def _build_cpp(self):
        cls = getattr(cpp_module(), "VehicleController", None)
        if cls is None:
            self.log.warning("fsd_cpp has no VehicleController — using python")
            return None
        for args, kwargs in (
            ((self.cfg_dict,), {}),
            ((), dict(self.params)),
            ((self.params,), {}),
            ((), {}),
        ):
            try:
                return cls(*args, **kwargs)
            except Exception as exc:
                self.log.debug("fsd_cpp.VehicleController ctor rejected: %s", exc)
        self.log.warning("fsd_cpp.VehicleController ctor failed — using python")
        return None

    # ------------------------------------------------------------------ API

    def compute(
        self,
        trajectory: Optional[Trajectory],
        ego: VehicleState,
        perception: Optional[PerceptionOutput] = None,
    ) -> ControlCommand:
        """Trajectory + state + perception -> clamped actuation command."""
        if not self._is_cpp:
            return self._impl.compute(trajectory, ego, perception)
        try:
            c_traj = trajectory_to_cpp(trajectory) if trajectory is not None else None
            c_ego = vehicle_state_to_cpp(ego)
            c_perc = perception_to_cpp(perception) \
                if perception is not None else None
            raw = self._impl.compute(c_traj, c_ego, c_perc)
            return command_from_cpp(raw)
        except Exception:
            self.log.exception("cpp controller.compute fault — firm brake")
            return ControlCommand(throttle=0.0, brake=0.8, steer=0.0).clamp()

    # Friendly aliases — the python controller exposes step = compute.
    step = compute

    # -------------------------------------------------------------- introspect

    @property
    def backend(self) -> str:
        return "cpp" if self._is_cpp else "python"

    @property
    def impl(self):
        return self._impl

    def reset(self) -> None:
        """Clear carried state (rate limiter, integrators) on the backend."""
        fn = getattr(self._impl, "reset", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                self.log.exception("backend reset() fault")
        self._prev_steer = 0.0
        self._prev_ts: Optional[float] = None

    def __getattr__(self, name):
        """Forward unimplemented attributes (lateral, mpc, ...) to the impl."""
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(name)
        try:
            return getattr(impl, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r} "
                f"(backend={self.backend})") from None


__all__ = ["CppVehicleController"]
