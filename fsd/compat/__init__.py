"""fsd.compat — optional C++ backend integration.

The ``cpp/`` tree builds ``fsd_cpp`` (pybind11): a hot-path port of the
safety monitor and vehicle controller. This package detects the extension,
adapts it to the ``fsd.core.types`` contracts, and falls back to the pure
Python implementations when the extension is not built.

Usage::

    from fsd.compat import CppSafetyMonitor, CppVehicleController, backend

    monitor = CppSafetyMonitor(cfg)         # same API as SafetyMonitor
    controller = CppVehicleController()     # same API as VehicleController
    mode = monitor.check(ego, perception, cmd, pipeline_alive)

Adapters are lazy-imported so ``from fsd.compat import has_cpp`` stays cheap.
"""
from __future__ import annotations

from fsd.compat.backend import (
    backend,
    cpp_module,
    diagnostics,
    has_cpp,
    import_error,
    refresh,
)

__all__ = [
    "has_cpp",
    "backend",
    "cpp_module",
    "import_error",
    "diagnostics",
    "refresh",
    "CppSafetyMonitor",
    "CppVehicleController",
]


def __getattr__(name: str):
    if name == "CppSafetyMonitor":
        from fsd.compat.safety import CppSafetyMonitor
        return CppSafetyMonitor
    if name == "CppVehicleController":
        from fsd.compat.control import CppVehicleController
        return CppVehicleController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
