"""Backend detection for the optional C++ hot-path extension ``fsd_cpp``.

The C++ port (``cpp/`` — pybind11, built by ``scripts/build_cpp.ps1``) exposes
the same contracts as the pure-Python safety/control layers. This module owns
one question: *is a usable ``fsd_cpp`` importable?*

Detection order:

1. Plain ``import fsd_cpp`` — the normal case when the extension is installed
   or its build dir is already on ``sys.path`` / ``PYTHONPATH``.
2. Repo-local discovery — look for a freshly built ``fsd_cpp*.pyd`` /
   ``fsd_cpp*.so`` under ``<repo>/cpp/build`` and retry with that directory on
   ``sys.path``. This makes ``python scripts/bench_cpp.py`` and the parity
   tests work right after a local cmake build without any manual copying.

Nothing here raises: a missing or broken extension simply means
``backend() == "python"``. The import failure is retained for diagnostics via
``import_error()``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_DIRS = (
    _REPO_ROOT / "cpp" / "build",
    _REPO_ROOT / "cpp" / "build" / "Release",
    _REPO_ROOT / "cpp" / "build" / "Debug",
    _REPO_ROOT / "build",
)

_MODULE = None            # the imported fsd_cpp module, or None
_IMPORT_ERROR: Optional[BaseException] = None
_SEARCHED = False


def _try_import() -> bool:
    """Attempt ``import fsd_cpp`` once more; record failure reason."""
    global _MODULE, _IMPORT_ERROR
    try:
        import fsd_cpp  # type: ignore
        _MODULE = fsd_cpp
        _IMPORT_ERROR = None
        return True
    except Exception as exc:  # ImportError, OSError (bad DLL), version mismatch
        _IMPORT_ERROR = exc
        return False


def _discover_built_extension() -> bool:
    """Scan known build dirs for an fsd_cpp binary and retry the import.

    Returns True when a directory containing the binary was added to
    ``sys.path`` (regardless of whether the re-import then succeeded).
    """
    found = False
    for build in _BUILD_DIRS:
        if not build.is_dir():
            continue
        try:
            hits = [p for p in build.rglob("fsd_cpp*")
                    if p.suffix in (".pyd", ".so") or p.name.startswith("fsd_cpp")]
        except OSError:
            continue
        for hit in hits:
            # Skip cmake scratch files (fsd_cpp.dir/, .obj, etc.) — only real
            # loadable extensions are worth putting on sys.path.
            if hit.suffix not in (".pyd", ".so"):
                continue
            parent = str(hit.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
                found = True
    return found


def _detect() -> None:
    global _SEARCHED
    if _SEARCHED:
        return
    _SEARCHED = True
    if _try_import():
        return
    if _discover_built_extension():
        _try_import()


_detect()


def has_cpp() -> bool:
    """True when the ``fsd_cpp`` extension module imported successfully."""
    return _MODULE is not None


def backend() -> str:
    """Name of the compute backend in use: ``'cpp'`` or ``'python'``."""
    return "cpp" if has_cpp() else "python"


def cpp_module():
    """The imported ``fsd_cpp`` module object, or None when unavailable."""
    return _MODULE


def import_error() -> Optional[BaseException]:
    """The last import failure, for diagnostics. None when import succeeded."""
    return _IMPORT_ERROR


def diagnostics() -> dict:
    """A small status dict suitable for logging or a status() payload."""
    return {
        "backend": backend(),
        "has_cpp": has_cpp(),
        "module": getattr(_MODULE, "__file__", None) if _MODULE else None,
        "import_error": repr(_IMPORT_ERROR) if _IMPORT_ERROR else None,
        "searched_build_dirs": [str(d) for d in _BUILD_DIRS if d.is_dir()],
    }


def refresh() -> bool:
    """Re-run detection (e.g. after building the extension in-session)."""
    global _SEARCHED
    if _MODULE is not None:
        return True
    _SEARCHED = False
    _detect()
    return has_cpp()


__all__ = [
    "has_cpp",
    "backend",
    "cpp_module",
    "import_error",
    "diagnostics",
    "refresh",
]
