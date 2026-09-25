"""Structured logger — console + rotating file, per-module level control."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys

_FMT = "%(asctime)s.%(msecs)03d [%(levelname).1s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_initialized = False


def init(level: int = logging.INFO, log_dir: str = "logs") -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger("fsd")
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "fsd.log"), maxBytes=8 << 20, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(fh)


def get(name: str) -> logging.Logger:
    init()
    return logging.getLogger(f"fsd.{name}")
