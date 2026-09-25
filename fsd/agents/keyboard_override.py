"""Manual takeover — keyboard listener stub with graceful degradation.

Uses ``pynput`` when available; on headless boxes or missing deps the whole
listener collapses to a no-op and :meth:`takeover_requested` simply never
fires — the autopilot keeps full authority.

Controls (when live):

    W / Up      throttle
    S / Down    brake
    A / Left    steer left
    D / Right   steer right
    Space       e-brake (hand brake)
    Esc         latch manual mode (vehicle holds a gentle stop)
    R           resume autonomy / unlatch
"""
from __future__ import annotations

import threading
from typing import Optional, Set

from fsd.core.logger import get
from fsd.core.types import ControlCommand

log = get("agents.keyboard_override")

_UP = {"w", "up"}
_DOWN = {"s", "down"}
_LEFT = {"a", "left"}
_RIGHT = {"d", "right"}
_BRAKE_HOLD = 0.5          # brake level while latched with no input
_STEER_GAIN = 0.55         # full deflection authority per key


class ManualOverride:
    """Poll-based manual takeover source.

    ``takeover_requested`` is True while a movement key is held or while the
    ESC latch is set. The autopilot polls it every tick; when True it applies
    :meth:`manual_command` instead of the pipeline output.
    """

    def __init__(self):
        self._pressed: Set[str] = set()
        self._latched = False
        self._listener = None
        self._lock = threading.Lock()
        self._available = False
        self._warned = False

    # ------------------------------------------------------------------ API

    def start(self) -> bool:
        """Start the pynput listener. Returns False when unavailable."""
        try:
            from pynput import keyboard  # type: ignore
        except Exception:
            if not self._warned:
                log.info("pynput not installed — manual override disabled")
                self._warned = True
            return False

        def _name(key) -> str:
            ch = getattr(key, "char", None)
            if ch:
                return str(ch).lower()
            return str(key).rsplit(".", 1)[-1].lower()

        def _on_press(key):
            name = _name(key)
            with self._lock:
                if name == "esc":
                    self._latched = True
                elif name == "r":
                    self._latched = False
                else:
                    self._pressed.add(name)

        def _on_release(key):
            with self._lock:
                self._pressed.discard(_name(key))

        try:
            self._listener = keyboard.Listener(
                on_press=_on_press, on_release=_on_release)
            self._listener.daemon = True
            self._listener.start()
            self._available = True
            log.info("manual override live: WASD/arrows drive, "
                     "ESC latches manual, R resumes")
        except Exception as exc:
            # e.g. X11/headless failure inside pynput
            log.info("keyboard listener unavailable (%s) — override disabled",
                     exc)
            self._listener = None
        return self._available

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self._available = False
        with self._lock:
            self._pressed.clear()
            self._latched = False

    # ------------------------------------------------------------------ state

    @property
    def available(self) -> bool:
        return self._available

    @property
    def takeover_requested(self) -> bool:
        with self._lock:
            return self._latched or bool(
                self._pressed & (_UP | _DOWN | _LEFT | _RIGHT | {"space"}))

    def manual_command(self) -> ControlCommand:
        """Map held keys to a ControlCommand (gentle hold when latched)."""
        with self._lock:
            pressed = set(self._pressed)
            latched = self._latched
        cmd = ControlCommand()
        if "space" in pressed:
            cmd.hand_brake = True
            cmd.brake = 1.0
            return cmd
        if pressed & _UP:
            cmd.throttle = 1.0
        if pressed & _DOWN:
            cmd.brake = 1.0
        steer = float(bool(pressed & _RIGHT)) - float(bool(pressed & _LEFT))
        cmd.steer = steer * _STEER_GAIN
        if latched and not (pressed & (_UP | _DOWN | _LEFT | _RIGHT)):
            cmd.brake = _BRAKE_HOLD       # hold the vehicle, don't coast
        return cmd.clamp()

    # Context manager -------------------------------------------------------

    def __enter__(self) -> "ManualOverride":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
