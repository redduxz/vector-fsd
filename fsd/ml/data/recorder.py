"""recorder.py — RunRecorder: logs (observation, command) pairs for imitation learning."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from fsd.core.logger import get
from fsd.core.types import ControlCommand

log = get("ml.recorder")


class RunRecorder:
    """Accumulates per-tick (obs, act) pairs, flushes to a compressed .npz episode."""

    def __init__(self, out_dir: str = "runs", obs_shape=(3, 128, 256),
                 flush_every: int = 2000):
        self.out_dir = out_dir
        self.obs_shape = obs_shape
        self.flush_every = flush_every
        self._obs, self._act = [], []
        self._episodes = 0
        os.makedirs(out_dir, exist_ok=True)

    def record(self, frame: np.ndarray, cmd: ControlCommand,
               ego_speed: float = 0.0) -> None:
        """frame: HxWx3 uint8 camera image; cmd: the command actually applied."""
        obs = self._encode_obs(frame, ego_speed)
        act = np.array([cmd.steer, cmd.throttle, cmd.brake], dtype=np.float32)
        self._obs.append(obs)
        self._act.append(act)
        if len(self._obs) >= self.flush_every:
            self.flush()

    def _encode_obs(self, frame: np.ndarray, speed: float) -> np.ndarray:
        """Downsample to obs_shape and append speed as a constant channel."""
        h, w = frame.shape[:2]
        c, oh, ow = self.obs_shape
        img = frame[np.ix_(
            np.linspace(0, h - 1, oh).astype(int),
            np.linspace(0, w - 1, ow).astype(int))].astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # CHW
        if c == 4:  # append speed channel
            spd = np.full((1, oh, ow), min(speed / 30.0, 1.0), dtype=np.float32)
            img = np.concatenate([img, spd])
        return img[:c]

    def flush(self) -> Optional[str]:
        if not self._obs:
            return None
        path = os.path.join(
            self.out_dir, f"ep-{time.strftime('%Y%m%d-%H%M%S')}-{self._episodes}.npz")
        np.savez_compressed(
            path,
            obs=np.stack(self._obs).astype(np.float16),
            act=np.stack(self._act))
        log.info(f"flushed episode: {path} ({len(self._obs)} samples)")
        self._obs, self._act = [], []
        self._episodes += 1
        return path

    def close(self) -> Optional[str]:
        return self.flush()
