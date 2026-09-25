"""train_e2e.py — imitation-learning trainer on recorded CARLA runs (npz episodes).

If torch is available: small CNN policy net. Otherwise: numpy ridge baseline so the
pipeline always runs. Checkpoints land in fsd/ml/checkpoints/.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from fsd.core.logger import get

log = get("ml.train")

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def load_episodes(pattern: str):
    obs, act = [], []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        obs.append(z["obs"].astype(np.float32))
        act.append(z["act"].astype(np.float32))
    if not obs:
        raise SystemExit(f"no episodes matched: {pattern}")
    return np.concatenate(obs), np.concatenate(act)


def train_numpy(obs, act, epochs=20, lr=1e-3, l2=1e-4):
    """Ridge-regression baseline: obs flattened -> [steer, throttle, brake]."""
    X = obs.reshape(len(obs), -1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = np.hstack([X, np.ones((len(X), 1))])
    W = np.zeros((X.shape[1], act.shape[1]))
    for ep in range(epochs):
        grad = (X.T @ (X @ W - act)) / len(X) + l2 * W
        W -= lr * grad
        if ep % 5 == 0:
            mse = float(((X @ W - act) ** 2).mean())
            log.info(f"epoch {ep}: mse={mse:.4f}")
    return W


def train_torch(obs, act, epochs=20, lr=1e-3):
    import torch
    import torch.nn as nn

    class PolicyNet(nn.Module):
        def __init__(self, in_shape, n_out=3):
            super().__init__()
            c, h, w = in_shape
            self.features = nn.Sequential(
                nn.Conv2d(c, 16, 5, 2), nn.ReLU(),
                nn.Conv2d(16, 32, 5, 2), nn.ReLU(),
                nn.Conv2d(32, 64, 5, 2), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Linear(64, n_out)

        def forward(self, x):
            return self.head(self.features(x))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyNet(obs.shape[1:]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    X = torch.tensor(obs, device=dev)
    Y = torch.tensor(act, device=dev)
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(net(X), Y)
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            log.info(f"epoch {ep}: mse={loss.item():.4f}")
    return net


def run(argv):
    ap = argparse.ArgumentParser(description="Train e2e policy on recorded episodes")
    ap.add_argument("data", help="glob of npz episodes, e.g. 'runs/*.npz'")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--numpy", action="store_true", help="force numpy baseline")
    a = ap.parse_args(argv)

    obs, act = load_episodes(a.data)
    log.info(f"{len(obs)} samples, obs={obs.shape}, act={act.shape}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    try:
        import torch  # noqa: F401
        have_torch = not a.numpy
    except ImportError:
        have_torch = False

    if have_torch:
        net = train_torch(obs, act, a.epochs, a.lr)
        path = os.path.join(CKPT_DIR, f"policy-{ts}.pt")
        torch.save(net.state_dict(), path)
    else:
        W = train_numpy(obs, act, a.epochs, a.lr)
        path = os.path.join(CKPT_DIR, f"policy-{ts}.npz")
        np.savez(path, W=W)
    log.info(f"saved checkpoint: {path}")
    return path


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
