"""infer.py — InferenceEngine: batching, device selection, preprocess/postprocess."""
from __future__ import annotations

import time
from typing import Any, List, Optional

import numpy as np

from fsd.core.logger import get

log = get("ml.infer")


def pick_device() -> str:
    """cuda -> directml -> cpu, whatever is actually available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        if "CUDAExecutionProvider" in provs:
            return "onnx-cuda"
        if "DmlExecutionProvider" in provs:
            return "onnx-directml"
    except ImportError:
        pass
    return "cpu"


class InferenceEngine:
    """Runs a detection model. Batches frames, normalizes inputs, NMS postprocess."""

    def __init__(self, model, input_size: int = 640, conf_thresh: float = 0.4,
                 nms_iou: float = 0.45, max_batch: int = 4):
        self.model = model
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou
        self.max_batch = max_batch
        self.device = pick_device()
        log.info(f"inference device: {self.device}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        s = self.input_size
        # letterbox-resize without cv2: nearest-neighbor decimation
        ys = (np.linspace(0, h - 1, s)).astype(int)
        xs = (np.linspace(0, w - 1, s)).astype(int)
        img = frame[np.ix_(ys, xs)].astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return img.transpose(2, 0, 1)[None]  # NCHW

    def postprocess(self, raw: np.ndarray) -> np.ndarray:
        """raw: (N, 6) = x1,y1,x2,y2,conf,cls -> NMS-filtered detections."""
        if raw.size == 0:
            return raw
        keep = raw[:, 4] >= self.conf_thresh
        return self._nms(raw[keep], self.nms_iou)

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, dets: np.ndarray, iou_thresh: float) -> np.ndarray:
        order = np.argsort(-dets[:, 4])
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            order = order[1:]
            order = order[[j for j in order
                          if self._iou(dets[i], dets[j]) < iou_thresh
                          or dets[i][5] != dets[j][5]]]
        return dets[sorted(keep)]

    def predict(self, frames: List[np.ndarray]) -> List[Any]:
        out = []
        for i in range(0, len(frames), self.max_batch):
            batch = np.concatenate([self.preprocess(f) for f in frames[i:i + self.max_batch]])
            t0 = time.time()
            raw = self.model.predict(batch)
            dt = time.time() - t0
            if dt > 0.1:
                log.warning(f"inference slow: {dt*1000:.0f}ms batch={len(batch)}")
            for det in raw:
                out.append(self.postprocess(np.asarray(det)))
        return out
