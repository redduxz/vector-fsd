"""Model registry + detector implementations for the perception front-end.

Design rules:

* ``onnxruntime`` and ``torch`` are imported lazily inside methods — this
  module is importable and fully functional with only numpy installed.
* ``predict(frame) -> List[DetectedObject]`` is the uniform interface. Frames
  are HxWx{1,3,4} ndarrays; detections are returned in the ego-relative camera
  frame (+x forward, +y right, CARLA convention) via a ground-plane pinhole
  approximation — the fusion stage owns the world-frame transform.
* The registry always has a working default: if every checkpoint fails to
  load (or none exist), the geometric fallback detector serves predictions.
  ML failure must never take the perception pipe down.
"""
from __future__ import annotations

import glob
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fsd.core.logger import get
from fsd.core.types import DetectedObject, Vec3

log = get("ml.models")

CLASSES = ("vehicle", "pedestrian", "cyclist", "sign", "misc")
# DAVE-2-style input geometry — the architecture this stack imitates.
INPUT_H, INPUT_W = 66, 200
# Pinhole approximation constants for the ego camera.
CAM_HEIGHT_M = 1.4
CAM_HFOV_DEG = 90.0
HORIZON_FRAC = 0.45          # horizon line as a fraction of frame height


def focal_px(frame_w: int) -> float:
    return frame_w / (2.0 * math.tan(math.radians(CAM_HFOV_DEG) / 2.0))


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pure-numpy bilinear resize for HxW or HxWxC float/uint8 arrays."""
    in_h, in_w = img.shape[:2]
    if in_h == out_h and in_w == out_w:
        return img
    ys = np.linspace(0.0, in_h - 1, out_h)
    xs = np.linspace(0.0, in_w - 1, out_w)
    y0 = np.clip(ys.astype(np.int64), 0, in_h - 1)
    x0 = np.clip(xs.astype(np.int64), 0, in_w - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (ys - y0)[:, None].astype(np.float32)
    wx = (xs - x0)[None, :].astype(np.float32)

    img_f = img.astype(np.float32)
    if img_f.ndim == 2:
        Ia, Ib = img_f[y0[:, None], x0[None, :]], img_f[y1[:, None], x0[None, :]]
        Ic, Id = img_f[y0[:, None], x1[None, :]], img_f[y1[:, None], x1[None, :]]
        wy2, wx2 = wy, wx
    else:
        Ia, Ib = img_f[y0[:, None], x0[None, :]], img_f[y1[:, None], x0[None, :]]
        Ic, Id = img_f[y0[:, None], x1[None, :]], img_f[y1[:, None], x1[None, :]]
        wy2, wx2 = wy[:, :, None], wx[:, :, None]
    top = Ia * (1 - wx2) + Ic * wx2
    bot = Ib * (1 - wx2) + Id * wx2
    return top * (1 - wy2) + bot * wy2


def _as_hwc3(frame: np.ndarray) -> np.ndarray:
    """Normalize any sane frame layout to HxWx3 float32."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"frame shape {arr.shape} is not image-like")
    return arr.astype(np.float32)


def project_box(
    cls: str,
    score: float,
    box: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int],
    timestamp: Optional[float] = None,
    obj_id: int = 0,
) -> DetectedObject:
    """Ground-plane pinhole projection: image box -> ego-frame DetectedObject.

    Depth is estimated from the box's bottom edge resting on the ground plane;
    lateral offset from the box center's angular position. Honest about its
    limits: velocity is unknown (zeros) and extents are rough.
    """
    h, w = frame_shape
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    bottom = min(max(y2, 0.0), float(h - 1))
    horizon = h * HORIZON_FRAC
    f = focal_px(w)
    depth = CAM_HEIGHT_M * f / max(1.0, bottom - horizon)
    depth = float(min(depth, 200.0))
    lateral = (cx - w * 0.5) * depth / f
    width_m = max(0.3, (x2 - x1) * depth / f)
    height_m = max(0.3, (y2 - y1) * depth / f)
    kwargs = {"timestamp": timestamp} if timestamp is not None else {}
    return DetectedObject(
        obj_id=obj_id,
        cls=cls,
        position=Vec3(x=depth, y=lateral, z=0.0),
        velocity=Vec3(0.0, 0.0, 0.0),          # single frame -> motion unknown
        bbox_extent=Vec3(x=0.5 * depth * 0.05 + 0.9, y=width_m * 0.5, z=height_m * 0.5),
        confidence=float(min(max(score, 0.0), 1.0)),
        **kwargs,
    )


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Standard greedy NMS. boxes: (N,4) xyxy, scores: (N,). Returns kept idx."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def parse_detection_output(
    out: np.ndarray,
    input_size: Tuple[int, int] = (INPUT_H, INPUT_W),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-effort decode of common ONNX/torch detector heads.

    Supports: (N,6) or (1,N,6) [x1,y1,x2,y2,conf,cls]; (1,N,C>=7) YOLO-style
    [cx,cy,w,h,obj,cls...]. Returns (boxes_xyxy_pixels, scores, class_ids).
    Unknown layouts decode to empty — a misparsed tensor is worse than none.
    """
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    h, w = input_size
    if arr.shape[1] == 6:
        boxes, scores, cls = arr[:, :4], arr[:, 4], arr[:, 5].astype(np.int64)
    elif arr.shape[1] >= 7:
        cx, cy, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        obj = arr[:, 4]
        cls_idx = np.argmax(arr[:, 5:], axis=1)
        cls_conf = arr[np.arange(len(arr)), 5 + cls_idx]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        scores = obj * cls_conf
        cls = cls_idx.astype(np.int64)
    else:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    # Heuristic: max <= ~1.5 means normalized coords -> scale to input pixels.
    if boxes.size and float(np.abs(boxes).max()) <= 1.5:
        boxes = boxes * np.array([w, h, w, h], np.float32)
    return boxes.astype(np.float32), scores.astype(np.float32), cls


class BaseModel:
    """Uniform detector interface. ``kind`` describes the backend."""

    name = "base"
    kind = "abstract"
    supports_tensor = False
    input_size = (INPUT_H, INPUT_W)

    def predict(self, frame: np.ndarray) -> List[DetectedObject]:
        raise NotImplementedError

    def forward_batch(self, tensor: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Optional NN path: NCHW float32 batch -> per-frame (boxes, scores, cls)."""
        raise NotImplementedError(f"{self.name} has no tensor backend")

    def close(self) -> None:
        pass


class GeometricDetector(BaseModel):
    """Dependency-free fallback: connected-component blob detection.

    Thresholds edges/dark regions below the horizon on a coarse grid, extracts
    blobs, and projects them through the pinhole model. Produces low-confidence
    detections — honest about being a placeholder, but the pipeline never
    starves.
    """

    name = "geometric-fallback"
    kind = "geometric"
    GRID = (36, 64)
    MIN_BLOB_CELLS = 8
    MAX_DETECTIONS = 12
    EDGE_THRESH = 28.0
    DARK_THRESH = 90.0

    def predict(self, frame: np.ndarray) -> List[DetectedObject]:
        arr = _as_hwc3(frame)
        h, w = arr.shape[:2]
        gh, gw = self.GRID
        small = _resize_bilinear(arr, gh, gw)
        gray = small.mean(axis=-1)
        gy, gx = np.gradient(gray)
        edge = np.hypot(gx, gy)
        mask = (edge > self.EDGE_THRESH) | (gray < self.DARK_THRESH)
        horizon = int(gh * HORIZON_FRAC)
        mask[:horizon, :] = False          # only objects resting on the road plane
        labels, n = self._label(mask)

        raws: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            area = int(ys.size)
            if area < self.MIN_BLOB_CELLS:
                continue
            y0, y1c, x0, x1c = ys.min(), ys.max(), xs.min(), xs.max()
            box = (x0 / gw * w, y0 / gh * h, (x1c + 1) / gw * w, (y1c + 1) / gh * h)
            density = float(edge[labels == i].mean())
            score = min(0.5, 0.08 + 4.0 * area / (gh * gw) + density / 255.0)
            cell_h, cell_w = (y1c - y0 + 1), (x1c - x0 + 1)
            cls = "pedestrian" if cell_h > 1.4 * cell_w else "vehicle"
            raws.append((cls, score, box))

        raws.sort(key=lambda r: -r[1])
        return [
            project_box(cls, score, box, (h, w), obj_id=i + 1)
            for i, (cls, score, box) in enumerate(raws[: self.MAX_DETECTIONS])
        ]

    @staticmethod
    def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """4-connected component labeling on a small boolean grid (BFS)."""
        gh, gw = mask.shape
        labels = np.zeros((gh, gw), np.int32)
        current = 0
        seen = np.zeros_like(mask, dtype=bool)
        for sy, sx in zip(*np.nonzero(mask)):
            if seen[sy, sx]:
                continue
            current += 1
            stack = [(int(sy), int(sx))]
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                labels[y, x] = current
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < gh and 0 <= nx < gw and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        return labels, current


class ONNXModel(BaseModel):
    """ONNX detector. ``onnxruntime`` imported lazily on first use."""

    kind = "onnx"
    supports_tensor = True

    def __init__(self, path: str, input_size: Optional[Tuple[int, int]] = None) -> None:
        self.path = str(path)
        self.name = f"onnx:{os.path.basename(self.path)}"
        if input_size is not None:
            self.input_size = tuple(input_size)
        self._session = None
        self._input_name: Optional[str] = None

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort  # lazy — module must import without it

        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in avail]
        self._session = ort.InferenceSession(self.path, providers=providers or avail)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape  # expect (N, C, H, W); static dims refine input_size
        if len(shape) == 4 and all(isinstance(d, int) and d > 0 for d in shape[2:]):
            self.input_size = (int(shape[2]), int(shape[3]))
        log.info("loaded %s on providers=%s input=%s", self.name, providers, inp.shape)

    def forward_batch(self, tensor: np.ndarray):
        self._load()
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        outs = self._session.run(None, {self._input_name: tensor})
        arr = np.asarray(outs[0])
        if arr.ndim == 3:      # batched head -> decode per frame
            return [parse_detection_output(arr[i], self.input_size)
                    for i in range(arr.shape[0])]
        return [parse_detection_output(arr, self.input_size)]

    def predict(self, frame: np.ndarray) -> List[DetectedObject]:
        arr = _as_hwc3(frame)
        ih, iw = self.input_size
        tensor = _resize_bilinear(arr, ih, iw).transpose(2, 0, 1)[None] / 255.0
        per_frame = self.forward_batch(tensor.astype(np.float32))
        return self._to_objects(per_frame[0], arr.shape[:2])

    def _to_objects(self, parsed, frame_shape) -> List[DetectedObject]:
        boxes, scores, cls_ids = parsed
        dets = []
        for i in range(len(boxes)):
            cid = int(cls_ids[i]) if i < len(cls_ids) else 0
            cls = CLASSES[cid] if 0 <= cid < len(CLASSES) else "misc"
            dets.append(project_box(cls, float(scores[i]),
                                    tuple(float(v) for v in boxes[i]),
                                    frame_shape, obj_id=i + 1))
        return dets


class TorchModel(BaseModel):
    """Torch detector/policy checkpoint. ``torch`` imported lazily."""

    kind = "torch"
    supports_tensor = True

    def __init__(self, path: str, input_size: Optional[Tuple[int, int]] = None) -> None:
        self.path = str(path)
        self.name = f"torch:{os.path.basename(self.path)}"
        if input_size is not None:
            self.input_size = tuple(input_size)
        self._model = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            obj = torch.load(self.path, map_location=self._device, weights_only=False)
        except TypeError:  # older torch without weights_only
            obj = torch.load(self.path, map_location=self._device)
        model = getattr(obj, "model", obj)  # unwrap {"model": module} containers
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"{self.path} did not contain a torch.nn.Module "
                "(state_dicts need their architecture — export a full module)")
        model.to(self._device).eval()
        self._model = model
        log.info("loaded %s on %s", self.name, self._device)

    def forward_batch(self, tensor: np.ndarray):
        import torch  # lazy

        self._load()
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32))
            out = self._model(x.to(self._device))
            arr = out[0] if isinstance(out, (tuple, list)) else out
            arr = arr.detach().float().cpu().numpy()
        if arr.ndim == 3:
            return [parse_detection_output(arr[i], self.input_size)
                    for i in range(arr.shape[0])]
        return [parse_detection_output(arr, self.input_size)]

    def predict(self, frame: np.ndarray) -> List[DetectedObject]:
        arr = _as_hwc3(frame)
        ih, iw = self.input_size
        tensor = _resize_bilinear(arr, ih, iw).transpose(2, 0, 1)[None] / 255.0
        per_frame = self.forward_batch(tensor.astype(np.float32))
        boxes, scores, cls_ids = per_frame[0]
        dets = []
        for i in range(len(boxes)):
            cid = int(cls_ids[i]) if i < len(cls_ids) else 0
            cls = CLASSES[cid] if 0 <= cid < len(CLASSES) else "misc"
            dets.append(project_box(cls, float(scores[i]),
                                    tuple(float(v) for v in boxes[i]),
                                    arr.shape[:2], obj_id=i + 1))
        return dets


@dataclass
class ModelInfo:
    name: str
    kind: str
    path: Optional[str]
    supports_tensor: bool
    is_fallback: bool


class ModelRegistry:
    """Named collection of detectors with a guaranteed-working default.

    ``get()``/``predict()`` fall back to the geometric detector whenever the
    requested model is missing or fails, so callers never have to special-case
    ML outages.
    """

    CHECKPOINT_EXTS = (".onnx", ".pt", ".pth")

    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        self._models: Dict[str, BaseModel] = {}
        self._fallback = GeometricDetector()
        self._default: str = self._fallback.name
        self.register(self._fallback)

    # -------------------------------------------------------------- registry

    def register(self, model: BaseModel, default: bool = False) -> None:
        self._models[model.name] = model
        if default or self._default == self._fallback.name and model.kind != "geometric":
            if model is not self._fallback:
                self._default = model.name

    def load(self, path: str, name: Optional[str] = None, default: bool = True) -> Optional[BaseModel]:
        """Construct, validate (first forward pass), and register a checkpoint."""
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".onnx":
                model: BaseModel = ONNXModel(path)
            elif ext in (".pt", ".pth"):
                model = TorchModel(path)
            else:
                log.warning("unsupported checkpoint extension %r for %s", ext, path)
                return None
            if name:
                model.name = name
            # Smoke-test on a black frame — a bad checkpoint fails here, not
            # in the driving loop.
            probe = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
            model.predict(probe)
        except Exception as exc:
            log.warning("checkpoint %s failed to load (%r) — skipped", path, exc)
            return None
        self.register(model, default=default)
        log.info("registered model %s (kind=%s)", model.name, model.kind)
        return model

    def discover(self, directory: Optional[str] = None) -> List[str]:
        """Scan a directory for checkpoints; load each. Returns loaded names."""
        d = directory or self.checkpoint_dir
        loaded: List[str] = []
        for ext in self.CHECKPOINT_EXTS:
            for path in sorted(glob.glob(os.path.join(d, f"*{ext}"))):
                model = self.load(path)
                if model is not None:
                    loaded.append(model.name)
        if not loaded:
            log.info("no checkpoints in %s — geometric fallback active", d)
        return loaded

    def get(self, name: Optional[str] = None) -> BaseModel:
        if name is None:
            name = self._default
        model = self._models.get(name)
        if model is None:
            log.warning("model %r not registered — using geometric fallback", name)
            return self._fallback
        return model

    def predict(self, frame: np.ndarray, name: Optional[str] = None) -> List[DetectedObject]:
        model = self.get(name)
        try:
            return model.predict(frame)
        except Exception as exc:
            log.warning("model %s predict failed (%r) — geometric fallback", model.name, exc)
            if model is self._fallback:
                return []
            return self._fallback.predict(frame)

    def available(self) -> Dict[str, ModelInfo]:
        return {
            name: ModelInfo(
                name=m.name, kind=m.kind,
                path=getattr(m, "path", None),
                supports_tensor=m.supports_tensor,
                is_fallback=m is self._fallback,
            )
            for name, m in self._models.items()
        }

    @property
    def default_name(self) -> str:
        return self._default


_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Process-wide default registry (lazy-constructed, checkpoints discovered)."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _registry.discover()
    return _registry


def predict(frame: np.ndarray, model: Optional[str] = None) -> List[DetectedObject]:
    """Module-level convenience: default registry -> detections."""
    return get_registry().predict(frame, model)
