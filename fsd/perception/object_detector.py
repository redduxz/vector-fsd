"""Object detection front-end.

Primary path converts CARLA actor ground truth into ``DetectedObject``s.
A pluggable vision backend ('yolo' via ultralytics, 'onnx' via onnxruntime)
runs when a camera image is supplied; 2-D boxes are projected onto the
ground plane with a pinhole model to yield metric positions.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fsd.core.logger import get
from fsd.core.types import DetectedObject, Vec3

try:
    import carla
except ImportError:
    carla = None

log = get("perception.objects")

# COCO-style label -> stack class
_CLS_MAP = {
    "person": "pedestrian", "bicycle": "cyclist", "motorcycle": "cyclist",
    "car": "vehicle", "bus": "vehicle", "truck": "vehicle",
    "traffic light": "sign", "stop sign": "sign",
}
_EXTENT = {   # half-extents (x fwd, y right, z up), meters
    "vehicle": Vec3(2.2, 0.9, 0.8), "pedestrian": Vec3(0.3, 0.3, 0.9),
    "cyclist": Vec3(0.9, 0.35, 0.85), "sign": Vec3(0.2, 0.2, 0.6),
    "misc": Vec3(0.5, 0.5, 0.8),
}
_BIKE_HINTS = ("bicycle", "motorcycle", "harley", "yamaha", "kawasaki",
               "vespa", "crossbike", "bike", "dirtbike")


def _ego_xy_yaw(ego) -> Tuple[float, float, float]:
    """(x, y, yaw_rad) from carla.Actor/Transform, VehicleState, dict, or None."""
    if ego is None:
        return 0.0, 0.0, 0.0
    if hasattr(ego, "get_transform"):
        t = ego.get_transform()
        return t.location.x, t.location.y, math.radians(t.rotation.yaw)
    if hasattr(ego, "rotation") and hasattr(ego, "location"):
        return ego.location.x, ego.location.y, math.radians(ego.rotation.yaw)
    if all(hasattr(ego, k) for k in ("x", "y", "yaw")):
        return float(ego.x), float(ego.y), float(ego.yaw)
    if isinstance(ego, dict):
        return float(ego["x"]), float(ego["y"]), float(ego.get("yaw", 0.0))
    return 0.0, 0.0, 0.0


class ObjectDetector:
    """Detects objects from CARLA ground truth and/or a vision backend."""

    BACKENDS = ("none", "yolo", "onnx")

    def __init__(self, backend: str = "none", model_path: Optional[str] = None,
                 conf_threshold: float = 0.35, max_range_m: float = 120.0,
                 camera_height_m: float = 1.4, hfov_deg: float = 90.0):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")
        self.backend = backend
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.max_range_m = max_range_m
        self.cam_h = camera_height_m
        self.tan_hfov = math.tan(math.radians(hfov_deg) * 0.5)
        self._model = None          # lazily created backend handle
        self._warned = False
        self._vision_id = -1        # negative ids distinguish vision dets

    # ------------------------------------------------------------------ API
    def detect(self, image=None, world=None, ego=None) -> List[DetectedObject]:
        objs: List[DetectedObject] = []
        if world is not None:
            objs.extend(self._from_carla(world, ego))
        if image is not None and self.backend != "none":
            objs.extend(self._from_vision(np.asarray(image), ego))
        return objs

    # ----------------------------------------------------------- CARLA path
    def _from_carla(self, world, ego) -> List[DetectedObject]:
        ex, ey, _ = _ego_xy_yaw(ego)
        ego_id = getattr(ego, "id", None)
        if ego_id is None and isinstance(ego, dict):
            ego_id = ego.get("id")
        out = []
        for actor in world.get_actors():
            cls = self._classify(getattr(actor, "type_id", "") or "")
            if cls is None or (ego_id is not None and actor.id == ego_id):
                continue
            try:
                t = actor.get_transform()
                v = actor.get_velocity()
            except Exception:
                continue
            dx, dy = t.location.x - ex, t.location.y - ey
            d2 = dx * dx + dy * dy
            if d2 > self.max_range_m ** 2:
                continue
            if d2 < 1.5 * 1.5:            # the ego actor itself (no id given)
                continue
            bb = getattr(actor, "bounding_box", None)
            extent = (Vec3(bb.extent.x, bb.extent.y, bb.extent.z)
                      if bb is not None else _EXTENT[cls])
            out.append(DetectedObject(
                obj_id=int(actor.id), cls=cls,
                position=Vec3(t.location.x, t.location.y, t.location.z),
                velocity=Vec3(v.x, v.y, v.z),
                bbox_extent=extent, confidence=1.0))
        return out

    @staticmethod
    def _classify(type_id: str) -> Optional[str]:
        if type_id.startswith("walker") or "pedestrian" in type_id:
            return "pedestrian"
        if type_id.startswith("vehicle"):
            return "cyclist" if any(h in type_id for h in _BIKE_HINTS) else "vehicle"
        if type_id.startswith("traffic"):
            return "sign"
        if type_id.startswith("static"):
            return "misc"
        return None

    # ----------------------------------------------------------- vision path
    def _from_vision(self, img: np.ndarray, ego) -> List[DetectedObject]:
        ex, ey, eyaw = _ego_xy_yaw(ego)
        cos_y, sin_y = math.cos(eyaw), math.sin(eyaw)
        out = []
        for name, conf, (x1, y1, x2, y2) in self._backend_infer(img):
            if conf < self.conf_threshold:
                continue
            cls = _CLS_MAP.get(name, "misc")
            g = self._ground_project(0.5 * (x1 + x2), y2, img.shape[1], img.shape[0])
            if g is None:
                continue
            fwd, right = g
            if fwd > self.max_range_m:
                continue
            out.append(DetectedObject(
                obj_id=self._vision_id, cls=cls,
                position=Vec3(ex + cos_y * fwd - sin_y * right,
                              ey + sin_y * fwd + cos_y * right, 0.0),
                velocity=Vec3(0.0, 0.0, 0.0),
                bbox_extent=_EXTENT[cls], confidence=float(conf) * 0.85))
            self._vision_id -= 1
        return out

    def _ground_project(self, px: float, py_bottom: float,
                        W: int, H: int) -> Optional[Tuple[float, float]]:
        """Pinhole ground-plane projection; returns (fwd, right) meters in ego frame."""
        fx = W / (2.0 * self.tan_hfov)
        fy = fx
        cx, cy = 0.5 * W, 0.5 * H
        dx = (px - cx) / fx                       # right, in units of fwd
        dy = (py_bottom - cy) / fy                # down, in units of fwd
        if dy <= 0.02:                            # ray above horizon
            return None
        fwd = self.cam_h / dy
        return fwd, dx * fwd

    # ------------------------------------------------------------- backends
    def _backend_infer(self, img: np.ndarray) -> List[Tuple[str, float, Tuple]]:
        """Returns [(cls_name, conf, (x1,y1,x2,y2)), ...]."""
        try:
            if self.backend == "yolo":
                return self._infer_yolo(img)
            if self.backend == "onnx":
                return self._infer_onnx(img)
        except Exception as e:                    # backend failure is non-fatal
            if not self._warned:
                log.warning("vision backend '%s' unavailable: %s", self.backend, e)
                self._warned = True
        return []

    def _infer_yolo(self, img):
        from ultralytics import YOLO              # lazy optional dep
        if self._model is None:
            self._model = YOLO(self.model_path or "yolov8n.pt")
        res = self._model.predict(img, verbose=False)[0]
        names = res.names
        return [(names[int(c)], float(cf), tuple(map(float, b)))
                for b, cf, c in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.conf.cpu().numpy(),
                                    res.boxes.cls.cpu().numpy())]

    def _infer_onnx(self, img):
        import onnxruntime as ort                 # lazy optional dep
        if self._model is None:
            if not self.model_path:
                raise RuntimeError("backend='onnx' requires model_path")
            self._model = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"])
        inp = self._model.get_inputs()[0]
        S = inp.shape[2] if len(inp.shape) > 2 and isinstance(inp.shape[2], int) else 640
        small = self._resize(img, S, S).astype(np.float32) / 255.0
        blob = small.transpose(2, 0, 1)[None]     # NCHW
        pred = self._model.run(None, {inp.name: blob})[0]
        pred = pred[0].T if pred.shape[1] < pred.shape[-1] else pred[0]
        boxes, confs, names = [], [], []
        xy = pred[:, :4]
        scores = pred[:, 4:]
        cid = scores.argmax(1)
        cf = scores[np.arange(len(scores)), cid]
        sx, sy = img.shape[1] / S, img.shape[0] / S
        for (cx, cy, w, h), c, k in zip(xy, cf, cid):
            if c < self.conf_threshold:
                continue
            boxes.append([(cx - w / 2) * sx, (cy - h / 2) * sy,
                          (cx + w / 2) * sx, (cy + h / 2) * sy])
            confs.append(float(c))
            names.append(str(k))
        keep = self._nms(boxes, confs, 0.45)
        return [(names[i], confs[i], tuple(boxes[i])) for i in keep]

    @staticmethod
    def _resize(img: np.ndarray, W: int, H: int) -> np.ndarray:
        """Nearest-neighbor resize (keeps the module numpy-only)."""
        ys = (np.linspace(0, img.shape[0] - 1, H)).astype(np.int64)
        xs = (np.linspace(0, img.shape[1] - 1, W)).astype(np.int64)
        return img[np.ix_(ys, xs)]

    @staticmethod
    def _nms(boxes: Sequence, confs: Sequence, iou_thr: float) -> List[int]:
        order = np.argsort(np.asarray(confs))[::-1]
        keep = []
        while len(order):
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            rest = order[1:]
            ax1, ay1, ax2, ay2 = boxes[i]
            b = np.asarray([boxes[j] for j in rest])
            ix1 = np.maximum(ax1, b[:, 0]); iy1 = np.maximum(ay1, b[:, 1])
            ix2 = np.minimum(ax2, b[:, 2]); iy2 = np.minimum(ay2, b[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            order = rest[inter / np.maximum(area_a + area_b - inter, 1e-9) < iou_thr]
        return keep
