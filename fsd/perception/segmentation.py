"""Lightweight semantic segmentation interface.

CARLA's semantic camera emits per-pixel tags (Cityscapes palette). This module
maps raw tag ids — or palette-colored RGB frames — onto a small set of
drivable-scene classes and produces boolean masks / BEV projections without
any heavy dependencies.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np

from fsd.core.logger import get

log = get("perception.segment")


class SegClass(IntEnum):
    BACKGROUND = 0
    ROAD = 1
    LANE = 2          # lane markings
    VEHICLE = 3
    PEDESTRIAN = 4
    OTHER = 5


# CARLA semantic tag -> SegClass (carla.CityscapesLabel ids)
_TAG_TO_CLASS = {
    0: SegClass.BACKGROUND,      # Unlabeled
    4: SegClass.PEDESTRIAN,      # Pedestrian
    6: SegClass.LANE,            # RoadLine
    7: SegClass.ROAD,            # Road
    10: SegClass.VEHICLE,        # Vehicles
    13: SegClass.BACKGROUND,     # Sky
    20: SegClass.VEHICLE,        # Dynamic (riders etc. count as obstacles)
}
_LUT = np.full(256, SegClass.OTHER, dtype=np.int8)
for _tag, _cls in _TAG_TO_CLASS.items():
    _LUT[_tag] = _cls

# CARLA CityScapesPalette RGB for the tags we care about: tag -> (R,G,B)
_PALETTE = {
    0: (0, 0, 0), 4: (220, 20, 60), 6: (157, 234, 50), 7: (128, 64, 128),
    10: (0, 0, 142), 13: (70, 130, 180), 20: (0, 0, 60),
    1: (70, 70, 70), 2: (190, 153, 153), 3: (250, 170, 160),
    5: (153, 153, 153), 8: (244, 35, 232), 9: (107, 142, 35),
    11: (102, 102, 156), 12: (220, 220, 0), 14: (80, 90, 110),
    15: (81, 0, 81), 16: (230, 150, 140), 17: (180, 165, 180),
    18: (250, 100, 0), 19: (150, 100, 100), 21: (61, 230, 250),
    22: (145, 170, 100),
}


class SemanticSegmenter:
    """Classifies semantic-camera frames into ``SegClass`` label maps."""

    def __init__(self, color_tol: float = 40.0):
        self.color_tol = color_tol
        tags = sorted(_PALETTE)
        self._pal_rgb = np.asarray([_PALETTE[t] for t in tags], np.float32)
        self._pal_cls = np.asarray([_LUT[t] for t in tags], np.int8)

    # ------------------------------------------------------------------ API
    def segment(self, frame, input_mode: str = "auto") -> np.ndarray:
        """Return an (H, W) int8 label map of ``SegClass`` values.

        input_mode:
            'auto'  - 2-D / single-channel -> raw tags; 3-channel -> RGB colors
            'tags'  - pixel values are CARLA semantic tag ids
            'rgb'   - palette-colored image (converted semantic frame)
        """
        img = np.asarray(frame)
        if input_mode == "auto":
            input_mode = "tags" if img.ndim == 2 or \
                (img.ndim == 3 and img.shape[2] == 1) else "rgb"
        if input_mode == "tags":
            tags = img if img.ndim == 2 else img[..., 0]
            return _LUT[np.clip(tags, 0, 255).astype(np.uint8)]
        if input_mode == "rgb":
            return self._segment_rgb(img[..., :3].astype(np.float32))
        raise ValueError(f"unknown input_mode {input_mode!r}")

    def mask(self, labels: np.ndarray, *classes: SegClass) -> np.ndarray:
        out = np.zeros(labels.shape, bool)
        for c in classes:
            out |= labels == np.int8(c)
        return out

    def road_mask(self, labels: np.ndarray) -> np.ndarray:
        """Drivable surface incl. lane markings."""
        return self.mask(labels, SegClass.ROAD, SegClass.LANE)

    def obstacle_mask(self, labels: np.ndarray) -> np.ndarray:
        return self.mask(labels, SegClass.VEHICLE, SegClass.PEDESTRIAN)

    def class_histogram(self, labels: np.ndarray) -> Dict[str, int]:
        counts = np.bincount(labels.ravel().astype(np.uint8),
                             minlength=len(SegClass))
        return {c.name.lower(): int(counts[int(c)]) for c in SegClass}

    # ------------------------------------------------------------- BEV util
    @staticmethod
    def bev_warp(img: np.ndarray, M_dst_to_src: np.ndarray,
                 out_shape: Tuple[int, int]) -> np.ndarray:
        """Nearest-neighbor perspective warp; M maps dst px -> src px.

        Same convention as ``lane_detector.warp``; duplicated here so this
        module stays standalone (pass the inverse of the img->BEV homography).
        """
        H, W = out_shape
        cols, rows = np.meshgrid(np.arange(W), np.arange(H))
        p = np.stack([cols.ravel(), rows.ravel(), np.ones(H * W)])
        q = M_dst_to_src @ p
        w = q[2]
        ok = np.abs(w) > 1e-6
        sx = np.zeros(H * W)
        sy = np.zeros(H * W)
        np.divide(q[0], w, out=sx, where=ok)
        np.divide(q[1], w, out=sy, where=ok)
        sx = np.round(sx).astype(np.int64)
        sy = np.round(sy).astype(np.int64)
        ok &= (sx >= 0) & (sx < img.shape[1]) & (sy >= 0) & (sy < img.shape[0])
        out = np.zeros(H * W, dtype=img.dtype)
        out[ok] = img[sy[ok], sx[ok]]
        return out.reshape(out_shape)

    def bev_labels(self, frame, M_dst_to_src: np.ndarray,
                   out_shape: Tuple[int, int]) -> np.ndarray:
        """Segment then warp a frame straight into a BEV label map."""
        return self.bev_warp(self.segment(frame), M_dst_to_src, out_shape)

    # --------------------------------------------------------------- helpers
    def _segment_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Assign each pixel to the nearest palette color -> its class."""
        flat = rgb.reshape(-1, 3)
        labels = np.zeros(len(flat), np.int8)
        best_d = np.full(len(flat), np.inf)
        for color, cls in zip(self._pal_rgb, self._pal_cls):
            d = np.abs(flat - color).sum(1)        # L1 distance, cheap
            better = d < best_d
            best_d[better] = d[better]
            labels[better] = cls
        labels[best_d > 3.0 * self.color_tol] = SegClass.OTHER
        return labels.reshape(rgb.shape[:2])
