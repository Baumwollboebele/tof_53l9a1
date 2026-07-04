"""
Pluggable object detectors for the ToF viewer (tof_app.py).

A Detector takes the current frame channels and returns a list of Detection
(label, score, box, optional mask). Two backends are provided:

  * YoloDetector    - deep-learning object detection (Ultralytics YOLO, COCO
                      classes). Multi-class, with segmentation masks when a -seg
                      model is used. Requires `pip install ultralytics`.
  * HeuristicDetector - no-dependency fallback: the depth foreground blob from
                      segmenter.py, returned as a single "object" detection.

NOTE on expectations: YOLO is trained on RGB photos. A 54x42 ToF amplitude map
upscaled to the model input is far from that domain, so detections will be
sparse/unreliable. For good results, fine-tune a small detector on ToF data.
The plumbing here (detect -> boxes/masks -> overlay / show-only) is model-agnostic.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from segmenter import DepthBlobSegmenter

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None


@dataclass
class Detection:
    label: str
    score: float
    box: Tuple[float, float, float, float]  # x0,y0,x1,y1 in ToF-grid pixel coords
    mask: Optional[np.ndarray] = None       # (h,w) bool, or None (box only)


class Detector:
    name = "none"
    requires = ()  # pip packages needed

    def detect(self, distance, amplitude=None, ambient=None) -> List[Detection]:
        return []


def _normalize_u8(a):
    """Contrast-stretch a channel to uint8 using robust percentiles."""
    a = a.astype(np.float32)
    lo = np.percentile(a, 2.0)
    hi = np.percentile(a, 98.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _stretch(a_u8, size):
    """Upscale (h,w)->(size,size). Uses smooth cubic interpolation via OpenCV
    when available (better detection on the enlarged image), else numpy nearest."""
    if _cv2 is not None:
        return _cv2.resize(a_u8, (size, size), interpolation=_cv2.INTER_CUBIC)
    h, w = a_u8.shape
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return a_u8[np.ix_(ys, xs)]


def _resize_mask(m, w, h):
    """Nearest-neighbour resize of a (H,W) mask to (h,w) bool."""
    mh, mw = m.shape
    ys = (np.arange(h) * mh / h).astype(int)
    xs = (np.arange(w) * mw / w).astype(int)
    return m[np.ix_(ys, xs)] > 0.5


class HeuristicDetector(Detector):
    """Depth foreground blob as a single detection (no ML deps)."""

    name = "heuristic"

    def __init__(self, label="object", **kw):
        self.seg = DepthBlobSegmenter(**kw)
        self.label = label

    def detect(self, distance, amplitude=None, ambient=None):
        mask = self.seg.segment(distance, amplitude, ambient)
        if not mask.any():
            return []
        ys, xs = np.where(mask)
        box = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
        return [Detection(self.label, 1.0, box, mask)]


class YoloDetector(Detector):
    """
    Ultralytics YOLO object detection on the upscaled ToF amplitude image.

        model : weights (auto-downloaded). Use a '-seg' model for masks.
        imgsz : square size fed to the model.
        conf  : confidence threshold.
        source: 'amplitude' (default, most photo-like) or 'distance'.
        classes: optional list of COCO class ids to keep (None = all).
    """

    name = "yolo"
    requires = ("ultralytics",)

    def __init__(self, model="yolov8n-seg.pt", imgsz=320, conf=0.25,
                 source="amplitude", classes=None):
        self.model_path = model
        self.model = None  # loaded lazily on first detect()
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.source = source
        self.classes = classes

    def _ensure_model(self):
        if self.model is None:
            from ultralytics import YOLO  # heavy + may download weights
            self.model = YOLO(self.model_path)

    def detect(self, distance, amplitude=None, ambient=None):
        self._ensure_model()
        h, w = distance.shape
        src = amplitude if (self.source == "amplitude" and amplitude is not None) else distance
        img2d = _stretch(_normalize_u8(src), self.imgsz)          # (S,S) uint8
        img = np.repeat(img2d[:, :, None], 3, axis=2)             # fake 3-channel
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf,
                                 classes=self.classes, verbose=False)[0]
        S = self.imgsz
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        for i in range(len(xyxy)):
            x0, y0, x1, y1 = xyxy[i]
            box = (x0 * w / S, y0 * h / S, x1 * w / S, y1 * h / S)  # -> grid coords
            m = _resize_mask(masks[i], w, h) if masks is not None else None
            out.append(Detection(res.names[cls[i]], float(conf[i]), box, m))
        return out


def make_detector(kind="yolo", **kw):
    """Factory used by tof_app.py."""
    if kind in ("yolo", "ultralytics", "dl"):
        return YoloDetector(**kw)
    if kind in ("heuristic", "depth", "depth-blob"):
        return HeuristicDetector(**kw)
    if kind in ("none", "off"):
        return Detector()
    raise ValueError(f"unknown detector '{kind}'")
