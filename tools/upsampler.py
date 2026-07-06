"""
Deep-learning super-resolution for the low-res ToF channels.

Uses FSRCNN (Fast Super-Resolution CNN, Dong et al. 2016) through OpenCV's
dnn_superres module - the models are ~40 KB and fast enough to run on every
frame on CPU. Pretrained weights live in tools/models/FSRCNN_x{2,3,4}.pb and
are fetched from the FSRCNN_Tensorflow repo on first use if missing.

The ToF channels are float measurement arrays, not photos, so each frame is
contrast-normalized to uint8, super-resolved, then mapped back to its
original value range. Invalid pixels (value == 0) are median-filled before
inference so they don't bleed halos into their neighbours, and re-masked
(nearest-upscaled) afterwards so they stay invalid in the output.

Requires: opencv-contrib-python (plain opencv-python lacks dnn_superres).
"""

import os
import urllib.request

import numpy as np
import cv2

_MODEL_URL = ("https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/"
              "master/models/FSRCNN_x{s}.pb")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class DeepUpsampler:
    """FSRCNN super-resolution of single-channel float frames. scale: 2, 3 or 4."""

    def __init__(self, scale=2):
        if scale not in (2, 3, 4):
            raise ValueError("FSRCNN supports scale 2, 3 or 4")
        if not hasattr(cv2, "dnn_superres"):
            raise RuntimeError("cv2.dnn_superres missing: "
                               "pip install opencv-contrib-python")
        self.scale = int(scale)
        path = os.path.join(MODELS_DIR, f"FSRCNN_x{self.scale}.pb")
        if not os.path.exists(path):
            os.makedirs(MODELS_DIR, exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL.format(s=self.scale), path)
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(path)
        self.sr.setModel("fsrcnn", self.scale)

    def upsample(self, arr):
        """(h,w) float -> (h*s,w*s) float32; zero (invalid) pixels stay zero."""
        a = np.asarray(arr, dtype=np.float32)
        h, w = a.shape
        big_wh = (w * self.scale, h * self.scale)
        valid = a > 0
        if not valid.any():
            return cv2.resize(a, big_wh, interpolation=cv2.INTER_NEAREST)
        lo, hi = float(a[valid].min()), float(a[valid].max())
        if hi <= lo:
            hi = lo + 1.0
        filled = a.copy()
        filled[~valid] = float(np.median(a[valid]))
        g = ((filled - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)
        big = self.sr.upsample(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))[:, :, 0]
        out = big.astype(np.float32) * ((hi - lo) / 255.0) + lo
        if not valid.all():
            inv = cv2.resize((~valid).astype(np.uint8), big_wh,
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            out[inv] = 0.0
        return out
