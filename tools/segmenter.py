"""
Pluggable segmenters for the ToF viewer (tof_app.py).

A Segmenter takes the current frame channels and returns a boolean (h, w) mask
marking "person" pixels. The default DepthBlobSegmenter needs no ML deps and
works on the depth channel. To use a deep-learning model instead, subclass
Segmenter (see the OnnxSegmenter example at the bottom) and pass an instance to
tof_app via --segmenter, or edit make_segmenter().
"""

import numpy as np

try:
    from scipy import ndimage as _ndi
    _HAVE_SCIPY = True
except ImportError:  # scipy is optional; a numpy fallback is used instead
    _HAVE_SCIPY = False

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None


# --------------------------------------------------------------------------
# Sobel edge helpers (depth gradients = object boundaries on a ToF frame)
# --------------------------------------------------------------------------
def sobel_magnitude(img):
    """3x3 Sobel gradient magnitude of a 2-D array (float32)."""
    a = img.astype(np.float32)
    if _cv2 is not None:
        gx = _cv2.Sobel(a, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(a, _cv2.CV_32F, 0, 1, ksize=3)
    else:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        gx = _conv3(a, kx)
        gy = _conv3(a, kx.T)
    return np.hypot(gx, gy)


def _conv3(a, k):
    ap = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * ap[i:i + a.shape[0], j:j + a.shape[1]]
    return out


def binarize_otsu(x):
    """Binarize a float array with Otsu's threshold (percentile fallback)."""
    xn = x - x.min()
    mx = xn.max() or 1.0
    xn = (xn * (255.0 / mx)).astype(np.uint8)
    if _cv2 is not None:
        _, b = _cv2.threshold(xn, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        return b > 0
    return xn > np.percentile(xn, 80)


def sobel_binary(img):
    """Binary edge map: Sobel magnitude then Otsu threshold."""
    return binarize_otsu(sobel_magnitude(img))


def _dilate(m, it=1):
    if _cv2 is not None:
        return _cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=it) > 0
    if _HAVE_SCIPY:
        return _ndi.binary_dilation(m, iterations=it)
    return _majority3(m) | m


def _fill_holes(m):
    if _HAVE_SCIPY:
        return _ndi.binary_fill_holes(m)
    if _cv2 is not None:
        mm = (m.astype(np.uint8)) * 255
        ff = mm.copy()
        hh, ww = mm.shape
        _cv2.floodFill(ff, np.zeros((hh + 2, ww + 2), np.uint8), (0, 0), 255)
        return (mm | _cv2.bitwise_not(ff)) > 0
    return m


class Segmenter:
    """Base class. Return a boolean (h, w) 'person' mask."""

    name = "none"

    def segment(self, distance, amplitude=None, ambient=None):
        """distance/amplitude/ambient: (h, w) uint16 arrays; distance in mm,
        0 = invalid. Return a boolean (h, w) mask."""
        return np.zeros(distance.shape, dtype=bool)


class DepthBlobSegmenter(Segmenter):
    """
    Foreground-person heuristic on the depth channel.

    A person in front of the sensor is the nearest sizeable object. We take the
    valid pixels whose distance is within `band_mm` of the nearest surface,
    denoise the mask, and keep the largest connected blob if it is big enough.

    Tunables:
        min_mm, max_mm : valid distance window (mm)
        band_mm        : depth thickness kept behind the nearest surface (mm)
        min_frac       : min blob size as a fraction of the frame to count as a person
        near_pct       : percentile used as the robust "nearest" distance
    """

    name = "depth-blob"

    def __init__(self, min_mm=150, max_mm=4000, band_mm=700, min_frac=0.03,
                 max_frac=0.70, near_pct=5.0, edge_refine=False):
        self.min_mm = min_mm
        self.max_mm = max_mm
        self.band_mm = band_mm
        self.min_frac = min_frac    # too small -> not a person
        self.max_frac = max_frac    # too large (fills frame) -> a wall, not a person
        self.near_pct = near_pct
        self.edge_refine = edge_refine  # cut the mask at depth (Sobel) edges

    def segment(self, distance, amplitude=None, ambient=None):
        empty = np.zeros(distance.shape, dtype=bool)
        d = distance.astype(np.float32)
        valid = (d > self.min_mm) & (d < self.max_mm)
        nvalid = int(valid.sum())
        if nvalid == 0:
            return empty
        near = float(np.percentile(d[valid], self.near_pct))
        fg = valid & (d <= near + self.band_mm)
        # A person is a foreground object: if the near band already covers most
        # of the valid scene it is a flat surface (e.g. a wall), not a person.
        if fg.sum() > self.max_frac * nvalid:
            return empty
        if self.edge_refine:
            # Depth discontinuities (Sobel edges) mark object boundaries. Cutting
            # the foreground at strong edges removes boundary/flying-pixel halo
            # and separates objects whose depth bands merge; holes are filled back.
            flat = np.where(valid, d, near)  # avoid huge false edges at invalid px
            edges = _dilate(sobel_binary(flat), 1)
            fg = fg & ~edges
            fg = self._clean(fg)
            fg = _fill_holes(fg)
        else:
            fg = self._clean(fg)
        n = fg.sum()
        if n < self.min_frac * fg.size or n > self.max_frac * fg.size:
            return empty
        return fg

    def _clean(self, fg):
        if _HAVE_SCIPY:
            fg = _ndi.binary_opening(fg, iterations=1)
            fg = _ndi.binary_closing(fg, iterations=1)
            lbl, n = _ndi.label(fg)
            if n <= 1:
                return fg
            sizes = _ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
            keep = 1 + int(np.argmax(sizes))
            return lbl == keep
        return _majority3(fg)  # numpy-only denoise (no connected-components)


def _majority3(mask):
    """3x3 majority filter (fallback denoise when scipy is unavailable)."""
    m = mask.astype(np.uint8)
    s = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s += np.roll(np.roll(m, dy, 0), dx, 1)
    return s >= 5


def make_segmenter(kind="depth-blob"):
    """Factory used by tof_app.py."""
    if kind in ("depth-blob", "depth", "heuristic"):
        return DepthBlobSegmenter()
    if kind in ("none", "off"):
        return Segmenter()
    raise ValueError(f"unknown segmenter '{kind}'")


# ---------------------------------------------------------------------------
# Example deep-learning backend. Uncomment and adapt to your model, then use
#   make_segmenter -> return OnnxSegmenter("model.onnx")
# NOTE: RGB-trained person-seg models generally do NOT work well on a 54x42 ToF
# amplitude map; train/fine-tune on ToF data for good results.
# ---------------------------------------------------------------------------
# class OnnxSegmenter(Segmenter):
#     name = "onnx"
#
#     def __init__(self, path, size=256, thresh=0.5):
#         import onnxruntime as ort
#         self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
#         self.inp = self.sess.get_inputs()[0].name
#         self.size = size
#         self.thresh = thresh
#
#     def segment(self, distance, amplitude=None, ambient=None):
#         src = amplitude if amplitude is not None else distance
#         img = src.astype(np.float32)
#         img /= (img.max() or 1.0)
#         # nearest-neighbour upscale to the model input, fake 3-channel
#         h, w = distance.shape
#         ys = (np.arange(self.size) * h / self.size).astype(int)
#         xs = (np.arange(self.size) * w / self.size).astype(int)
#         big = img[np.ix_(ys, xs)]
#         x = np.repeat(big[None, None], 3, axis=1).astype(np.float32)  # NCHW
#         out = self.sess.run(None, {self.inp: x})[0]
#         prob = out[0, 0] if out.ndim == 4 else out.squeeze()
#         small = prob[np.ix_((np.arange(h) * self.size / h).astype(int),
#                             (np.arange(w) * self.size / w).astype(int))]
#         return small > self.thresh
