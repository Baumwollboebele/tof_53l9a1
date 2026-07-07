"""
Shared 3-D geometry for the VL53L9CX tools: sensor optics, depth filters and
point-cloud helpers used by the live 3-D view (tof_gui.py) and the 3-D map
recorder (mapper.py).

The projection model is a numpy port of ST's vl53l9-transform-c
(radial_to_perp.c): a single-focal camera derived from the SPAD geometry
(EFL 2428.16 um over 10.17 um SPADs on a 216x168 array), a radial distortion
polynomial and an emitter/receiver parallax correction whose optical centre
shifts with depth. Passing an explicit hfov_deg falls back to the old
aspect-derived pinhole (for non-default lenses / --fov overrides).

Camera frame convention: X right, Y down, Z forward (row index grows down),
distances in metres unless a name says _mm. `to_display()` converts to the
Z-up frame the GL view expects.
"""

import math

import numpy as np

# ST default calibration (vl53l9_algo_radial_to_perp_init_default_params)
EFL_UM = 2428.16
SPAD_UM = 10.17
SPADS_X, SPADS_Y = 216, 168
ALPHA, BETA, GAMMA, KAPPA = -0.00015, 0.0, 0.0, 0.0
BASELINE_MM = 7.166            # emitter->receiver parallax numerator
PARALLAX_LIMIT_MM = 50.0       # clamp: below this depth the shift saturates


class SensorModel:
    """Cached per-(w, h, binning) projection model. Use get_model()."""

    def __init__(self, w, h, binning=2, hfov_deg=None):
        self.w, self.h = int(w), int(h)
        self.cx, self.cy = (w - 1) / 2.0, (h - 1) / 2.0
        cc, rr = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        self.cc = cc
        self._dx0 = cc - self.cx
        self._dy = rr - self.cy
        self.pinhole = hfov_deg is not None
        if self.pinhole:
            hfov = math.radians(hfov_deg)
            vfov = hfov * h / w
            self.fx = (w / 2.0) / math.tan(hfov / 2.0)
            self.fy = (h / 2.0) / math.tan(vfov / 2.0)
            self._tx = self._dx0 / self.fx
            self._ty = self._dy / self.fy
            self._inv_norm0 = 1.0 / np.sqrt(self._tx ** 2 + self._ty ** 2 + 1.0)
        else:
            mpix = SPADS_X / float(w)          # SPADs per output pixel
            self.focal = EFL_UM / (SPAD_UM * mpix)
            b = float(binning)
            self._bd = 1.0 if b == 2.0 else abs(math.sqrt(2.0) * (b * b / 4.0 - b / 2.0))
            self._pk = EFL_UM * BASELINE_MM / (SPAD_UM * mpix)  # px shift = pk / z_mm
            rsq0 = self._dx0 ** 2 + self._dy ** 2
            self._ud0 = self._distort(rsq0)
            self._inv_norm0 = 1.0 / np.sqrt(
                1.0 + rsq0 / (self._ud0 ** 2 * self.focal ** 2))
            self._tx = self._dx0 / (self._ud0 * self.focal)
            self._ty = self._dy / (self._ud0 * self.focal)

    def _distort(self, rsq):
        return 1.0 + self._bd * (ALPHA * rsq + BETA * rsq ** 2
                                 + GAMMA * rsq ** 3 + KAPPA)

    def tangents(self):
        """(tan_x, tan_y) grids: per-pixel ray direction is (tx, ty, 1)."""
        return self._tx, self._ty

    def z_from_radial(self, dist_mm, parallax=True):
        """Radial-distance image (mm, 0 = invalid) -> perpendicular Z (m)."""
        return self.unproject(dist_mm, parallax)[2]

    def unproject(self, dist_mm, parallax=True):
        """Radial mm image -> (X, Y, Z) metre images (camera frame, 0-Z invalid)."""
        d = np.asarray(dist_mm, np.float32)
        if self.pinhole or not parallax:
            z_mm = d * self._inv_norm0
            tx, ty = self._tx, self._ty
        else:
            z0 = d * self._inv_norm0                       # first pass, no shift
            dx = self.cc - (self.cx - self._pk / np.maximum(PARALLAX_LIMIT_MM, z0))
            rsq = dx ** 2 + self._dy ** 2
            ud = self._distort(rsq)
            z_mm = d / np.sqrt(1.0 + rsq / (ud ** 2 * self.focal ** 2))
            s = 1.0 / (ud * self.focal)
            tx, ty = dx * s, self._dy * s
        invalid = d <= 0
        z_mm = z_mm.copy()
        z_mm[invalid] = 0.0
        z = z_mm / 1000.0
        return (tx * z).astype(np.float32), (ty * z).astype(np.float32), \
            z.astype(np.float32)

    def project(self, pts_m, parallax=True):
        """Camera-frame points (N,3, m) -> (u, v) float pixel coords.
        Points at/behind the camera get non-finite coords (filter with z>0)."""
        p = np.asarray(pts_m, np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = p[:, 0] / p[:, 2]
            ty = p[:, 1] / p[:, 2]
        if self.pinhole:
            return tx * self.fx + self.cx, ty * self.fy + self.cy
        cx = self.cx
        if parallax:
            z_mm = p[:, 2] * 1000.0
            cx = cx - self._pk / np.maximum(PARALLAX_LIMIT_MM, z_mm)
        # r_px = rho * ud(r_px^2) with rho the undistorted radius; damped
        # fixed point (the plain iteration oscillates at the image corners)
        rho = np.sqrt(tx ** 2 + ty ** 2) * self.focal
        r = rho.copy()
        for _ in range(8):
            r = 0.5 * (r + rho * self._distort(r ** 2))
        ud = self._distort(r ** 2)
        return tx * self.focal * ud + cx, ty * self.focal * ud + self.cy


_MODELS = {}


def get_model(w, h, binning=2, hfov_deg=None):
    key = (int(w), int(h), int(binning), None if hfov_deg is None else round(hfov_deg, 3))
    m = _MODELS.get(key)
    if m is None:
        m = _MODELS[key] = SensorModel(w, h, binning, hfov_deg)
    return m


# ---- depth-image filters ------------------------------------------------------
def drop_flying_pixels(z, jump):
    """Zero out pixels whose depth differs from any valid 4-neighbour by more
    than `jump` metres (classic ToF edge artifacts hanging in mid-air)."""
    bad = np.zeros(z.shape, bool)
    v = z > 0
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        zn = np.roll(z, sh, axis=ax)
        vn = np.roll(v, sh, axis=ax)
        d = np.abs(z - zn)
        bad |= v & vn & (d > jump)
    out = z.copy()
    out[bad] = 0.0
    return out


def amplitude_gate(z, amp, frac):
    """Drop pixels whose amplitude is below `frac` of the frame's p99.
    Returns (gated z copy, gray u8 image scaled to the p99)."""
    a = np.asarray(amp, np.float32)
    hi = float(np.percentile(a[a > 0], 99)) if (a > 0).any() else 1.0
    out = z.copy()
    out[a < frac * hi] = 0.0
    gray = np.clip(a * (255.0 / max(hi, 1.0)), 0, 255).astype(np.uint8)
    return out, gray


# ---- point clouds -------------------------------------------------------------
def to_display(pts):
    """Camera/world frame (X right, Y down, Z fwd) -> GL Z-up display frame
    (X right, Y fwd/depth, Z up) so the grid reads as a floor."""
    if len(pts) == 0:
        return pts
    return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)


def live_pointcloud(dist_mm, amp=None, binning=2, hfov_deg=None, dmax_mm=4000,
                    amp_frac=0.0, edge_jump=0.0, color="depth", cmap_id=2,
                    parallax=True):
    """One radial frame -> (pts Nx3 m in display frame, rgba Nx4 0..1).

    color: "depth" (colormap over distance), "amplitude", or "solid".
    amp_frac / edge_jump enable the confidence gate and flying-pixel filter."""
    model = get_model(*dist_mm.shape[::-1], binning=binning, hfov_deg=hfov_deg)
    x, y, z = model.unproject(dist_mm, parallax)
    if edge_jump > 0:
        z = drop_flying_pixels(z, edge_jump)
    if amp_frac > 0 and amp is not None:
        z, _ = amplitude_gate(z, amp, amp_frac)
    valid = (z > 0) & (z <= dmax_mm / 1000.0)
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1)

    d = np.asarray(dist_mm, np.float32)[valid]
    if color == "amplitude" and amp is not None:
        a = np.asarray(amp, np.float32)[valid]
        hi = max(float(np.percentile(a, 99)) if len(a) else 1.0, 1.0)
        scalar = np.clip(a / hi, 0, 1)
    elif color == "solid":
        scalar = None
    else:
        scalar = np.clip(d / max(dmax_mm, 1), 0, 1)
    if scalar is None:
        rgba = np.full((len(pts), 4), (0.75, 0.78, 0.82, 1.0), np.float32)
    else:
        rgba = _colorize(scalar, cmap_id)
    return to_display(pts), rgba


def _colorize(scalar01, cmap_id):
    gi = (np.clip(scalar01, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    try:
        import cv2
        bgr = cv2.applyColorMap(gi, cmap_id).reshape(-1, 3).astype(np.float32) / 255.0
        rgb = bgr[:, ::-1]
    except ImportError:
        g = gi.astype(np.float32) / 255.0
        rgb = np.concatenate([g, g, g], axis=1)
    return np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=1)
