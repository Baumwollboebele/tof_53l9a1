#!/usr/bin/env python3
"""
VL53L9CX ToF desktop app (PySide6/Qt).

A polished front-end over the SWD link: crisp upscaled image (OpenCV colormap),
grouped controls, live stats, object detection with a "show only detected" mode,
and pixel hover read-out. Reuses the rendering/detection from tof_cv.py.

Usage:
    python tof_gui.py                       # YOLO detector (needs ultralytics)
    python tof_gui.py --detector heuristic  # no ML deps
Requires: PySide6, opencv-python, pyocd, numpy (+ ultralytics for --detector yolo)
"""

import argparse
import math
import os
import sys
import time
import warnings
from collections import Counter, deque

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required:  pip install opencv-python")

try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
        QComboBox, QSlider, QCheckBox, QGroupBox, QFormLayout, QLineEdit, QFrame,
        QPushButton, QScrollArea, QFileDialog)
except ImportError:
    sys.exit("PySide6 is required:  pip install PySide6")

import tof_link as TL
from detector import make_detector
from segmenter import sobel_binary
from tof_cv import render, draw_detections, _cmaps, CHANNELS, USECASES

# selectable views: raw channels + a computed SNR (confidence) map
VIEW_CHANNELS = list(CHANNELS) + ["snr"]


def depth_to_pointcloud(dist, hfov_deg, dmax, cmap_id):
    """Convert a radial-distance depth map to a coloured 3-D point cloud.

    Each pixel maps to a ray from the sensor FoV; the radial distance becomes a
    perpendicular Z. Returns (pts Nx3 in metres [X right, Y up, Z forward],
    rgba Nx4 in 0..1) coloured by distance. hfov_deg is the horizontal FoV
    (vertical is derived from the frame aspect ratio - it is approximate)."""
    h, w = dist.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    hfov = math.radians(hfov_deg)
    vfov = hfov * h / w
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    fy = (h / 2.0) / math.tan(vfov / 2.0)
    cc, rr_ = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xx = (cc - cx) / fx
    yy = (rr_ - cy) / fy
    norm = np.sqrt(xx * xx + yy * yy + 1.0)
    d = np.asarray(dist, dtype=np.float32)
    z = d / norm                                   # radial -> perpendicular
    valid = d > 0
    pts = np.stack([(xx * z)[valid], -(yy * z)[valid], z[valid]], axis=1) / 1000.0
    g = np.clip(d[valid] / max(dmax, 1), 0, 1)
    gi = (g * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(gi, cmap_id).reshape(-1, 3).astype(np.float32) / 255.0
    rgba = np.concatenate([bgr[:, ::-1], np.ones((len(bgr), 1), np.float32)], axis=1)
    return pts, rgba


class PointCloudWindow(QMainWindow):
    """Pop-out rotatable 3-D view (needs pyqtgraph + PyOpenGL)."""

    def __init__(self):
        import pyqtgraph.opengl as gl  # optional deps, imported lazily
        super().__init__()
        self.setWindowTitle("ToF 3D point cloud")
        self.resize(720, 620)
        self.view = gl.GLViewWidget()
        self.setCentralWidget(self.view)
        self.view.setCameraPosition(distance=3.0, elevation=12, azimuth=-70)
        grid = gl.GLGridItem(); grid.setSize(4, 4); grid.setSpacing(0.25, 0.25)
        self.view.addItem(grid)
        self._sp = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=4)
        self.view.addItem(self._sp)

    def set_points(self, pts, colors):
        if len(pts) == 0:
            pts = np.zeros((1, 3), np.float32)
            colors = np.zeros((1, 4), np.float32)
        self._sp.setData(pos=pts, color=colors, size=4)

DARK = """
QWidget { background:#20242b; color:#e6e6e6; font-size:12px; }
QGroupBox { border:1px solid #3a4048; border-radius:6px; margin-top:10px; padding:8px; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#8ab4f8; }
QComboBox, QLineEdit { background:#2b3038; border:1px solid #3a4048; border-radius:4px; padding:3px; }
QSlider::groove:horizontal { height:5px; background:#3a4048; border-radius:2px; }
QSlider::handle:horizontal { background:#8ab4f8; width:14px; margin:-5px 0; border-radius:7px; }
QCheckBox::indicator { width:16px; height:16px; }
QLabel#stat { font-family: Consolas, monospace; color:#b8f0c0; }
QLabel#title { color:#8ab4f8; font-weight:bold; }
"""


class ImageView(QLabel):
    """Displays a BGR frame and reports the hovered pixel as (fx, fy) in 0..1."""
    hovered = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.StyledPanel)

    def set_bgr(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
        self.setPixmap(pm)

    def mouseMoveEvent(self, ev):
        pm = self.pixmap()
        if pm is None or pm.width() == 0:
            self.hovered.emit(None); return
        x0 = (self.width() - pm.width()) / 2
        y0 = (self.height() - pm.height()) / 2
        px, py = ev.position().x() - x0, ev.position().y() - y0
        if 0 <= px < pm.width() and 0 <= py < pm.height():
            self.hovered.emit((px / pm.width(), py / pm.height()))
        else:
            self.hovered.emit(None)


def slider(mn, mx, val, step=1):
    s = QSlider(Qt.Horizontal)
    s.setRange(mn, mx); s.setValue(val); s.setSingleStep(step)
    return s


class ToFWindow(QMainWindow):
    def __init__(self, link, detector, cmaps, hfov=60.0):
        super().__init__()
        self.link, self.detector, self.cmaps, self.hfov = link, detector, cmaps, hfov
        self.setWindowTitle("VL53L9CX ToF")
        self.resize(1100, 640)

        self.dist = self.amp = self.amb = None
        self.frame_wh = None
        self._buf, self._buf_wh = {}, None  # temporal-denoise ring buffers
        self.last_dets, self.last_det_t, self.det_err = [], 0.0, None
        self.last_seq, self.t_prev, self.fps_meas = -1, time.time(), 0.0
        self.bg = self.prev_depth = None        # background model / previous depth
        self.pcl_win, self._pcl_err = None, None
        self._upsampler, self._dl_err = None, None
        self.mapper, self.mapping, self._map_seq = None, False, -1
        self._motion, self._presence, self._fg_frac = 0.0, None, 0.0

        # ---- image ----
        self.view = ImageView()
        self.view.hovered.connect(self.on_hover)

        # ---- controls ----
        self.chan = QComboBox()
        self.chan.addItems([("SNR" if c == "snr" else c.capitalize()) for c in VIEW_CHANNELS])
        self.cmap = QComboBox(); self.cmap.addItems([n for n, _ in cmaps]); self.cmap.setCurrentIndex(min(1, len(cmaps) - 1))
        self.size = slider(320, 1280, 720)
        self.dmax = slider(20, 600, 250)
        self.dmax_lbl = QLabel("2.5 m")
        disp = QGroupBox("Display"); f = QFormLayout(disp)
        f.addRow("Channel", self.chan); f.addRow("Colormap", self.cmap)
        f.addRow("Size", self.size)
        dr = QHBoxLayout(); dr.addWidget(self.dmax); dr.addWidget(self.dmax_lbl)
        wdr = QWidget(); wdr.setLayout(dr); f.addRow("Max dist", wdr)
        self.minint = slider(0, 100, 0); self.minint_lbl = QLabel("off")
        mr = QHBoxLayout(); mr.addWidget(self.minint); mr.addWidget(self.minint_lbl)
        wmr = QWidget(); wmr.setLayout(mr); f.addRow("Min intensity", wmr)
        self.edges = QCheckBox("Sobel edges overlay"); f.addRow(self.edges)
        self.pcl = QCheckBox("3D point cloud (window)"); f.addRow(self.pcl)

        self.uc = QComboBox(); self.uc.addItems(USECASES); self.uc.setCurrentIndex(1)
        self.fps = slider(1, 30, 15); self.fps_lbl = QLabel("15")
        sens = QGroupBox("Sensor"); fs = QFormLayout(sens)
        fs.addRow("Resolution", self.uc)
        fr = QHBoxLayout(); fr.addWidget(self.fps); fr.addWidget(self.fps_lbl)
        wfr = QWidget(); wfr.setLayout(fr); fs.addRow("Frame rate", wfr)

        self.detect = QCheckBox(f"Detect objects ({detector.name})")
        self.showonly = QCheckBox("Show only detected")
        self.edge_refine = QCheckBox("Edge-refine mask (Sobel, heuristic)")
        self.conf = slider(1, 95, 25); self.conf_lbl = QLabel("0.25")
        self.classes = QLineEdit(); self.classes.setPlaceholderText("COCO ids e.g. 0=person (blank=all)")
        det = QGroupBox("Object detection"); fd = QFormLayout(det)
        fd.addRow(self.detect); fd.addRow(self.showonly); fd.addRow(self.edge_refine)
        cr = QHBoxLayout(); cr.addWidget(self.conf); cr.addWidget(self.conf_lbl)
        wcr = QWidget(); wcr.setLayout(cr); fd.addRow("Confidence", wcr)
        fd.addRow("Classes", self.classes)

        self.stat = QLabel(); self.stat.setObjectName("stat"); self.stat.setAlignment(Qt.AlignTop)
        self.hover_lbl = QLabel("pixel: -"); self.hover_lbl.setObjectName("stat")
        info = QGroupBox("Info"); fi = QVBoxLayout(info)
        fi.addWidget(self.stat); fi.addWidget(self.hover_lbl)

        # ---- processing ----
        self.temporal = slider(1, 16, 1); self.temporal_lbl = QLabel("off")
        self.bilateral = QCheckBox("Edge-preserving smooth")
        self.bstrength = slider(1, 50, 10); self.bstrength_lbl = QLabel("10")
        proc = QGroupBox("Processing"); fp2 = QFormLayout(proc)
        tr = QHBoxLayout(); tr.addWidget(self.temporal); tr.addWidget(self.temporal_lbl)
        wtr = QWidget(); wtr.setLayout(tr); fp2.addRow("Temporal", wtr)
        fp2.addRow(self.bilateral)
        br = QHBoxLayout(); br.addWidget(self.bstrength); br.addWidget(self.bstrength_lbl)
        wbr = QWidget(); wbr.setLayout(br); fp2.addRow("Smooth strength", wbr)
        self.dl_up = QCheckBox("DL upsample (FSRCNN)")
        self.dl_scale = QComboBox(); self.dl_scale.addItems(["x2", "x3", "x4"])
        ur = QHBoxLayout(); ur.addWidget(self.dl_up); ur.addWidget(self.dl_scale)
        wur = QWidget(); wur.setLayout(ur); fp2.addRow(wur)
        self.temporal.valueChanged.connect(lambda v: self.temporal_lbl.setText("off" if v <= 1 else str(v)))
        self.bstrength.valueChanged.connect(lambda v: self.bstrength_lbl.setText(str(v)))

        # ---- motion / presence ----
        self.bg_learn = QPushButton("Learn background")
        self.bg_sub = QCheckBox("Background subtract")
        self.bg_thr = slider(5, 200, 20); self.bg_thr_lbl = QLabel("20 cm")
        self.bg_only = QCheckBox("Show only foreground")
        mp = QGroupBox("Motion / presence"); fmp = QFormLayout(mp)
        fmp.addRow(self.bg_learn); fmp.addRow(self.bg_sub)
        bgr = QHBoxLayout(); bgr.addWidget(self.bg_thr); bgr.addWidget(self.bg_thr_lbl)
        wbgr = QWidget(); wbgr.setLayout(bgr); fmp.addRow("Threshold", wbgr)
        fmp.addRow(self.bg_only)
        self.bg_thr.valueChanged.connect(lambda v: self.bg_thr_lbl.setText(f"{v} cm"))
        self.bg_learn.clicked.connect(self.learn_bg)

        # ---- 3D map recording (ICP + TSDF, see mapper.py) ----
        self.map_btn = QPushButton("Start recording")
        self.map_show = QCheckBox("Show map in 3D window"); self.map_show.setChecked(True)
        self.map_save = QPushButton("Save mesh (.ply)"); self.map_save.setEnabled(False)
        self.map_reset = QPushButton("Reset")
        self.map_lbl = QLabel("idle"); self.map_lbl.setObjectName("stat")
        m3 = QGroupBox("3D map (record && move sensor slowly)")
        fm3 = QFormLayout(m3)
        fm3.addRow(self.map_btn); fm3.addRow(self.map_show)
        m3r = QHBoxLayout(); m3r.addWidget(self.map_save); m3r.addWidget(self.map_reset)
        wm3 = QWidget(); wm3.setLayout(m3r); fm3.addRow(wm3)
        fm3.addRow(self.map_lbl)
        self.map_btn.clicked.connect(self.toggle_mapping)
        self.map_save.clicked.connect(self.save_map)
        self.map_reset.clicked.connect(self.reset_map)

        panel = QVBoxLayout()
        for g in (disp, proc, sens, det, mp, m3, info):
            panel.addWidget(g)
        panel.addStretch(1)
        pw = QWidget(); pw.setLayout(panel)
        sc = QScrollArea(); sc.setWidget(pw); sc.setWidgetResizable(True)
        sc.setFixedWidth(345); sc.setFrameShape(QFrame.NoFrame)

        root = QHBoxLayout(); root.addWidget(self.view, 1); root.addWidget(sc)
        c = QWidget(); c.setLayout(root); self.setCentralWidget(c)

        # ---- signals ----
        self.dmax.valueChanged.connect(lambda v: self.dmax_lbl.setText(f"{v/100:.1f} m"))
        self.fps.valueChanged.connect(lambda v: self.fps_lbl.setText(str(v)))
        self.conf.valueChanged.connect(lambda v: self.conf_lbl.setText(f"{v/100:.2f}"))
        self.classes.editingFinished.connect(self.apply_classes)
        # only reconfigure the sensor on release / actual change (avoid churn)
        self.uc.currentIndexChanged.connect(self.apply_settings)
        self.fps.sliderReleased.connect(self.apply_settings)

        self.apply_settings()
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(30)

    # ---- control handlers ----
    def apply_settings(self):
        try:
            self.link.send_ctrl(self.uc.currentIndex(), self.fps.value())
        except Exception:  # noqa: BLE001
            pass

    def apply_classes(self):
        txt = self.classes.text().strip()
        if hasattr(self.detector, "classes"):
            try:
                self.detector.classes = [int(x) for x in txt.split(",")] if txt else None
            except ValueError:
                pass

    def on_hover(self, frac):
        if frac is None or self.frame_wh is None or self.dist is None:
            self.hover_lbl.setText("pixel: -"); return
        w, h, _, _ = self.frame_wh
        gx = min(w - 1, max(0, int(frac[0] * w)))
        gy = min(h - 1, max(0, int(frac[1] * h)))
        try:
            self.hover_lbl.setText(
                f"pixel ({gx},{gy})\n dist {int(self.dist[gy, gx])} mm\n"
                f" amp  {int(self.amp[gy, gx])}\n amb  {int(self.amb[gy, gx])}")
        except Exception:  # noqa: BLE001
            pass

    def learn_bg(self):
        if self.dist is not None:
            self.bg = np.asarray(self.dist, dtype=np.float32).copy()

    # ---- 3D map recording ----
    def toggle_mapping(self):
        if not self.mapping:
            try:
                if self.mapper is None:
                    from mapper import DepthMapper  # lazy: needs scipy
                    self.mapper = DepthMapper(hfov_deg=self.hfov)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.map_lbl.setText(("mapper unavailable: " + str(e))[:60])
                return
            # points beyond the display "Max dist" are excluded from the map,
            # so a turntable-style scan can cut away the background
            self.mapper.depth_max = self.dmax.value() / 100.0
            self.mapping = True
            self.map_btn.setText("Stop recording")
            self.map_lbl.setText("recording ... move slowly")
        else:
            self.mapping = False
            self.map_btn.setText("Start recording")
            if self.mapper is not None and self.mapper.keyframes:
                self.map_lbl.setText(f"stopped: {self.mapper.tracked} frames, "
                                     f"{len(self.mapper.keyframes)} keyframes")

    def reset_map(self):
        if self.mapper is not None:
            self.mapper.reset()
        self._map_seq = -1
        self.map_save.setEnabled(False)
        self.map_lbl.setText("map cleared")

    def save_map(self):
        if self.mapper is None or not self.mapper.keyframes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save 3D map", "tof_map.ply", "PLY mesh (*.ply)")
        if not path:
            return
        self.map_lbl.setText("meshing ...")
        QApplication.processEvents()
        try:
            self.map_lbl.setText(self.mapper.finish(path))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.map_lbl.setText(("save failed: " + str(e))[:60])

    def ensure_pcl(self):
        if self.pcl_win is not None:
            if not self.pcl_win.isVisible():
                self.pcl_win.show()
            return True
        try:
            self.pcl_win = PointCloudWindow(); self.pcl_win.show(); self._pcl_err = None
            return True
        except Exception as e:  # noqa: BLE001  (pyqtgraph/PyOpenGL missing, etc.)
            import traceback
            traceback.print_exc()
            self._pcl_err = ("pip install pyqtgraph PyOpenGL"
                             if isinstance(e, ModuleNotFoundError) else str(e))
            self.pcl.setChecked(False)
            return False

    # ---- processing pipeline (temporal denoise + edge-preserving smooth) ----
    def process(self, raw, w, h):
        dist, amp, amb = raw
        if (w, h) != self._buf_wh:                       # resolution changed -> reset history
            self._buf = {k: deque(maxlen=16) for k in ("distance", "amplitude", "ambient")}
            self._buf_wh = (w, h)
        n = self.temporal.value()
        out = {}
        for name, arr in (("distance", dist), ("amplitude", amp), ("ambient", amb)):
            dq = self._buf[name]
            dq.append(arr.astype(np.float32))
            if n > 1 and len(dq) > 1:
                # per-pixel median over the last n frames, ignoring invalid (0)
                st = np.stack(list(dq)[-n:])
                st[st == 0] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = np.nanmedian(st, axis=0)
                out[name] = np.nan_to_num(m, nan=0.0)
            else:
                out[name] = arr.astype(np.float32)
        dist, amp, amb = out["distance"], out["amplitude"], out["ambient"]
        if self.bilateral.isChecked():
            s = float(self.bstrength.value())
            dist = cv2.bilateralFilter(dist, 5, s * 10.0, 5.0)       # edge-preserving
            amp = cv2.bilateralFilter(amp, 5, max(s * 5.0, 10.0), 5.0)
        if self.dl_up.isChecked():
            scale = 2 + self.dl_scale.currentIndex()
            try:
                if self._upsampler is None or self._upsampler.scale != scale:
                    from upsampler import DeepUpsampler   # lazy: needs opencv-contrib
                    self._upsampler = DeepUpsampler(scale)
                dist = self._upsampler.upsample(dist)
                amp = self._upsampler.upsample(amp)
                amb = self._upsampler.upsample(amb)
                self._dl_err = None
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._dl_err = str(e)
                self.dl_up.setChecked(False)
        return dist, amp, amb

    # ---- frame loop ----
    def tick(self):
        try:
            hd = self.link.header()
        except Exception:  # noqa: BLE001
            return
        if hd["magic"] != TL.MAGIC or hd["w"] == 0 or hd["w"] * hd["h"] > TL.MAX_PX:
            self.stat.setText(f"no frame yet\nfsm={hd['fsm']}  power-cycle the board?")
            return
        w, h = hd["w"], hd["h"]
        try:
            raw = self.link.all_channels(w, h)
        except Exception:  # noqa: BLE001
            return
        self.dist, self.amp, self.amb = self.process(raw, w, h)
        h, w = self.dist.shape  # DL upsample may have grown the working grid
        view = VIEW_CHANNELS[self.chan.currentIndex()]
        if view == "snr":
            chan = self.amp / (self.amb + 1.0)   # confidence: signal / ambient noise floor
        else:
            chan = {"distance": self.dist, "amplitude": self.amp, "ambient": self.amb}[view]

        if hd["seq"] != self.last_seq:
            now = time.time(); dt = now - self.t_prev; self.t_prev = now
            if dt > 0:
                self.fps_meas = 0.9 * self.fps_meas + 0.1 / dt
            self.last_seq = hd["seq"]

        out_w = self.size.value(); out_h = int(round(out_w * h / w))
        cmap_id = self.cmaps[self.cmap.currentIndex()][1]
        dmax = max(200, self.dmax.value() * 10)
        col = render(chan, view, dmax, out_w, out_h, cmap_id)

        # hide low-intensity (low-amplitude = low-confidence) pixels
        pct = self.minint.value()
        if pct > 0 and self.amp is not None:
            ref = max(1.0, float(np.percentile(self.amp, 99)))
            thr = pct / 100.0 * ref
            self.minint_lbl.setText(f"< {thr:.0f}")
            dim = cv2.resize((self.amp < thr).astype(np.uint8), (out_w, out_h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            col[dim] = 0
        else:
            self.minint_lbl.setText("off")

        # motion + background subtraction / presence
        self._motion = 0.0
        if self.prev_depth is not None and self.prev_depth.shape == self.dist.shape:
            vv = (self.dist > 0) & (self.prev_depth > 0)
            if vv.any():
                self._motion = float(np.abs(self.dist[vv] - self.prev_depth[vv]).mean())
        self.prev_depth = np.asarray(self.dist, dtype=np.float32).copy()
        self._presence, self._fg_frac = None, 0.0
        if self.bg_sub.isChecked() and self.bg is not None and self.bg.shape == self.dist.shape:
            thr = self.bg_thr.value() * 10.0
            valid = (self.dist > 0) & (self.bg > 0)
            fg = valid & (np.abs(self.dist - self.bg) > thr)
            self._fg_frac = float(fg.mean())
            self._presence = self._fg_frac > 0.02
            fg_up = cv2.resize(fg.astype(np.uint8), (out_w, out_h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if self.bg_only.isChecked():
                col[~fg_up] = 0
            else:
                col[fg_up] = (0.5 * np.array((0, 255, 0)) + 0.5 * col[fg_up]).astype(np.uint8)

        # apply Sobel edge-refine to the heuristic segmenter (if in use)
        if hasattr(self.detector, "seg"):
            self.detector.seg.edge_refine = self.edge_refine.isChecked()

        dets = self.last_dets
        if self.detect.isChecked():
            now = time.time()
            if now - self.last_det_t > 0.15:  # throttle detection so the view stays smooth
                if hasattr(self.detector, "conf"):
                    self.detector.conf = self.conf.value() / 100.0
                try:
                    dets = self.detector.detect(self.dist, self.amp, self.amb)
                    self.det_err = None
                except Exception as e:  # noqa: BLE001
                    dets, self.det_err = [], str(e)
                self.last_dets, self.last_det_t = dets, now
            col = draw_detections(col, dets, w, h, self.showonly.isChecked())
        else:
            self.last_dets = []

        # crisp Sobel-edge overlay computed on the upscaled depth
        if self.edges.isChecked() and self.dist is not None:
            dd = self.dist.astype(np.float32)
            v = dd > 0
            if v.any() and not v.all():
                dd = dd.copy(); dd[~v] = float(np.median(dd[v]))  # avoid false edges at invalid px
            d_big = cv2.resize(dd, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            col[sobel_binary(d_big)] = (255, 255, 0)  # cyan edges

        # 3D map recording (only feed genuinely new frames to ICP)
        if self.mapping and self.mapper is not None and hd["seq"] != self._map_seq:
            self._map_seq = hd["seq"]
            try:
                self.mapper.add_frame(self.dist, self.amp)
                m = self.mapper
                self.map_lbl.setText(
                    f"{'LOST - move back' if m.lost else 'tracking'}  "
                    f"fit {m.fitness:.2f}\n"
                    f"frames {m.tracked}/{m.frames}  kf {len(m.keyframes)}  "
                    f"map {len(m.map_pts)} pts")
                self.map_save.setEnabled(bool(m.keyframes))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.mapping = False
                self.map_btn.setText("Start recording")
                self.map_lbl.setText(("map error: " + str(e))[:60])

        # 3D window: recorded map takes precedence over the live cloud
        show_map = (self.map_show.isChecked() and self.mapper is not None
                    and len(self.mapper.map_pts) > 0)
        if (self.pcl.isChecked() or show_map) and self.dist is not None:
            if self.ensure_pcl():
                try:
                    if show_map:
                        pts, cols = self.mapper.map_points()
                    else:
                        pts, cols = depth_to_pointcloud(
                            np.asarray(self.dist), self.hfov, dmax, cmap_id)
                    self.pcl_win.set_points(pts, cols)
                except Exception:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
        elif self.pcl_win is not None and self.pcl_win.isVisible():
            self.pcl_win.hide()

        self.view.set_bgr(col)
        self.frame_wh = (w, h, out_w, out_h)

        res = f"{w}x{h}" if (w, h) == (hd["w"], hd["h"]) else \
            f"{hd['w']}x{hd['h']} SR-> {w}x{h}"
        lines = [f"{res}  usecase {hd['usecase']} bin{hd['binning']}",
                 f"{self.fps_meas:4.1f} fps   frame {hd['fc']}",
                 f"temp {hd['temp']}   dropped {hd['dropped']}",
                 f"motion {self._motion:.0f} mm"]
        if self.bg_sub.isChecked():
            lines.append("bg: not learned" if self.bg is None else
                         f"presence: {'YES' if self._presence else 'no'}  fg {self._fg_frac * 100:.0f}%")
        if self._pcl_err:
            lines.append("3D unavailable:\n " + self._pcl_err[:24])
        if self._dl_err:
            lines.append("DL upsample unavailable:\n " + self._dl_err[:24])
        if self.detect.isChecked():
            if self.det_err:
                lines.append("detector error:\n " + self.det_err[:26])
            elif self.last_dets:
                c = Counter(d.label for d in self.last_dets)
                lines.append("objects: " + ", ".join(f"{v}x {k}" for k, v in c.items()))
            else:
                lines.append("objects: none")
        self.stat.setText("\n".join(lines))

    def closeEvent(self, ev):
        try:
            if self.pcl_win is not None:
                self.pcl_win.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.session.close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(ev)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="VL53L9CX ToF desktop app (Qt)")
    ap.add_argument("--elf", default=os.path.join(here, "..", "build", "53L9A1_LiveDepth.elf"))
    ap.add_argument("--gl", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--gc", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--target", default="stm32h533retx")
    ap.add_argument("--detector", default="yolo", help="yolo | heuristic")
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="horizontal field of view (deg) for the 3D point cloud")
    args = ap.parse_args()

    det_kw = dict(model=args.model) if args.detector in ("yolo", "ultralytics", "dl") else {}
    try:
        detector = make_detector(args.detector, **det_kw)
    except Exception as exc:  # noqa: BLE001
        print(f"detector '{args.detector}' unavailable ({exc}); using heuristic")
        detector = make_detector("heuristic")

    try:
        session, link = TL.connect(args.target, elf=args.elf, gl=args.gl, gc=args.gc)
    except Exception as exc:  # noqa: BLE001
        sys.exit(str(exc))

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK)
    win = ToFWindow(link, detector, _cmaps(), hfov=args.fov)
    win.session = session
    win.show()
    ret = app.exec()
    session.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
