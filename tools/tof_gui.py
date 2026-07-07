#!/usr/bin/env python3
"""
VL53L9CX ToF desktop app (PySide6/Qt).

A polished front-end over the SWD link: crisp upscaled image (OpenCV colormap),
grouped controls, live stats, object detection with a "show only detected" mode,
and pixel hover read-out. Reuses the rendering/detection from tof_cv.py.

Threading: an AcquisitionWorker owns the pyOCD link (pyocd sessions are not
thread-safe) and streams new frames to the GUI; a MappingWorker owns the
DepthMapper so ICP/fusion never blocks the UI. The GUI thread only processes
and renders.

Usage:
    python tof_gui.py                       # YOLO detector (needs ultralytics)
    python tof_gui.py --detector heuristic  # no ML deps
Requires: PySide6, opencv-python, pyocd, numpy (+ ultralytics for --detector yolo)
"""

import argparse
import os
import sys
import threading
import time
import warnings
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required:  pip install opencv-python")

try:
    from PySide6.QtCore import Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
        QComboBox, QSlider, QCheckBox, QGroupBox, QFormLayout, QLineEdit, QFrame,
        QPushButton, QScrollArea, QFileDialog)
except ImportError:
    sys.exit("PySide6 is required:  pip install PySide6")

import tof3d
import tof_link as TL
from detector import make_detector
from segmenter import sobel_binary
from tof_cv import render, draw_detections, _cmaps, CHANNELS, USECASES

# selectable views: raw channels + a computed SNR (confidence) map
VIEW_CHANNELS = list(CHANNELS) + ["snr"]

CLOUD_COLORS = ["depth", "amplitude", "solid"]


@dataclass
class FramePacket:
    seq: int
    w: int
    h: int
    binning: int
    dist: np.ndarray
    amp: np.ndarray
    amb: np.ndarray
    t: float
    hdr: dict


class AcquisitionWorker(QThread):
    """Owns the SWD link: polls the header, reads only genuinely new frames,
    applies queued sensor-control writes. Emits FramePackets to the GUI and
    hands the same packet to the MappingWorker's mailbox (worker-to-worker,
    so mapping rate is independent of GUI paint rate)."""

    frame_ready = Signal(object)   # FramePacket
    status = Signal(object)        # raw header dict while no valid frame

    def __init__(self, link):
        super().__init__()
        self.link = link
        self.mailbox = None        # optional MappingWorker
        self._ctrl = deque()
        self._halt = threading.Event()
        self._gui_ack = threading.Event()
        self._gui_ack.set()

    def request_ctrl(self, usecase, fps):
        self._ctrl.append((usecase, fps))

    def ack(self):
        self._gui_ack.set()

    def stop(self):
        self._halt.set()

    def run(self):
        last_seq = -1
        while not self._halt.is_set():
            try:
                while self._ctrl:
                    uc, fps = self._ctrl.popleft()
                    self.link.send_ctrl(uc, fps)
                hd = self.link.header()
            except Exception:  # noqa: BLE001  (probe glitch: retry)
                time.sleep(0.2)
                continue
            if hd["magic"] != TL.MAGIC or hd["w"] == 0 or \
                    hd["w"] * hd["h"] > TL.MAX_PX:
                self.status.emit(hd)
                time.sleep(0.1)
                continue
            if hd["seq"] == last_seq:
                time.sleep(0.004)
                continue
            w, h = hd["w"], hd["h"]
            try:
                dist, amp, amb = self.link.all_channels(w, h)
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
                continue
            last_seq = hd["seq"]
            pkt = FramePacket(hd["seq"], w, h, hd["binning"] or 2,
                              dist, amp, amb, time.time(), hd)
            if self.mailbox is not None:
                self.mailbox.submit(pkt)
            # drop GUI frames while the GUI is still busy with the last one
            if self._gui_ack.is_set():
                self._gui_ack.clear()
                self.frame_ready.emit(pkt)


class MappingWorker(QThread):
    """Owns the DepthMapper. Frames arrive via a capacity-1 latest-frame-wins
    mailbox (a stale frame only increases ICP inter-frame motion); commands
    via a deque. A generation counter makes reset race-free: the GUI drops
    results carrying a stale gen. Map updates are coalesced: at most one
    in flight, at least 150 ms apart."""

    map_updated = Signal(object)   # dict(gen, text, kf, npts[, pts, cols])
    op_done = Signal(object)       # dict(gen, text) for finish()

    def __init__(self, hfov=None):
        super().__init__()
        self.hfov = hfov
        self._cond = threading.Condition()
        self._slot = None
        self._cmds = deque()
        self._halt = threading.Event()
        self._acked = threading.Event()
        self._acked.set()
        self._last_emit = 0.0

    def submit(self, pkt):
        with self._cond:
            self._slot = pkt
            self._cond.notify()

    def send(self, *cmd):
        self._cmds.append(cmd)
        with self._cond:
            self._cond.notify()

    def ack(self):
        self._acked.set()

    def stop(self):
        self._halt.set()
        with self._cond:
            self._cond.notify()

    def _emit_update(self, gen, mapper, force=False, with_cloud=True):
        now = time.time()
        if not force and (now - self._last_emit < 0.15
                          or not self._acked.is_set()):
            return
        d = dict(gen=gen, kf=len(mapper.keyframes), npts=len(mapper.vox),
                 text=(f"{'LOST - move back' if mapper.lost else 'tracking'}  "
                       f"fit {mapper.fitness:.2f}\n"
                       f"frames {mapper.tracked}/{mapper.frames}  "
                       f"kf {len(mapper.keyframes)}  map {len(mapper.vox)} vox"))
        if with_cloud:
            pts, cols = mapper.map_points()
            d["pts"], d["cols"] = tof3d.to_display(pts), cols
        self._acked.clear()
        self._last_emit = now
        self.map_updated.emit(d)

    def run(self):
        mapper, active, gen = None, False, 0
        while not self._halt.is_set():
            while self._cmds:
                cmd = self._cmds.popleft()
                if cmd[0] == "start":
                    gen, depth_max = cmd[1], cmd[2]
                    if mapper is None:
                        try:
                            from mapper import DepthMapper  # lazy: needs scipy
                            mapper = DepthMapper(hfov_deg=self.hfov)
                        except Exception as e:  # noqa: BLE001
                            self.map_updated.emit(dict(
                                gen=gen, kf=0, npts=0,
                                text=("mapper unavailable: " + str(e))[:60]))
                            continue
                    mapper.depth_max = depth_max
                    active = True
                elif cmd[0] == "stop":
                    active = False
                    if mapper is not None:
                        self.map_updated.emit(dict(
                            gen=gen, kf=len(mapper.keyframes),
                            npts=len(mapper.vox),
                            text=(f"stopped: {mapper.tracked} frames, "
                                  f"{len(mapper.keyframes)} keyframes")))
                elif cmd[0] == "reset":
                    gen = cmd[1]
                    if mapper is not None:
                        mapper.reset()
                    self.map_updated.emit(dict(gen=gen, kf=0, npts=0,
                                               text="map cleared"))
                elif cmd[0] == "finish":
                    g, path = cmd[1], cmd[2]
                    try:
                        txt = mapper.finish(path) if mapper is not None \
                            else "no frames recorded"
                    except Exception as e:  # noqa: BLE001
                        txt = ("save failed: " + str(e))[:60]
                    self.op_done.emit(dict(gen=g, text=txt))

            pkt = None
            with self._cond:
                if self._slot is None and not self._cmds:
                    self._cond.wait(0.1)
                pkt, self._slot = self._slot, None
            if pkt is None or not active or mapper is None:
                continue
            try:
                mapper.add_frame(pkt.dist, pkt.amp, pkt.binning)
                self._emit_update(gen, mapper)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                active = False
                self.map_updated.emit(dict(gen=gen, kf=0, npts=0,
                                           text=("map error: " + str(e))[:60]))


class PointCloudWindow(QMainWindow):
    """Pop-out rotatable 3-D view (needs pyqtgraph + PyOpenGL). Z is up,
    depth extends along the grid (tof3d.to_display frame)."""

    def __init__(self):
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl  # optional deps, imported lazily
        super().__init__()
        self._pg = pg
        self.setWindowTitle("ToF 3D point cloud")
        self.resize(720, 620)
        self.view = gl.GLViewWidget()
        self.setCentralWidget(self.view)
        self.view.setCameraPosition(distance=3.0, elevation=22, azimuth=-135)
        grid = gl.GLGridItem(); grid.setSize(4, 4); grid.setSpacing(0.25, 0.25)
        self.view.addItem(grid)
        axis = gl.GLAxisItem(); axis.setSize(0.5, 0.5, 0.5)
        self.view.addItem(axis)
        self._sp = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=4)
        self.view.addItem(self._sp)
        self._centered = False

    def set_points(self, pts, colors, size=4):
        if len(pts) == 0:
            pts = np.zeros((1, 3), np.float32)
            colors = np.zeros((1, 4), np.float32)
        elif not self._centered and len(pts) > 100:
            c = np.median(pts, axis=0)
            self.view.opts["center"] = self._pg.Vector(*c.tolist())
            self._centered = True
        self._sp.setData(pos=pts, color=colors, size=size)

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
    def __init__(self, link, detector, cmaps, hfov=None):
        super().__init__()
        self.detector, self.cmaps, self.hfov = detector, cmaps, hfov
        self.setWindowTitle("VL53L9CX ToF")
        self.resize(1100, 640)

        self.dist = self.amp = self.amb = None
        self.frame_wh = None
        self._pkt = None
        self._buf, self._buf_wh = {}, None  # temporal-denoise ring buffers
        self.last_dets, self.last_det_t, self.det_err = [], 0.0, None
        self.t_prev, self.fps_meas = time.time(), 0.0
        self.bg = self.prev_depth = None        # background model / previous depth
        self.pcl_win, self._pcl_err = None, None
        self._upsampler, self._dl_err = None, None
        self.mapping = False
        self._map_gen, self._map_kf, self._map_npts = 0, 0, 0
        self._rec = None                        # raw-frame recorder state
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

        # ---- 3D view ----
        self.pcl = QCheckBox("3D point cloud (window)")
        self.cloud_color = QComboBox(); self.cloud_color.addItems(["Depth", "Amplitude", "Solid"])
        self.cloud_filter = QCheckBox("Noise filter (confidence + edges)")
        self.cloud_filter.setChecked(True)
        self.cloud_size = slider(1, 12, 4)
        v3 = QGroupBox("3D view"); f3 = QFormLayout(v3)
        f3.addRow(self.pcl)
        f3.addRow("Cloud color", self.cloud_color)
        f3.addRow(self.cloud_filter)
        f3.addRow("Point size", self.cloud_size)

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

        # ---- 3D map recording (projective ICP + TSDF, see mapper.py) ----
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

        # ---- raw capture (offline tuning / replay.py) ----
        self.rec_btn = QPushButton("Record raw frames")
        self.rec_lbl = QLabel("idle"); self.rec_lbl.setObjectName("stat")
        cap = QGroupBox("Capture (raw .npz for replay.py)")
        fc = QFormLayout(cap)
        fc.addRow(self.rec_btn); fc.addRow(self.rec_lbl)
        self.rec_btn.clicked.connect(self.toggle_record)

        panel = QVBoxLayout()
        for g in (disp, v3, proc, sens, det, mp, m3, cap, info):
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

        # ---- workers ----
        self.acq = AcquisitionWorker(link)
        self.map_worker = MappingWorker(hfov=self.hfov)
        self.acq.mailbox = self.map_worker
        self.acq.frame_ready.connect(self.on_frame)
        self.acq.status.connect(self.on_status)
        self.map_worker.map_updated.connect(self.on_map_update)
        self.map_worker.op_done.connect(self.on_op_done)
        self.map_worker.start()
        self.acq.start()
        self.apply_settings()

        # slow refresh so UI control changes take effect between frames
        self.refresh = QTimer(self)
        self.refresh.timeout.connect(self.render_frame)
        self.refresh.start(150)

    # ---- control handlers ----
    def apply_settings(self):
        self.acq.request_ctrl(self.uc.currentIndex(), self.fps.value())

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
            # points beyond the display "Max dist" are excluded from the map,
            # so a turntable-style scan can cut away the background
            self.map_worker.send("start", self._map_gen, self.dmax.value() / 100.0)
            self.mapping = True
            self.map_btn.setText("Stop recording")
            self.map_lbl.setText("recording ... move slowly")
        else:
            self.mapping = False
            self.map_btn.setText("Start recording")
            self.map_worker.send("stop",)

    def reset_map(self):
        self._map_gen += 1
        self._map_kf, self._map_npts = 0, 0
        self.map_worker.send("reset", self._map_gen)
        self.map_save.setEnabled(False)
        self.map_lbl.setText("map cleared")
        if self.mapping:
            self.map_worker.send("start", self._map_gen, self.dmax.value() / 100.0)

    def save_map(self):
        if self._map_kf == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save 3D map", "tof_map.ply", "PLY mesh (*.ply)")
        if not path:
            return
        self.map_lbl.setText("meshing ...")
        self.map_worker.send("finish", self._map_gen, path)

    def on_map_update(self, d):
        try:
            if d.get("gen") != self._map_gen:
                return
            self._map_kf = d.get("kf", self._map_kf)
            self._map_npts = d.get("npts", self._map_npts)
            if d.get("text"):
                self.map_lbl.setText(d["text"])
            self.map_save.setEnabled(self._map_kf > 0)
            pts = d.get("pts")
            if pts is not None and self.map_show.isChecked() and len(pts) > 0 \
                    and self.ensure_pcl():
                self.pcl_win.set_points(pts, d["cols"], self.cloud_size.value())
        finally:
            self.map_worker.ack()

    def on_op_done(self, d):
        if d.get("gen") == self._map_gen:
            self.map_lbl.setText(d["text"])

    # ---- raw capture ----
    def toggle_record(self):
        if self._rec is None:
            self._rec = dict(dist=[], amp=[], amb=[], seq=[], t=[],
                             wh=None, binning=2, skipped=0)
            self.rec_btn.setText("Stop && save .npz")
            self.rec_lbl.setText("recording raw frames ...")
            return
        rec, self._rec = self._rec, None
        self.rec_btn.setText("Record raw frames")
        if not rec["dist"]:
            self.rec_lbl.setText("nothing recorded")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save raw capture", "tof_capture.npz", "NumPy archive (*.npz)")
        if not path:
            self.rec_lbl.setText("discarded")
            return
        np.savez_compressed(
            path, dist=np.stack(rec["dist"]), amp=np.stack(rec["amp"]),
            amb=np.stack(rec["amb"]), seq=np.array(rec["seq"], np.int64),
            t=np.array(rec["t"]), binning=rec["binning"])
        self.rec_lbl.setText(f"{len(rec['dist'])} frames -> {os.path.basename(path)}")

    def _record(self, pkt):
        rec = self._rec
        if rec["wh"] is None:
            rec["wh"], rec["binning"] = (pkt.w, pkt.h), pkt.binning
        if rec["wh"] != (pkt.w, pkt.h):
            rec["skipped"] += 1
            return
        if len(rec["dist"]) >= 20000:       # ~250 MB uncompressed: hard stop
            self.toggle_record()
            return
        rec["dist"].append(np.asarray(pkt.dist, np.uint16).copy())
        rec["amp"].append(np.asarray(pkt.amp, np.uint16).copy())
        rec["amb"].append(np.asarray(pkt.amb, np.uint16).copy())
        rec["seq"].append(pkt.seq)
        rec["t"].append(pkt.t)
        self.rec_lbl.setText(f"recording ... {len(rec['dist'])} frames"
                             + (f" ({rec['skipped']} skipped)" if rec["skipped"] else ""))

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

    # ---- frame handling ----
    def on_status(self, hd):
        if self._pkt is None or time.time() - self._pkt.t > 1.0:
            self.stat.setText(f"no frame yet\nfsm={hd['fsm']}  power-cycle the board?")

    def on_frame(self, pkt):
        try:
            self._pkt = pkt
            self.dist, self.amp, self.amb = self.process(
                (pkt.dist, pkt.amp, pkt.amb), pkt.w, pkt.h)

            now = time.time(); dt = now - self.t_prev; self.t_prev = now
            if dt > 0:
                self.fps_meas = 0.9 * self.fps_meas + 0.1 / dt

            # motion metric (per new frame, not per repaint)
            self._motion = 0.0
            if self.prev_depth is not None and self.prev_depth.shape == self.dist.shape:
                vv = (self.dist > 0) & (self.prev_depth > 0)
                if vv.any():
                    self._motion = float(np.abs(self.dist[vv] - self.prev_depth[vv]).mean())
            self.prev_depth = np.asarray(self.dist, dtype=np.float32).copy()

            if self._rec is not None:
                self._record(pkt)

            self.render_frame()
            self.update_cloud()
        finally:
            self.acq.ack()

    def update_cloud(self):
        """Live point cloud from the current frame (the recorded map, when
        shown, arrives via on_map_update and takes precedence)."""
        show_map = self.map_show.isChecked() and self._map_npts > 0
        if self.pcl.isChecked() and not show_map and self.dist is not None:
            if self.ensure_pcl():
                try:
                    filt = self.cloud_filter.isChecked()
                    pts, cols = tof3d.live_pointcloud(
                        np.asarray(self.dist), self.amp,
                        binning=self._pkt.binning if self._pkt else 2,
                        hfov_deg=self.hfov,
                        dmax_mm=max(200, self.dmax.value() * 10),
                        amp_frac=0.05 if filt else 0.0,
                        edge_jump=0.06 if filt else 0.0,
                        color=CLOUD_COLORS[self.cloud_color.currentIndex()],
                        cmap_id=self.cmaps[self.cmap.currentIndex()][1])
                    self.pcl_win.set_points(pts, cols, self.cloud_size.value())
                except Exception:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
        elif not self.pcl.isChecked() and not show_map and \
                self.pcl_win is not None and self.pcl_win.isVisible():
            self.pcl_win.hide()

    def render_frame(self):
        pkt = self._pkt
        if pkt is None or self.dist is None:
            return
        hd = pkt.hdr
        h, w = self.dist.shape  # DL upsample may have grown the working grid
        view = VIEW_CHANNELS[self.chan.currentIndex()]
        if view == "snr":
            chan = self.amp / (self.amb + 1.0)   # confidence: signal / ambient noise floor
        else:
            chan = {"distance": self.dist, "amplitude": self.amp, "ambient": self.amb}[view]

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

        # background subtraction / presence
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
        for worker in (self.map_worker, self.acq):
            try:
                worker.stop()
                worker.wait(3000)
            except Exception:  # noqa: BLE001
                pass
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
    ap.add_argument("--fov", type=float, default=None,
                    help="override horizontal FoV (deg); default: ST optical model")
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
    try:
        session.close()
    except Exception:  # noqa: BLE001  (already closed in closeEvent)
        pass
    sys.exit(ret)


if __name__ == "__main__":
    main()
