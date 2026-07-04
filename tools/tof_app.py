#!/usr/bin/env python3
"""
VL53L9CX live viewer + control panel, over the SWD debug link (pyOCD).

Reads the firmware's RAM mailbox `g_live` (distance / amplitude / ambient, up to
full 54x42) and writes the control block `g_ctrl` to change settings at runtime.
No UART is involved.

  - pick the channel to display: Distance / Amplitude / Ambient
  - pick the use case (resolution): full 54x42 or 24x20, range or precision
  - set the frame rate
  - hover the image to read the exact distance / amplitude / ambient of a pixel
  - live stats: resolution, measured fps, temperature, dropped frames

Usage:
    python tof_app.py                       # auto: elf=build/..., target auto
    python tof_app.py --elf path/to.elf
    python tof_app.py --gl 0x.... --gc 0x....

Requires: pyocd, numpy, matplotlib   (pip install -r requirements.txt)
"""

import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.widgets import RadioButtons, Slider, CheckButtons

from detector import make_detector

MAGIC = 0x4556494C
MAX_W, MAX_H = 54, 42
MAX_PX = MAX_W * MAX_H

# g_live field offsets (see vl53l9_app.c live_mailbox_t)
HDR_FMT = "<13I"                 # 13 uint32
HDR_U32 = struct.calcsize(HDR_FMT)      # 52
OFF_WH = HDR_U32                        # width,height (u16,u16) at 52
HDR_SIZE = OFF_WH + 4                   # 56
OFF_DIST = HDR_SIZE                      # distance[]
OFF_AMP = OFF_DIST + MAX_PX * 2          # amplitude[]
OFF_AMB = OFF_AMP + MAX_PX * 2           # ambient[]

CHANNELS = ["Distance (mm)", "Amplitude", "Ambient"]
CHAN_OFF = {"Distance (mm)": OFF_DIST, "Amplitude": OFF_AMP, "Ambient": OFF_AMB}

# use case -> label; index written to g_ctrl.usecase
USECASES = [
    ("Full 54x42 (range)", 0),
    ("Full 54x42 (precision)", 1),
    ("24x20 (range)", 2),
    ("24x20 (fast)", 3),
]


def find_symbol_addr(elf, name):
    for nm in ("arm-none-eabi-nm", "nm"):
        try:
            out = subprocess.check_output([nm, elf], stderr=subprocess.DEVNULL).decode()
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            p = line.split()
            if len(p) == 3 and p[2] == name:
                return int(p[0], 16)
    return None


def get_cmap(name):
    try:
        return matplotlib.colormaps[name].copy()
    except (AttributeError, KeyError):
        from matplotlib import cm
        return cm.get_cmap(name).copy()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_elf = os.path.join(here, "..", "build", "53L9A1_LiveDepth.elf")

    ap = argparse.ArgumentParser(description="VL53L9CX live viewer + control (SWD)")
    ap.add_argument("--elf", default=default_elf)
    ap.add_argument("--gl", type=lambda x: int(x, 0), default=None, help="g_live address")
    ap.add_argument("--gc", type=lambda x: int(x, 0), default=None, help="g_ctrl address")
    ap.add_argument("--target", default="stm32h533retx")
    ap.add_argument("--dmax", type=int, default=2500, help="distance color-scale max (mm)")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--detector", default="yolo",
                    help="object-detection backend: yolo (Ultralytics, needs "
                         "`pip install ultralytics`) | heuristic (depth blob, no deps)")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="YOLO weights (auto-downloaded)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--classes", default=None,
                    help="comma-separated COCO class ids to keep (e.g. 0 = person only)")
    args = ap.parse_args()

    det_kw = {}
    if args.detector in ("yolo", "ultralytics", "dl"):
        det_kw = dict(model=args.model, conf=args.conf,
                      classes=[int(c) for c in args.classes.split(",")] if args.classes else None)
    try:
        detector = make_detector(args.detector, **det_kw)
    except Exception as exc:  # noqa: BLE001
        print(f"detector '{args.detector}' unavailable ({exc}); using heuristic")
        detector = make_detector("heuristic")
    print(f"detector: {detector.name}")

    gl, gc = args.gl, args.gc
    if gl is None or gc is None:
        if not os.path.exists(args.elf):
            sys.exit(f"ELF not found: {args.elf}. Build first or pass --gl/--gc.")
        gl = gl or find_symbol_addr(args.elf, "g_live")
        gc = gc or find_symbol_addr(args.elf, "g_ctrl")
        if gl is None or gc is None:
            sys.exit("Could not resolve g_live/g_ctrl (need arm-none-eabi-nm), pass --gl/--gc.")
    print(f"g_live @ 0x{gl:08x}   g_ctrl @ 0x{gc:08x}")

    try:
        from pyocd.core.helpers import ConnectHelper
    except ImportError:
        sys.exit("pyocd is required:  pip install pyocd")

    session = ConnectHelper.session_with_chosen_probe(
        target_override=args.target, options={"connect_mode": "attach", "frequency": 4000000})
    if session is None:
        sys.exit("No debug probe found. Connect the board via ST-Link USB.")
    session.open()
    target = session.target
    try:
        if target.get_state().name == "HALTED":
            target.resume()
    except Exception:  # noqa: BLE001
        pass

    def rd(addr, n):
        return bytes(target.read_memory_block8(addr, n))

    def wr32(addr, val):
        target.write_memory(addr, val & 0xFFFFFFFF)

    # current g_ctrl cmd_seq so we can bump it
    ctrl = struct.unpack("<4I", rd(gc, 16))
    state = {"cmd_seq": ctrl[0], "usecase": ctrl[1], "fps": 15,
             "channel": CHANNELS[0], "seq": -1, "t": time.time(), "fps_meas": 0.0,
             "detect": False, "showonly": False, "dets": [], "det_err": None}

    def send_ctrl():
        wr32(gc + 4, state["usecase"])                 # usecase
        fp = int(1_000_000 / max(state["fps"], 1))
        wr32(gc + 8, fp)                               # frame_period_us
        wr32(gc + 12, 0)                               # exposure = default
        state["cmd_seq"] += 1
        wr32(gc + 0, state["cmd_seq"])                 # cmd_seq (apply)

    send_ctrl()  # push initial settings

    # ---- figure / widgets ----
    cmap = get_cmap(args.cmap)
    cmap.set_bad(color="black")
    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0.05, 0.08, 0.60, 0.85])
    img = ax.imshow(np.ma.masked_equal(np.zeros((MAX_H, MAX_W), np.uint16), 0),
                    cmap=cmap, vmin=0, vmax=args.dmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
    # people-segmentation overlay layer (RGBA), transparent until enabled
    overlay_img = ax.imshow(np.zeros((MAX_H, MAX_W, 4), np.float32),
                            interpolation="nearest", zorder=3)
    ttl = ax.set_title("starting ...")

    ax_ch = fig.add_axes([0.70, 0.66, 0.27, 0.24]); ax_ch.set_title("Channel", fontsize=9)
    rb_ch = RadioButtons(ax_ch, CHANNELS)
    ax_uc = fig.add_axes([0.70, 0.34, 0.27, 0.28]); ax_uc.set_title("Resolution / use case", fontsize=9)
    rb_uc = RadioButtons(ax_uc, [u[0] for u in USECASES], active=state["usecase"])
    ax_fps = fig.add_axes([0.70, 0.30, 0.27, 0.03])
    sl_fps = Slider(ax_fps, "fps", 1, 30, valinit=state["fps"], valstep=1)
    ax_dt = fig.add_axes([0.70, 0.16, 0.27, 0.11]); ax_dt.set_frame_on(False)
    cb_dt = CheckButtons(ax_dt, [f"Detect objects ({detector.name})", "Show only detected"],
                         [False, False])
    ax_txt = fig.add_axes([0.70, 0.02, 0.27, 0.12]); ax_txt.axis("off")
    txt = ax_txt.text(0, 1, "", va="top", ha="left", fontsize=9, family="monospace")

    det_artists = []  # box/label patches redrawn each frame

    def on_checks(_label):
        st = cb_dt.get_status()
        state["detect"], state["showonly"] = st[0], st[1]
        state["seq"] = -1  # force redraw
    cb_dt.on_clicked(on_checks)

    def on_channel(label):
        state["channel"] = label
        state["seq"] = -1  # force redraw
    rb_ch.on_clicked(on_channel)

    def on_usecase(label):
        for name, idx in USECASES:
            if name == label:
                state["usecase"] = idx
        send_ctrl(); state["seq"] = -1
    rb_uc.on_clicked(on_usecase)

    def on_fps(val):
        state["fps"] = int(val); send_ctrl()
    sl_fps.on_changed(on_fps)

    hover = {"x": None, "y": None}

    def on_move(ev):
        if ev.inaxes is ax and ev.xdata is not None:
            hover["x"], hover["y"] = int(round(ev.xdata)), int(round(ev.ydata))
        else:
            hover["x"] = hover["y"] = None
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    def read_header():
        b = rd(gl, HDR_SIZE)
        f = struct.unpack(HDR_FMT, b[:HDR_U32])
        w, h = struct.unpack("<HH", b[OFF_WH:OFF_WH + 4])
        return dict(magic=f[0], hb=f[1], fsm=f[2], frames=f[3], seq=f[4], dropped=f[5],
                    applied=f[6], usecase=f[7], binning=f[8], fp=f[9], exp=f[10],
                    temp=f[11], fc=f[12], w=w, h=h)

    def update(_):
        try:
            hd = read_header()
        except Exception:  # noqa: BLE001
            return (img, ttl, txt)
        if hd["magic"] != MAGIC or hd["w"] == 0 or hd["w"] * hd["h"] > MAX_PX:
            ttl.set_text(f"no frame yet (fsm={hd.get('fsm')}, power-cycle the board?)")
            return (img, ttl, txt)
        w, h, n = hd["w"], hd["h"], hd["w"] * hd["h"]
        if hd["seq"] != state["seq"]:
            state["seq"] = hd["seq"]
            now = time.time(); dt = now - state["t"]; state["t"] = now
            if dt > 0:
                state["fps_meas"] = 0.9 * state["fps_meas"] + 0.1 / dt
            off = CHAN_OFF[state["channel"]]
            try:
                raw = rd(gl + off, n * 2)
            except Exception:  # noqa: BLE001
                return
            frame = (np.frombuffer(raw, dtype="<u2") & 0x7FFF).reshape(h, w)

            # clear previous detection boxes/labels
            for a in det_artists:
                try:
                    a.remove()
                except Exception:  # noqa: BLE001
                    pass
            det_artists.clear()

            ov = np.zeros((h, w, 4), np.float32)   # RGBA mask overlay
            keep = None                            # union of detected regions

            if state["detect"]:
                def chan(ch_off):
                    if ch_off == off:
                        return frame
                    try:
                        return (np.frombuffer(rd(gl + ch_off, n * 2), dtype="<u2") & 0x7FFF).reshape(h, w)
                    except Exception:  # noqa: BLE001
                        return frame
                try:
                    dets = detector.detect(chan(OFF_DIST), chan(OFF_AMP), chan(OFF_AMB))
                    state["dets"], state["det_err"] = dets, None
                except Exception as e:  # noqa: BLE001
                    dets, state["dets"], state["det_err"] = [], [], str(e)
                keep = np.zeros((h, w), bool)
                palette = plt.cm.tab10.colors
                for k, det in enumerate(dets):
                    col = palette[k % len(palette)]
                    x0, y0, x1, y1 = det.box
                    rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), max(x1 - x0, 1), max(y1 - y0, 1),
                                         fill=False, edgecolor=col, lw=1.5, zorder=4)
                    ax.add_patch(rect); det_artists.append(rect)
                    lab = ax.text(x0 - 0.5, y0 - 0.8, f"{det.label} {det.score:.2f}",
                                  color="white", fontsize=7, zorder=5,
                                  bbox=dict(fc=col, alpha=0.75, pad=0.4, ec="none"))
                    det_artists.append(lab)
                    if det.mask is not None:
                        ov[det.mask] = (col[0], col[1], col[2], 0.45)
                        keep |= det.mask
                    else:
                        xa, xb = max(0, int(x0)), min(w, int(round(x1)))
                        ya, yb = max(0, int(y0)), min(h, int(round(y1)))
                        keep[ya:yb, xa:xb] = True
                        ov[ya:yb, xa:xb] = (col[0], col[1], col[2], 0.20)
            else:
                state["dets"] = []

            # base image, optionally masked to detected regions only
            if state["channel"].startswith("Distance"):
                img.set_clim(0, args.dmax)
                bad = (frame == 0)
            else:
                img.set_clim(0, max(1, int(np.percentile(frame, 99))))
                bad = np.zeros((h, w), bool)
            if state["showonly"] and keep is not None:
                bad = bad | (~keep)
            disp = np.ma.masked_where(bad, frame) if bad.any() else frame
            img.set_data(disp)
            img.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
            overlay_img.set_data(ov)
            overlay_img.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))

            valid = frame[(frame > 0) & (frame < 30000)]
            stats = (f"{w}x{h}  usecase {hd['usecase']} bin{hd['binning']}\n"
                     f"{state['fps_meas']:4.1f} fps  set {state['fps']}\n"
                     f"temp {hd['temp']}  frame {hd['fc']}\n"
                     f"dropped {hd['dropped']}\n")
            if state["detect"]:
                if state["det_err"]:
                    stats += "detector error:\n " + str(state["det_err"])[:22] + "\n"
                elif not state["dets"]:
                    stats += "objects: none\n"
                else:
                    from collections import Counter
                    cnt = Counter(d.label for d in state["dets"])
                    stats += "objects: " + ", ".join(f"{v}x {k}" for k, v in cnt.items())[:34] + "\n"
            if valid.size:
                stats += f"{state['channel']}:\n min {valid.min()} mean {int(valid.mean())} max {valid.max()}\n"
            # hovered pixel
            hx, hy = hover["x"], hover["y"]
            if hx is not None and 0 <= hx < w and 0 <= hy < h:
                i = hy * w + hx
                try:
                    d = struct.unpack("<H", rd(gl + OFF_DIST + i * 2, 2))[0] & 0x7FFF
                    a = struct.unpack("<H", rd(gl + OFF_AMP + i * 2, 2))[0]
                    b = struct.unpack("<H", rd(gl + OFF_AMB + i * 2, 2))[0]
                    stats += f"\npixel ({hx},{hy}):\n dist {d} mm\n amp  {a}\n amb  {b}"
                except Exception:  # noqa: BLE001
                    pass
            txt.set_text(stats)
            ttl.set_text(f"{state['channel']}   frame {hd['frames']}")
        return (img, ttl, txt)

    ani = matplotlib.animation.FuncAnimation(fig, update, interval=33, blit=False,
                                             cache_frame_data=False)
    try:
        plt.show()
    finally:
        session.close()
        print("bye")


if __name__ == "__main__":
    main()
