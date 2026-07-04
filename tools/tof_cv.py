#!/usr/bin/env python3
"""
Crisp OpenCV live viewer for the VL53L9CX ToF (over SWD, no UART).

Reads the firmware mailbox over the debug link, upscales the low-res frame
smoothly (cubic) with a colormap for a clean image, optionally runs object
detection on the enlarged frame, and can show only the detected objects.

Controls are trackbars in the "controls" window:
    channel     0=distance 1=amplitude 2=ambient
    usecase     0=full 54x42 range  1=full precision  2=24x20 range  3=24x20 fast
    fps         target frame rate
    size        output image width (px)
    dmax(cm)    distance color-scale max
    colormap    palette
    detect      0/1 run object detection
    show only   0/1 keep only detected regions
    conf%       YOLO confidence threshold
Keys: q or ESC to quit.

Usage:
    python tof_cv.py                      # elf=build/..., YOLO detector
    python tof_cv.py --detector heuristic # no ML deps
Requires: opencv-python, pyocd, numpy (+ ultralytics for --detector yolo)
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required:  pip install opencv-python")

import tof_link as TL
from detector import make_detector

CHANNELS = ["distance", "amplitude", "ambient"]
USECASES = ["full 54x42 range", "full 54x42 precision", "24x20 range", "24x20 fast"]
PALETTE = [(0, 0, 255), (0, 255, 0), (255, 128, 0), (0, 255, 255),
           (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)]


def _cmaps():
    names = ["JET", "TURBO", "INFERNO", "VIRIDIS", "HOT", "BONE", "OCEAN"]
    out = []
    for n in names:
        cid = getattr(cv2, "COLORMAP_" + n, None)
        if cid is not None:
            out.append((n, cid))
    return out or [("JET", cv2.COLORMAP_JET)]


def render(chan, name, dmax, out_w, out_h, cmap_id):
    if name == "distance":
        g = np.clip(chan.astype(np.float32) * (255.0 / max(dmax, 1)), 0, 255).astype(np.uint8)
        invalid = (chan == 0)
    else:
        hi = max(1.0, float(np.percentile(chan, 99)))
        g = np.clip(chan.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
        invalid = None
    big = cv2.resize(g, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap(big, cmap_id)
    if invalid is not None and invalid.any():
        inv = cv2.resize(invalid.astype(np.uint8), (out_w, out_h),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        col[inv] = 0
    return col


def draw_detections(col, dets, w, h, show_only):
    out_h, out_w = col.shape[:2]
    sx, sy = out_w / w, out_h / h
    keep = np.zeros((out_h, out_w), bool)
    masks = []
    boxes = []
    for k, det in enumerate(dets):
        color = PALETTE[k % len(PALETTE)]
        x0, y0, x1, y1 = det.box
        p0 = (int(x0 * sx), int(y0 * sy))
        p1 = (int(x1 * sx), int(y1 * sy))
        boxes.append((p0, p1, color, f"{det.label} {det.score:.2f}"))
        if det.mask is not None:
            m = cv2.resize(det.mask.astype(np.uint8), (out_w, out_h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            masks.append((m, color)); keep |= m
        else:
            keep[p0[1]:p1[1], p0[0]:p1[0]] = True

    if show_only:
        col = col.copy(); col[~keep] = 0
    else:
        col = col.copy()
    for m, color in masks:
        col[m] = (0.45 * np.array(color) + 0.55 * col[m]).astype(np.uint8)
    for p0, p1, color, label in boxes:
        cv2.rectangle(col, p0, p1, color, 2)
        cv2.putText(col, label, (p0[0], max(14, p0[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return col


def hud(col, lines):
    y = 6
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(col, (4, y), (10 + tw, y + th + 8), (0, 0, 0), -1)
        cv2.putText(col, ln, (7, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += th + 10


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Crisp OpenCV ToF viewer (SWD)")
    ap.add_argument("--elf", default=os.path.join(here, "..", "build", "53L9A1_LiveDepth.elf"))
    ap.add_argument("--gl", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--gc", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--target", default="stm32h533retx")
    ap.add_argument("--detector", default="yolo", help="yolo | heuristic")
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--classes", default=None, help="comma COCO ids, e.g. 0=person")
    args = ap.parse_args()

    det_kw = {}
    if args.detector in ("yolo", "ultralytics", "dl"):
        det_kw = dict(model=args.model,
                      classes=[int(c) for c in args.classes.split(",")] if args.classes else None)
    try:
        detector = make_detector(args.detector, **det_kw)
    except Exception as exc:  # noqa: BLE001
        print(f"detector '{args.detector}' unavailable ({exc}); using heuristic")
        detector = make_detector("heuristic")
    print(f"detector: {detector.name}")

    try:
        session, link = TL.connect(args.target, elf=args.elf, gl=args.gl, gc=args.gc)
    except Exception as exc:  # noqa: BLE001
        sys.exit(str(exc))

    cmaps = _cmaps()
    cv2.namedWindow("tof", cv2.WINDOW_NORMAL)
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("controls", 420, 300)
    noop = lambda v: None
    cv2.createTrackbar("channel", "controls", 0, 2, noop)
    cv2.createTrackbar("usecase", "controls", 1, 3, noop)
    cv2.createTrackbar("fps", "controls", 15, 30, noop)
    cv2.createTrackbar("size", "controls", 720, 1400, noop)
    cv2.createTrackbar("dmax(cm)", "controls", 250, 600, noop)
    cv2.createTrackbar("colormap", "controls", min(1, len(cmaps) - 1), len(cmaps) - 1, noop)
    cv2.createTrackbar("detect", "controls", 0, 1, noop)
    cv2.createTrackbar("show only", "controls", 0, 1, noop)
    cv2.createTrackbar("conf%", "controls", 25, 95, noop)

    def gt(n):
        return cv2.getTrackbarPos(n, "controls")

    last_ctrl = None
    last_seq = -1
    t_prev = time.time()
    fps_meas = 0.0
    det_err = None

    while True:
        ch = CHANNELS[gt("channel")]
        uc = gt("usecase")
        fps = max(1, gt("fps"))
        size = max(240, gt("size"))
        dmax = max(200, gt("dmax(cm)") * 10)
        cmap_id = cmaps[gt("colormap") % len(cmaps)][1]
        do_detect = gt("detect") == 1
        show_only = gt("show only") == 1
        conf = max(0.01, gt("conf%") / 100.0)

        if (uc, fps) != last_ctrl:
            try:
                link.send_ctrl(uc, fps)
            except Exception:  # noqa: BLE001
                pass
            last_ctrl = (uc, fps)

        try:
            hd = link.header()
        except Exception:  # noqa: BLE001
            hd = None

        if not hd or hd["magic"] != TL.MAGIC or hd["w"] == 0 or hd["w"] * hd["h"] > TL.MAX_PX:
            blank = np.zeros((size * 3 // 4, size, 3), np.uint8)
            fsm = hd["fsm"] if hd else "?"
            hud(blank, ["no frame yet", f"fsm={fsm}  power-cycle the board?"])
            cv2.imshow("tof", blank)
        else:
            w, h = hd["w"], hd["h"]
            out_w = size
            out_h = int(round(size * h / w))
            try:
                chan = link.channel(ch, w, h)
            except Exception:  # noqa: BLE001
                cv2.waitKey(30); continue

            if hd["seq"] != last_seq:
                now = time.time(); dt = now - t_prev; t_prev = now
                if dt > 0:
                    fps_meas = 0.9 * fps_meas + 0.1 / dt
                last_seq = hd["seq"]

            col = render(chan, ch, dmax, out_w, out_h, cmap_id)

            dets = []
            if do_detect:
                if hasattr(detector, "conf"):
                    detector.conf = conf
                try:
                    dist, amp, amb = link.all_channels(w, h)
                    dets = detector.detect(dist, amp, amb)
                    det_err = None
                except Exception as e:  # noqa: BLE001
                    dets, det_err = [], str(e)
                col = draw_detections(col, dets, w, h, show_only)

            lines = [f"{w}x{h}  usecase {hd['usecase']} bin{hd['binning']}  {ch}",
                     f"{fps_meas:4.1f} fps (set {fps})  temp {hd['temp']}  dropped {hd['dropped']}"]
            if do_detect:
                if det_err:
                    lines.append(f"detector error: {det_err[:28]}")
                elif dets:
                    from collections import Counter
                    c = Counter(d.label for d in dets)
                    lines.append("objects: " + ", ".join(f"{v}x {k}" for k, v in c.items())[:40])
                else:
                    lines.append("objects: none")
            hud(col, lines)
            cv2.imshow("tof", col)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if cv2.getWindowProperty("tof", cv2.WND_PROP_VISIBLE) < 1:
            break

    session.close()
    cv2.destroyAllWindows()
    print("bye")


if __name__ == "__main__":
    main()
