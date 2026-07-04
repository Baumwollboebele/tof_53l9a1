#!/usr/bin/env python3
"""
Live depth viewer for the NUCLEO-H533RE + X-NUCLEO-53L9A1 (VL53L9CX) firmware.

The firmware (53L9A1_LiveDepth) streams one binary packet per processed depth
frame over the ST-Link virtual COM port. This script reads that stream and
renders the depth map live as a color image.

Packet layout (little-endian):
    offset 0  : magic  b'L9DF'
    offset 4  : uint16 width
    offset 6  : uint16 height
    offset 8  : uint16 format        (0 = uint16 depth in millimetres)
    offset 10 : uint32 frame_counter
    offset 14 : width*height * uint16 depth values (mm); 0 = invalid / no target

Usage:
    python live_view.py --port COM7            (Windows)
    python live_view.py --port /dev/ttyACM0    (Linux)
    python live_view.py                        (auto-detect ST-Link VCOM)

Requirements: pyserial, numpy, matplotlib  (see requirements.txt)
"""

import argparse
import struct
import sys
import threading
import time

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("pyserial is required:  pip install pyserial")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation


def get_cmap(name):
    """Colormap lookup that works across matplotlib versions."""
    try:
        return matplotlib.colormaps[name].copy()          # matplotlib >= 3.5
    except (AttributeError, KeyError):
        from matplotlib import cm
        return cm.get_cmap(name).copy()                    # older matplotlib

MAGIC = b"L9DF"
HEADER_FMT = "<HHHI"  # width, height, format, frame_counter (after the 4-byte magic)
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 10 bytes


def auto_detect_port():
    """Return the most likely ST-Link VCOM port, or None."""
    candidates = list(serial.tools.list_ports.comports())
    for p in candidates:
        text = " ".join(str(x) for x in (p.description, p.manufacturer, p.product) if x).lower()
        if "st-link" in text or "stlink" in text or "stm" in text:
            return p.device
    # fall back to the first available port
    return candidates[0].device if candidates else None


class FrameReader(threading.Thread):
    """Background thread: parse the serial stream and keep the latest frame."""

    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=1)
        self.lock = threading.Lock()
        self.latest = None          # (frame_counter, np.ndarray[h, w] uint16)
        self.running = True
        self._buf = bytearray()

    def _read_exact(self, n):
        """Read exactly n bytes (or raise on timeout/closed)."""
        while len(self._buf) < n:
            chunk = self.ser.read(max(n - len(self._buf), 256))
            if not chunk:
                if not self.running:
                    raise IOError("stopped")
                continue
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def _sync_to_magic(self):
        """Consume bytes until the 4-byte magic has been seen."""
        window = bytearray()
        while True:
            b = self._read_exact(1)
            window += b
            if len(window) > 4:
                del window[0]
            if window == MAGIC:
                return

    def run(self):
        try:
            while self.running:
                self._sync_to_magic()
                header = self._read_exact(HEADER_SIZE)
                width, height, fmt, counter = struct.unpack(HEADER_FMT, header)
                if fmt != 0 or not (0 < width <= 256 and 0 < height <= 256):
                    # bad header, resync
                    continue
                payload = self._read_exact(width * height * 2)
                frame = np.frombuffer(payload, dtype="<u2").reshape(height, width)
                with self.lock:
                    self.latest = (counter, frame)
        except Exception as exc:  # noqa: BLE001
            print(f"reader stopped: {exc}")

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser(description="VL53L9CX live depth viewer")
    ap.add_argument("--port", default=None, help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=921600, help="baud rate (must match firmware)")
    ap.add_argument("--min", type=int, default=0, help="min depth (mm) for the color scale")
    ap.add_argument("--max", type=int, default=2000, help="max depth (mm) for the color scale")
    ap.add_argument("--cmap", default="turbo", help="matplotlib colormap")
    args = ap.parse_args()

    port = args.port or auto_detect_port()
    if not port:
        sys.exit("No serial port found. Plug in the board or pass --port.")
    print(f"Opening {port} @ {args.baud} baud ...")

    reader = FrameReader(port, args.baud)
    reader.start()

    # wait for the first frame to learn the resolution
    print("Waiting for first frame ...")
    first = None
    t0 = time.time()
    while first is None:
        first = reader.get()
        if time.time() - t0 > 10:
            reader.close()
            sys.exit("No frames received. Check baud rate, wiring and that the board is running.")
        time.sleep(0.05)

    _, frame0 = first
    h, w = frame0.shape
    print(f"Streaming {w}x{h} depth frames. Close the window to quit.")

    cmap = get_cmap(args.cmap)
    cmap.set_bad(color="black")  # invalid pixels (0) shown black

    plt.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    try:
        fig.canvas.manager.set_window_title("VL53L9CX live depth")
    except Exception:  # noqa: BLE001  (not supported on every backend)
        pass
    masked0 = np.ma.masked_equal(frame0, 0)
    img = ax.imshow(masked0, cmap=cmap, vmin=args.min, vmax=args.max, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("distance (mm)")
    title = ax.set_title("frame -   |   -- fps")

    state = {"last_counter": -1, "t": time.time(), "fps": 0.0}

    def update(_):
        item = reader.get()
        if item is None:
            return (img, title)
        counter, frame = item
        masked = np.ma.masked_equal(frame, 0)
        img.set_data(masked)
        now = time.time()
        dt = now - state["t"]
        if dt > 0:
            state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / dt)
        state["t"] = now
        title.set_text(f"frame {counter}   |   {state['fps']:4.1f} fps")
        return (img, title)

    ani = matplotlib.animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        reader.close()
        print("bye")


if __name__ == "__main__":
    main()
