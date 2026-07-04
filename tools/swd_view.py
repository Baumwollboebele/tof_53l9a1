#!/usr/bin/env python3
"""
Live depth viewer that reads frames straight out of the target's RAM over the
SWD debug link (via pyOCD) - no UART needed.

The firmware (53L9A1_LiveDepth) publishes each processed depth frame into a
fixed global mailbox `g_live`. This script finds that symbol's address from the
.elf, then continuously reads the mailbox over SWD (non-intrusively, without
halting the CPU) and renders the depth map live.

Mailbox layout (see vl53l9_app.c, little-endian):
    0  u32 magic 'LIVE' (0x4556494C)
    4  u32 heartbeat
    8  u32 fsm            (1=ready_to_boot, 2=standby, 3=streaming)
    12 u32 frame_ready
    16 u32 frames
    20 u32 seq            (bumped on every new frame)
    24 u32 dropped
    28 u16 width
    30 u16 height
    32 u16 data[width*height]   (depth in mm, 0 = invalid)

Usage:
    python swd_view.py                       # auto: elf=build/..., target auto
    python swd_view.py --elf path/to.elf
    python swd_view.py --addr 0x20000050 --width 24 --height 20

Requirements: pyocd, numpy, matplotlib  (pip install pyocd numpy matplotlib)
"""

import argparse
import os
import re
import struct
import subprocess
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation

MAGIC = 0x4556494C
HDR = "<IIIIIIIHH"          # up to and including width,height
HDR_SIZE = struct.calcsize(HDR)   # 32
MAX_PIXELS = 24 * 24


def find_symbol_addr(elf, name):
    """Return the address of `name` in `elf` using nm, or None."""
    for nm in ("arm-none-eabi-nm", "nm"):
        try:
            out = subprocess.check_output([nm, elf], stderr=subprocess.DEVNULL).decode()
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2] == name:
                return int(parts[0], 16)
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

    ap = argparse.ArgumentParser(description="VL53L9CX live depth viewer over SWD")
    ap.add_argument("--elf", default=default_elf, help="ELF to locate g_live symbol")
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=None,
                    help="g_live address (overrides --elf lookup)")
    ap.add_argument("--target", default="stm32h533retx")
    ap.add_argument("--min", type=int, default=0, help="color scale min (mm)")
    ap.add_argument("--max", type=int, default=2500, help="color scale max (mm)")
    ap.add_argument("--cmap", default="turbo")
    args = ap.parse_args()

    addr = args.addr
    if addr is None:
        if not os.path.exists(args.elf):
            sys.exit(f"ELF not found: {args.elf}\nBuild first, or pass --addr.")
        addr = find_symbol_addr(args.elf, "g_live")
        if addr is None:
            sys.exit("Could not find symbol 'g_live' (need arm-none-eabi-nm on PATH), "
                     "or pass --addr 0x........")
    print(f"g_live @ 0x{addr:08x}")

    try:
        from pyocd.core.helpers import ConnectHelper
    except ImportError:
        sys.exit("pyocd is required:  pip install pyocd")

    # connect_mode "attach" joins the running target without halting/resetting it
    session = ConnectHelper.session_with_chosen_probe(
        target_override=args.target,
        options={"connect_mode": "attach", "frequency": 4000000})
    if session is None:
        sys.exit("No debug probe found. Is the board connected via ST-Link USB?")
    session.open()
    target = session.target
    try:
        if target.get_state().name == "HALTED":
            target.resume()   # keep the firmware running while we read RAM
    except Exception:  # noqa: BLE001
        pass

    def read(n):
        return bytes(target.read_memory_block8(addr, n))

    # sanity: check magic
    hdr = read(HDR_SIZE)
    magic = struct.unpack("<I", hdr[:4])[0]
    if magic != MAGIC:
        print(f"WARNING: magic=0x{magic:08x} (expected 0x{MAGIC:08x}). "
              "Wrong address or firmware not running?")

    print("Reading frames over SWD. Close the window to quit.")
    cmap = get_cmap(args.cmap)
    cmap.set_bad(color="black")

    plt.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots(figsize=(6, 6))
    dummy = np.zeros((20, 24), dtype=np.uint16)
    img = ax.imshow(np.ma.masked_equal(dummy, 0), cmap=cmap,
                    vmin=args.min, vmax=args.max, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("distance (mm)")
    title = ax.set_title("waiting for frames ...")

    state = {"seq": -1, "shape": None}

    def update(_):
        try:
            hdr = read(HDR_SIZE)
        except Exception:  # noqa: BLE001 - transient SWD read
            return (img, title)
        (magic, hb, fsm, fr, frames, seq, dropped, w, h) = struct.unpack(HDR, hdr)
        if magic != MAGIC or w == 0 or h == 0 or w * h > MAX_PIXELS:
            title.set_text(f"no valid frame (magic=0x{magic:08x})")
            return (img, title)
        if seq == state["seq"]:
            return (img, title)          # nothing new
        state["seq"] = seq
        try:
            raw = read(HDR_SIZE + w * h * 2)[HDR_SIZE:]
        except Exception:  # noqa: BLE001
            return (img, title)
        frame = np.frombuffer(raw, dtype="<u2").reshape(h, w)
        if state["shape"] != (h, w):
            img.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
            state["shape"] = (h, w)
        img.set_data(np.ma.masked_equal(frame, 0))
        title.set_text(f"frame {frames}  fsm={fsm}  dropped={dropped}")
        return (img, title)

    ani = matplotlib.animation.FuncAnimation(
        fig, update, interval=33, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        session.close()
        print("bye")


if __name__ == "__main__":
    main()
