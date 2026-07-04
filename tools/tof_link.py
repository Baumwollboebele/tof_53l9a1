"""
Shared SWD link to the firmware's g_live / g_ctrl mailboxes (via pyOCD).

Used by the OpenCV viewer (tof_cv.py). Mailbox layout mirrors live_mailbox_t /
live_ctrl_t in vl53l9_app.c.
"""

import struct
import subprocess

import numpy as np

MAGIC = 0x4556494C
MAX_W, MAX_H = 54, 42
MAX_PX = MAX_W * MAX_H

HDR_FMT = "<13I"
HDR_U32 = struct.calcsize(HDR_FMT)      # 52
OFF_WH = HDR_U32                        # width,height (u16,u16)
HDR_SIZE = OFF_WH + 4                   # 56
OFF_DIST = HDR_SIZE
OFF_AMP = OFF_DIST + MAX_PX * 2
OFF_AMB = OFF_AMP + MAX_PX * 2
CHAN_OFF = {"distance": OFF_DIST, "amplitude": OFF_AMP, "ambient": OFF_AMB}

_HDR_KEYS = ["magic", "hb", "fsm", "frames", "seq", "dropped", "applied",
             "usecase", "binning", "fp", "exp", "temp", "fc"]


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


def connect(target="stm32h533retx", elf=None, gl=None, gc=None, freq=4_000_000):
    """Attach to the running target (no halt) and resolve g_live/g_ctrl.
    Returns (session, Link)."""
    from pyocd.core.helpers import ConnectHelper

    if gl is None and elf:
        gl = find_symbol_addr(elf, "g_live")
    if gc is None and elf:
        gc = find_symbol_addr(elf, "g_ctrl")
    if gl is None or gc is None:
        raise RuntimeError("could not resolve g_live/g_ctrl (pass --gl/--gc or a valid --elf)")

    session = ConnectHelper.session_with_chosen_probe(
        target_override=target, options={"connect_mode": "attach", "frequency": freq})
    if session is None:
        raise RuntimeError("no debug probe found (connect the board via ST-Link USB)")
    session.open()
    tgt = session.target
    try:
        if tgt.get_state().name == "HALTED":
            tgt.resume()
    except Exception:  # noqa: BLE001
        pass
    return session, Link(tgt, gl, gc)


class Link:
    def __init__(self, target, gl, gc):
        self.t = target
        self.gl = gl
        self.gc = gc
        self._cmd = None

    def _rd(self, addr, n):
        return bytes(self.t.read_memory_block8(addr, n))

    def header(self):
        b = self._rd(self.gl, HDR_SIZE)
        f = struct.unpack(HDR_FMT, b[:HDR_U32])
        w, h = struct.unpack("<HH", b[OFF_WH:OFF_WH + 4])
        d = dict(zip(_HDR_KEYS, f))
        d["w"], d["h"] = w, h
        return d

    def channel(self, name, w, h):
        n = w * h
        raw = self._rd(self.gl + CHAN_OFF[name], n * 2)
        return (np.frombuffer(raw, dtype="<u2") & 0x7FFF).reshape(h, w)

    def all_channels(self, w, h):
        return (self.channel("distance", w, h),
                self.channel("amplitude", w, h),
                self.channel("ambient", w, h))

    def send_ctrl(self, usecase, fps, exposure_ms=0):
        if self._cmd is None:
            self._cmd = struct.unpack("<I", self._rd(self.gc, 4))[0]
        self.t.write_memory(self.gc + 4, usecase & 0xFFFFFFFF)
        self.t.write_memory(self.gc + 8, int(1_000_000 / max(fps, 1)))
        self.t.write_memory(self.gc + 12, int(exposure_ms))
        self._cmd += 1
        self.t.write_memory(self.gc + 0, self._cmd & 0xFFFFFFFF)
