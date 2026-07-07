#!/usr/bin/env python3
"""
Replay a raw ToF capture (.npz from tof_gui's "Record raw frames") through the
3-D mapper offline - no hardware needed. Prints tracking health per frame and
can write the fused mesh, for tuning mapper parameters and regression runs.

Usage:
    python replay.py capture.npz
    python replay.py capture.npz --out map.ply --voxel 0.015 --depth-max 1.5
"""

import argparse
import time

import numpy as np

from mapper import DepthMapper


def main():
    ap = argparse.ArgumentParser(description="Offline 3-D mapper replay")
    ap.add_argument("npz", help="raw capture from tof_gui (Record raw frames)")
    ap.add_argument("--out", default=None, help="write the fused mesh (.ply)")
    ap.add_argument("--fov", type=float, default=None,
                    help="override horizontal FoV (deg); default: ST model")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--depth-max", type=float, default=4.0)
    ap.add_argument("--min-fitness", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    args = ap.parse_args()

    data = np.load(args.npz)
    dist, amp = data["dist"], data["amp"]
    binning = int(data["binning"]) if "binning" in data else 2
    n = len(dist)
    print(f"{args.npz}: {n} frames {dist.shape[2]}x{dist.shape[1]} "
          f"binning {binning}")

    m = DepthMapper(hfov_deg=args.fov, voxel=args.voxel,
                    depth_max=args.depth_max, min_fitness=args.min_fitness)
    t0 = time.perf_counter()
    ok = 0
    for i in range(0, n, args.every):
        good = m.add_frame(dist[i], amp[i], binning)
        ok += int(good)
        if i % (20 * args.every) == 0 or not good:
            print(f"[{i:4d}] {'ok  ' if good else 'LOST'} "
                  f"fit {m.fitness:.2f}  map {len(m.vox)} vox  "
                  f"kf {len(m.keyframes)}")
    dt = time.perf_counter() - t0
    used = len(range(0, n, args.every))
    print(f"\n{ok}/{used} frames tracked in {dt:.1f}s "
          f"({used / max(dt, 1e-9):.1f} fps)  "
          f"map {len(m.vox)} vox  kf {len(m.keyframes)}")
    if args.out:
        print(m.finish(args.out))


if __name__ == "__main__":
    main()
