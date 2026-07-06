"""
Incremental 3-D mapping ("Record 3D map") from the ToF depth stream.

KinectFusion-style pipeline, scaled down to the VL53L9CX and implemented
natively (numpy + scipy KD-tree + scikit-image marching cubes; open3d has no
Python 3.13 wheel on Windows):

    depth frame -> denoise gates -> point cloud -> point-to-plane ICP against
    the accumulated MAP (frame-to-model tracking, constant-velocity guess)
    -> pose chain -> weighted voxel fusion of the world cloud; keyframes are
    fused into a TSDF volume on finish(), and a triangle mesh is extracted
    (marching cubes) and written as .ply.

Noise handling (the raw sensor is cm-noisy):
  * amplitude gate       - pixels below a fraction of the frame's p99
                           amplitude are unreliable and dropped
  * flying-pixel filter  - ToF edges produce points mid-air between fore-
                           and background; pixels whose depth jumps vs. a
                           4-neighbour by more than `edge_jump` are dropped
  * statistical outliers - lone points far from their k neighbours are dropped
  * voxel confirmation   - the live map fuses points into voxels with hit
                           counts; only voxels seen `min_cnt`+ times are shown
                           and exported, so single-frame noise never sticks

The sensor reports RADIAL distance per pixel; frames are converted to
perpendicular-Z depth using the same pinhole model as the live 3-D view.

Move the sensor SLOWLY and keep geometry (corners, objects) in view;
`lost`/`fitness` report tracking health, rejected frames don't smear the map.

Requires: scipy (ICP correspondences), scikit-image (meshing; optional -
without it finish() exports the point cloud instead of a mesh).
"""

import math

import numpy as np
from scipy.spatial import cKDTree


def _pinhole(w, h, hfov_deg):
    """fx, fy, cx, cy for the sensor grid (same model as tof_gui's 3D view)."""
    hfov = math.radians(hfov_deg)
    vfov = hfov * h / w
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    fy = (h / 2.0) / math.tan(vfov / 2.0)
    return fx, fy, (w - 1) / 2.0, (h - 1) / 2.0


def radial_to_z(dist_mm, hfov_deg):
    """Radial-distance frame (mm, 0 = invalid) -> perpendicular Z in metres."""
    h, w = dist_mm.shape
    fx, fy, cx, cy = _pinhole(w, h, hfov_deg)
    cc, rr = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    norm = np.sqrt(((cc - cx) / fx) ** 2 + ((rr - cy) / fy) ** 2 + 1.0)
    return np.asarray(dist_mm, np.float32) / norm / 1000.0


def _drop_flying_pixels(z, jump):
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


def _cloud_from_z(z_m, gray_u8, hfov_deg):
    """Perpendicular-Z depth (m) -> (pts Nx3 camera-frame, gray N in 0..1)."""
    h, w = z_m.shape
    fx, fy, cx, cy = _pinhole(w, h, hfov_deg)
    cc, rr = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    v = z_m > 0
    z = z_m[v]
    pts = np.stack([(cc[v] - cx) / fx * z, (rr[v] - cy) / fy * z, z], axis=1)
    return pts.astype(np.float32), gray_u8[v].astype(np.float32) / 255.0


def _voxel_fuse(pts, gray, cnt, voxel):
    """Count-weighted average of points/gray over a voxel grid; counts add up.
    Repeated hits pull a voxel towards the true surface, single hits stay weak."""
    keys = np.floor(pts / voxel).astype(np.int64) + (1 << 20)
    packed = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    _, inv = np.unique(packed, return_inverse=True)
    n = inv.max() + 1
    wsum = np.zeros(n, np.float64)
    np.add.at(wsum, inv, cnt)
    out_p = np.zeros((n, 3), np.float64)
    np.add.at(out_p, inv, pts * cnt[:, None])
    out_g = np.zeros(n, np.float64)
    np.add.at(out_g, inv, gray * cnt)
    out_p /= wsum[:, None]
    out_g /= wsum
    return (out_p.astype(np.float32), out_g.astype(np.float32),
            wsum.astype(np.float32))


def _drop_outliers(pts, gray):
    """Remove points whose mean k-NN distance is far above the median."""
    if len(pts) < 60:
        return pts, gray
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=8)
    md = d[:, 1:].mean(axis=1)
    keep = md < 2.5 * np.median(md)
    return pts[keep], gray[keep]


def _normals(pts, k=12):
    """Per-point normals from local PCA (batched eigendecomposition)."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    nb = pts[idx]                                  # (N,k,3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vec = np.linalg.eigh(cov)                   # ascending eigenvalues
    nrm = vec[:, :, 0].astype(np.float32)
    return nrm, tree


def _icp_point_to_plane(src, dst, dst_nrm, dst_tree, init, max_dist, iters=15):
    """Align src->dst. Returns (4x4 transform, fitness = inlier fraction)."""
    T = init.copy()
    fitness = 0.0
    for _ in range(iters):
        sp = src @ T[:3, :3].T + T[:3, 3]
        dist, idx = dst_tree.query(sp, k=1, distance_upper_bound=max_dist)
        m = np.isfinite(dist)
        fitness = float(m.mean())
        if m.sum() < 20:
            break
        p, q, n = sp[m], dst[idx[m]], dst_nrm[idx[m]]
        r = np.einsum("ni,ni->n", n, p - q)        # signed plane residuals
        A = np.concatenate([np.cross(p, n), n], axis=1)   # (M,6): [w | t]
        try:
            x, *_ = np.linalg.lstsq(A, -r, rcond=None)
        except np.linalg.LinAlgError:
            break
        w, t = x[:3], x[3:]
        c, s = math.cos, math.sin
        rx, ry, rz = w
        R = (np.array([[c(rz), -s(rz), 0], [s(rz), c(rz), 0], [0, 0, 1]]) @
             np.array([[c(ry), 0, s(ry)], [0, 1, 0], [-s(ry), 0, c(ry)]]) @
             np.array([[1, 0, 0], [0, c(rx), -s(rx)], [0, s(rx), c(rx)]]))
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        T = dT @ T
        if np.abs(x).max() < 1e-6:
            break
    return T, fitness


class DepthMapper:
    """Feed depth frames with add_frame(); read the live map with map_points();
    finish() fuses the keyframes into a mesh and writes it to disk."""

    def __init__(self, hfov_deg=60.0, voxel=0.02, depth_max=4.0,
                 icp_dist=0.10, min_fitness=0.25, kf_every=4, max_keyframes=600,
                 amp_frac=0.05, edge_jump=0.06, min_cnt=2, track_cap=40000):
        self.hfov = float(hfov_deg)
        self.voxel = float(voxel)          # live-map / ICP downsample (m)
        self.depth_max = float(depth_max)  # ignore returns beyond this (m)
        self.icp_dist = float(icp_dist)    # ICP correspondence radius (m)
        self.min_fitness = float(min_fitness)
        self.kf_every = int(kf_every)
        self.max_keyframes = int(max_keyframes)
        self.amp_frac = float(amp_frac)    # amplitude gate vs. frame p99
        self.edge_jump = float(edge_jump)  # flying-pixel depth jump (m)
        self.min_cnt = float(min_cnt)      # voxel hits needed to be "real"
        self.track_cap = int(track_cap)    # max points in the tracking model
        self.reset()

    def reset(self):
        self.pose = np.eye(4)              # camera-to-world of the last frame
        self.delta = np.eye(4)             # last frame-to-frame motion
        self.map_pts = np.zeros((0, 3), np.float32)
        self.map_gray = np.zeros(0, np.float32)
        self.map_cnt = np.zeros(0, np.float32)
        self.track = None                  # (pts, normals, kdtree) world frame
        self.keyframes = []                # (z_depth_m, pose, w, h)
        self.frames = 0
        self.tracked = 0
        self.fitness = 1.0
        self.lost = False

    # ---- internals -----------------------------------------------------------
    def _denoise(self, dist_mm, amp):
        """Radial frame -> gated perpendicular-Z image + gray image."""
        h, w = dist_mm.shape
        z = radial_to_z(dist_mm, self.hfov)
        z[(z <= 0) | (z > self.depth_max)] = 0.0
        if amp is not None:
            a = np.asarray(amp, np.float32)
            hi = float(np.percentile(a[a > 0], 99)) if (a > 0).any() else 1.0
            z[a < self.amp_frac * hi] = 0.0     # low confidence -> drop
            gray = np.clip(a * (255.0 / max(hi, 1.0)), 0, 255).astype(np.uint8)
        else:
            gray = np.full((h, w), 180, np.uint8)
        z = _drop_flying_pixels(z, self.edge_jump)
        return z, gray

    def _rebuild_track_model(self):
        """Tracking target = confirmed map voxels (coarse-capped), with normals."""
        m = self.map_cnt >= min(self.min_cnt, self.tracked)
        pts = self.map_pts[m]
        if len(pts) < 100:                  # early frames: use everything
            pts = self.map_pts
        if len(pts) > self.track_cap:
            sel = np.random.default_rng(0).choice(len(pts), self.track_cap,
                                                  replace=False)
            pts = pts[sel]
        if len(pts) < 50:
            self.track = None
            return
        nrm, tree = _normals(pts)
        self.track = (pts, nrm, tree)

    # ---- per-frame ----------------------------------------------------------
    def add_frame(self, dist_mm, amp=None):
        """Register one radial-distance frame (mm). Returns True if it was
        integrated into the map, False if tracking rejected it."""
        self.frames += 1
        z, gray = self._denoise(np.asarray(dist_mm, np.float32), amp)
        pts, g = _cloud_from_z(z, gray, self.hfov)
        if len(pts) < 100:
            self.lost = True
            return False
        pts, g, _ = _voxel_fuse(pts, g, np.ones(len(pts), np.float32), self.voxel)
        pts, g = _drop_outliers(pts, g)
        if len(pts) < 50:
            self.lost = True
            return False

        if self.track is not None:
            tpts, tnrm, ttree = self.track
            init = self.pose @ self.delta          # constant-velocity guess
            T, fit = _icp_point_to_plane(pts, tpts, tnrm, ttree,
                                         init, self.icp_dist)
            self.fitness = fit
            if fit < self.min_fitness:
                self.lost = True
                return False                        # keep pose, drop the frame
            self.delta = np.linalg.inv(self.pose) @ T
            self.pose = T
        self.lost = False
        self.tracked += 1

        world = (pts @ self.pose[:3, :3].T + self.pose[:3, 3]).astype(np.float32)
        self.map_pts = np.concatenate([self.map_pts, world])
        self.map_gray = np.concatenate([self.map_gray, g])
        self.map_cnt = np.concatenate([self.map_cnt,
                                       np.ones(len(world), np.float32)])
        self.map_pts, self.map_gray, self.map_cnt = _voxel_fuse(
            self.map_pts, self.map_gray, self.map_cnt, self.voxel)
        if self.tracked <= 2 or self.tracked % 4 == 0:
            self._rebuild_track_model()

        if (self.tracked - 1) % self.kf_every == 0 and \
                len(self.keyframes) < self.max_keyframes:
            self.keyframes.append((z.copy(), self.pose.copy(),
                                   z.shape[1], z.shape[0]))
        return True

    # ---- outputs ------------------------------------------------------------
    def _confirmed(self):
        m = self.map_cnt >= min(self.min_cnt, max(1, self.tracked))
        return self.map_pts[m], self.map_gray[m]

    def map_points(self):
        """(pts Nx3 float32, rgba Nx4 float32) of the confirmed world cloud."""
        pts, gray = self._confirmed()
        if len(pts) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
        g = np.clip(gray, 0.15, 1.0)[:, None]      # keep dark points visible
        rgba = np.concatenate([g, g, g, np.ones_like(g)], axis=1)
        return pts, rgba.astype(np.float32)

    def _tsdf(self):
        """Fuse keyframes into (tsdf, weight, origin, voxel) uniform grids."""
        pts, _ = self._confirmed()
        vox = self.voxel
        lo = pts.min(axis=0) - 4 * vox
        hi = pts.max(axis=0) + 4 * vox
        # cap the volume at ~12M voxels by coarsening if needed
        while np.prod(np.ceil((hi - lo) / vox)) > 12e6:
            vox *= 1.26
        dims = np.maximum(np.ceil((hi - lo) / vox).astype(int), 2)
        trunc = 3 * vox
        xs, ys, zs = [lo[i] + (np.arange(dims[i]) + 0.5) * vox for i in range(3)]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
        tsdf = np.zeros(len(centers), np.float32)
        wsum = np.zeros(len(centers), np.float32)
        for z_img, pose, w, h in self.keyframes:
            fx, fy, cx, cy = _pinhole(w, h, self.hfov)
            inv = np.linalg.inv(pose)
            pc = centers @ inv[:3, :3].T + inv[:3, 3]
            zc = pc[:, 2]
            m = zc > 1e-3
            u = np.round(pc[m, 0] / zc[m] * fx + cx).astype(int)
            v = np.round(pc[m, 1] / zc[m] * fy + cy).astype(int)
            ib = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d = np.zeros(int(m.sum()), np.float32)
            d[ib] = z_img[v[ib], u[ib]]
            sdf = d - zc[m]
            upd = (d > 0) & (sdf > -trunc)
            val = np.clip(sdf[upd] / trunc, -1.0, 1.0)
            mi = np.where(m)[0][upd]
            tsdf[mi] = (tsdf[mi] * wsum[mi] + val) / (wsum[mi] + 1.0)
            wsum[mi] += 1.0
        # require the same multi-view confirmation as the live map
        known = wsum.reshape(dims) >= min(self.min_cnt, len(self.keyframes))
        return tsdf.reshape(dims), known, lo + 0.5 * vox, vox

    def finish(self, path):
        """TSDF-fuse the keyframes, extract a mesh and write it to `path`
        (.ply). Falls back to the point cloud if meshing is impossible.
        Returns a short status string."""
        if not self.keyframes:
            return "no frames recorded"
        try:
            from skimage import measure
            tsdf, known, origin, vox = self._tsdf()
            verts, faces, vnorm, _ = measure.marching_cubes(
                tsdf, level=0.0, mask=known)
            verts = origin + verts * vox
            _write_ply_mesh(path, verts, faces, vnorm)
            return (f"{len(verts)} verts / {len(faces)} tris from "
                    f"{len(self.keyframes)} keyframes -> {path}")
        except Exception as e:  # noqa: BLE001  (skimage missing, empty surface)
            pc_path = path.rsplit(".", 1)[0] + "_cloud.ply"
            pts, rgba = self.map_points()
            _write_ply_points(pc_path, pts, rgba[:, :3])
            return f"mesh failed ({str(e)[:40]}); wrote cloud {pc_path}"


# ---- minimal binary-PLY writers ---------------------------------------------
def _write_ply_points(path, pts, rgb01):
    col = np.clip(rgb01 * 255, 0, 255).astype(np.uint8)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(pts)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    rec = np.zeros(len(pts), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"], rec["rgb"] = pts.astype("<f4"), col
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(rec.tobytes())


def _write_ply_mesh(path, verts, faces, normals):
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(verts)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property float nx\nproperty float ny\nproperty float nz\n"
           f"element face {len(faces)}\n"
           "property list uchar int vertex_indices\n"
           "end_header\n")
    vrec = np.zeros(len(verts), dtype=[("v", "<f4", 3), ("n", "<f4", 3)])
    vrec["v"], vrec["n"] = verts.astype("<f4"), normals.astype("<f4")
    frec = np.zeros(len(faces), dtype=[("k", "u1"), ("idx", "<i4", 3)])
    frec["k"], frec["idx"] = 3, faces.astype("<i4")
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(vrec.tobytes())
        f.write(frec.tobytes())
