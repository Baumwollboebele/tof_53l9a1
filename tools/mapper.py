"""
Incremental 3-D mapping ("Record 3D map") from the ToF depth stream.

KinectFusion-style pipeline, scaled down to the VL53L9CX and implemented
natively (numpy + scipy KD-tree + scikit-image marching cubes; open3d has no
Python 3.13 wheel on Windows):

    depth frame -> denoise gates -> organized vertex/normal maps ->
    projective point-to-plane ICP against the confirmed voxel map
    (frame-to-model tracking, constant-velocity guess, Huber weights)
    -> pose chain -> incremental voxel-hash fusion; keyframes are fused
    into a TSDF volume on finish(), and a triangle mesh is extracted
    (marching cubes) and written as .ply.

Tracking is projective: confirmed map voxels inside the predicted camera
frustum are projected into the current depth image for correspondences
(O(N), no KD-tree). A KD-tree point-to-plane ICP survives as the recovery
path when projection finds too few inliers; after that the frame is
rejected (`lost`) and the next frame retries from the last good pose.

Noise handling (the raw sensor is cm-noisy):
  * amplitude gate       - pixels below a fraction of the frame's p99
                           amplitude are unreliable and dropped
  * flying-pixel filter  - ToF edges produce points mid-air between fore-
                           and background; pixels whose depth jumps vs. a
                           4-neighbour by more than `edge_jump` are dropped
  * statistical outliers - lone points far from their k neighbours are dropped
  * voxel confirmation   - the map fuses points into voxels with hit counts;
                           only voxels seen `min_cnt`+ times are shown and
                           exported, so single-frame noise never sticks

Geometry uses the ST optical model ported in tof3d.py (real focal length,
distortion and parallax from vl53l9-transform-c) unless an explicit
hfov_deg override is given.

Move the sensor SLOWLY and keep geometry (corners, objects) in view;
`lost`/`fitness` report tracking health, rejected frames don't smear the map.

Requires: scipy (outlier filter + recovery ICP), scikit-image (meshing;
optional - without it finish() exports the point cloud instead of a mesh).
"""

import math

import numpy as np
from scipy.spatial import cKDTree

import tof3d


# ---- incremental voxel-hash map ----------------------------------------------
class VoxelMap:
    """Count-weighted voxel fusion with O(frame) updates.

    Accumulators live in preallocated numpy arrays (amortized growth); a plain
    dict maps packed voxel keys to rows. Repeated hits pull a voxel towards
    the true surface, single hits stay weak until confirmed."""

    _OFF = 1 << 20          # centre the 21-bit-per-axis key packing

    def __init__(self, voxel, cap=1 << 15):
        self.voxel = float(voxel)
        self.n = 0
        self.frame = 0
        self.index = {}
        self._alloc(cap)

    def _alloc(self, cap):
        self.keys = np.zeros(cap, np.int64)
        self.sum_xyz = np.zeros((cap, 3), np.float64)
        self.sum_g = np.zeros(cap, np.float64)
        self.cnt = np.zeros(cap, np.float64)
        self.last_seen = np.zeros(cap, np.int32)

    def _grow(self, need):
        cap = max(2 * len(self.keys), self.n + need)
        old = (self.keys, self.sum_xyz, self.sum_g, self.cnt, self.last_seen)
        self._alloc(cap)
        n = self.n
        for dst, src in zip((self.keys, self.sum_xyz, self.sum_g, self.cnt,
                             self.last_seen), old):
            dst[:n] = src[:n]

    def __len__(self):
        return self.n

    def integrate(self, pts, gray, weights=None):
        self.frame += 1
        if len(pts) == 0:
            return
        k = np.floor(pts / self.voxel).astype(np.int64) + self._OFF
        packed = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
        uniq, inv = np.unique(packed, return_inverse=True)
        w = np.ones(len(pts), np.float64) if weights is None \
            else np.asarray(weights, np.float64)
        m = len(uniq)
        wsum = np.zeros(m, np.float64)
        np.add.at(wsum, inv, w)
        psum = np.zeros((m, 3), np.float64)
        np.add.at(psum, inv, pts * w[:, None])
        gsum = np.zeros(m, np.float64)
        np.add.at(gsum, inv, gray * w)

        idx = self.index
        rows = np.fromiter((idx.get(key, -1) for key in uniq.tolist()),
                           np.int64, m)
        new = rows < 0
        er = rows[~new]
        self.sum_xyz[er] += psum[~new]
        self.sum_g[er] += gsum[~new]
        self.cnt[er] += wsum[~new]
        self.last_seen[er] = self.frame
        nn = int(new.sum())
        if nn:
            if self.n + nn > len(self.keys):
                self._grow(nn)
            rr = np.arange(self.n, self.n + nn)
            self.keys[rr] = uniq[new]
            self.sum_xyz[rr] = psum[new]
            self.sum_g[rr] = gsum[new]
            self.cnt[rr] = wsum[new]
            self.last_seen[rr] = self.frame
            for key, r in zip(uniq[new].tolist(), rr.tolist()):
                idx[key] = r
            self.n += nn

    def confirmed(self, min_cnt):
        """(pts Nx3 f32, gray N f32) of voxels with at least min_cnt hits."""
        rows = np.where(self.cnt[:self.n] >= min_cnt)[0]
        c = self.cnt[rows]
        pts = (self.sum_xyz[rows] / c[:, None]).astype(np.float32)
        gray = (self.sum_g[rows] / c).astype(np.float32)
        return pts, gray

    def compact(self, min_cnt, stale):
        """Drop unconfirmed voxels not hit for `stale` frames; rebuild index."""
        n = self.n
        keep = (self.cnt[:n] >= min_cnt) | \
               (self.frame - self.last_seen[:n] <= stale)
        m = int(keep.sum())
        if m == n:
            return
        for arr in (self.keys, self.sum_xyz, self.sum_g, self.cnt,
                    self.last_seen):
            arr[:m] = arr[:n][keep]
        self.n = m
        self.index = dict(zip(self.keys[:m].tolist(), range(m)))


# ---- per-frame helpers --------------------------------------------------------
def _voxel_fuse(pts, gray, cnt, voxel):
    """Count-weighted average of points/gray over a voxel grid (one shot;
    used to downsample a single frame before tracking/integration)."""
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


def grid_normals(vmap, valid):
    """Per-pixel normals of an organized vertex map via central differences.
    Returns (normals (h,w,3) f32 oriented towards the camera, ok mask)."""
    h, w, _ = vmap.shape
    nrm = np.zeros((h, w, 3), np.float32)
    ok = np.zeros((h, w), bool)
    if h < 3 or w < 3:
        return nrm, ok
    du = vmap[1:-1, 2:] - vmap[1:-1, :-2]
    dv = vmap[2:, 1:-1] - vmap[:-2, 1:-1]
    cr = np.cross(du, dv)
    ln = np.linalg.norm(cr, axis=-1)
    good = (ln > 1e-12) & valid[1:-1, 1:-1] & \
        valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    cr /= np.maximum(ln, 1e-12)[..., None]
    flip = np.einsum("ijk,ijk->ij", cr, vmap[1:-1, 1:-1]) > 0
    cr[flip] *= -1.0
    nrm[1:-1, 1:-1] = cr
    ok[1:-1, 1:-1] = good
    return nrm, ok


def _normals(pts, k=12):
    """Per-point normals from local PCA (recovery-ICP path only)."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    nb = pts[idx]                                  # (N,k,3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vec = np.linalg.eigh(cov)                   # ascending eigenvalues
    nrm = vec[:, :, 0].astype(np.float32)
    return nrm, tree


# ---- ICP ----------------------------------------------------------------------
def _se3(x):
    """Small-motion 6-vector [wx wy wz tx ty tz] -> 4x4 transform."""
    c, s = math.cos, math.sin
    rx, ry, rz = x[:3]
    R = (np.array([[c(rz), -s(rz), 0], [s(rz), c(rz), 0], [0, 0, 1]]) @
         np.array([[c(ry), 0, s(ry)], [0, 1, 0], [-s(ry), 0, c(ry)]]) @
         np.array([[1, 0, 0], [0, c(rx), -s(rx)], [0, s(rx), c(rx)]]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = x[3:]
    return T


def _icp_projective(src_world, model, vmap, nmap, valid, init_pose,
                    gate=0.07, iters=6, huber=0.025):
    """Align map points onto the current organized depth frame.

    src_world: Nx3 confirmed map points (world), vmap/nmap: (h,w,3) camera-
    frame vertex/normal images, valid: (h,w) usable-pixel mask, init_pose:
    camera-to-world guess. Correspondence = projection into the image
    (nearest pixel), point-to-plane solve with Huber/IRLS weights.
    Returns (pose camera-to-world, fitness = inlier fraction of src)."""
    W = np.linalg.inv(init_pose)                   # world -> camera
    h, w = valid.shape
    fitness = 0.0
    for _ in range(iters):
        pc = src_world @ W[:3, :3].T + W[:3, 3]
        zc = pc[:, 2]
        u, v = model.project(pc)
        finite = np.isfinite(u) & np.isfinite(v) & (zc > 1e-3)
        ui = np.zeros(len(pc), np.int64)
        vi = np.zeros(len(pc), np.int64)
        ui[finite] = np.rint(u[finite]).astype(np.int64)
        vi[finite] = np.rint(v[finite]).astype(np.int64)
        inb = finite & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        uic = np.clip(ui, 0, w - 1)
        vic = np.clip(vi, 0, h - 1)
        ok = inb & valid[vic, uic]
        q = vmap[vic, uic]
        n = nmap[vic, uic]
        d = pc - q
        ok &= np.einsum("ni,ni->n", d, d) < gate * gate
        fitness = float(ok.mean())
        if int(ok.sum()) < 20:
            break
        p, qq, nn = pc[ok], q[ok], n[ok]
        r = np.einsum("ni,ni->n", nn, p - qq)      # signed plane residuals
        wgt = np.sqrt(np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-9)))
        A = np.concatenate([np.cross(p, nn), nn], axis=1) * wgt[:, None]
        try:
            x, *_ = np.linalg.lstsq(A, -r * wgt, rcond=None)
        except np.linalg.LinAlgError:
            break
        W = _se3(x) @ W
        if np.abs(x).max() < 1e-6:
            break
    return np.linalg.inv(W), fitness


def _icp_point_to_plane(src, dst, dst_nrm, dst_tree, init, max_dist, iters=15):
    """KD-tree recovery ICP: align src->dst. Returns (4x4, inlier fraction)."""
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
        r = np.einsum("ni,ni->n", n, p - q)
        A = np.concatenate([np.cross(p, n), n], axis=1)
        try:
            x, *_ = np.linalg.lstsq(A, -r, rcond=None)
        except np.linalg.LinAlgError:
            break
        T = _se3(x) @ T
        if np.abs(x).max() < 1e-6:
            break
    return T, fitness


class DepthMapper:
    """Feed depth frames with add_frame(); read the live map with map_points();
    finish() fuses the keyframes into a mesh and writes it to disk.

    Thread note: not internally locked - drive each instance from one thread
    (the GUI runs it inside MappingWorker)."""

    def __init__(self, hfov_deg=None, voxel=0.02, depth_max=4.0,
                 icp_dist=0.07, min_fitness=0.25, kf_every=4, max_keyframes=600,
                 amp_frac=0.05, edge_jump=0.06, min_cnt=2, track_cap=5000,
                 parallax=True):
        self.hfov = hfov_deg               # None = ST optical model
        self.voxel = float(voxel)          # map / per-frame downsample (m)
        self.depth_max = float(depth_max)  # ignore returns beyond this (m)
        self.icp_dist = float(icp_dist)    # projective correspondence gate (m)
        self.min_fitness = float(min_fitness)
        self.kf_every = int(kf_every)
        self.max_keyframes = int(max_keyframes)
        self.amp_frac = float(amp_frac)    # amplitude gate vs. frame p99
        self.edge_jump = float(edge_jump)  # flying-pixel depth jump (m)
        self.min_cnt = float(min_cnt)      # voxel hits needed to be "real"
        self.track_cap = int(track_cap)    # max map points fed to ICP
        self.parallax = bool(parallax)
        self.reset()

    def reset(self):
        self.pose = np.eye(4)              # camera-to-world of the last frame
        self.delta = np.eye(4)             # last frame-to-frame motion
        self.vox = VoxelMap(self.voxel)
        self.keyframes = []                # (z_depth_m, pose, model)
        self.frames = 0
        self.tracked = 0
        self.fitness = 1.0
        self.lost = False

    # ---- internals -----------------------------------------------------------
    def _confirm_thresh(self):
        return min(self.min_cnt, max(1.0, self.tracked))

    def _frustum_points(self, model, pose_guess):
        """Confirmed map points inside the predicted camera frustum (world)."""
        pts, _ = self.vox.confirmed(self._confirm_thresh())
        if len(pts) < 100:
            pts, _ = self.vox.confirmed(1.0)
        if len(pts) < 100:
            return None
        W = np.linalg.inv(pose_guess)
        pc = pts @ W[:3, :3].T + W[:3, 3]
        front = (pc[:, 2] > 0.05) & (pc[:, 2] < self.depth_max * 1.25)
        u, v = model.project(pc)
        with np.errstate(invalid="ignore"):
            inb = front & np.isfinite(u) & np.isfinite(v) & \
                (u > -2) & (u < model.w + 1) & (v > -2) & (v < model.h + 1)
        sel = pts[inb]
        if len(sel) < 100:                 # bad guess / tiny overlap: use all
            sel = pts
        if len(sel) > self.track_cap:
            pick = np.random.default_rng(0).choice(len(sel), self.track_cap,
                                                   replace=False)
            sel = sel[pick]
        return sel

    def _track(self, model, vmap, nmap, valid, frame_pts):
        """Fallback ladder: projective ICP (cv guess -> last pose -> wide
        gate) then KD-tree recovery ICP. Returns pose or None (lost)."""
        guess = self.pose @ self.delta
        src = self._frustum_points(model, guess)
        if src is None:
            return self.pose               # map too small to track against
        for init, gate in ((guess, self.icp_dist),
                           (self.pose, self.icp_dist),
                           (self.pose, 2 * self.icp_dist)):
            T, fit = _icp_projective(src, model, vmap, nmap, valid, init,
                                     gate=gate)
            if fit >= self.min_fitness:
                self.fitness = fit
                return T
            self.fitness = fit
        nrm, tree = _normals(src)
        T, fit = _icp_point_to_plane(frame_pts, src, nrm, tree, guess,
                                     2 * self.icp_dist)
        if fit >= self.min_fitness:
            self.fitness = fit
            return T
        self.fitness = fit
        return None

    # ---- per-frame ----------------------------------------------------------
    def add_frame(self, dist_mm, amp=None, binning=2):
        """Register one radial-distance frame (mm). Returns True if it was
        integrated into the map, False if tracking rejected it."""
        self.frames += 1
        d = np.asarray(dist_mm, np.float32)
        h, w = d.shape
        model = tof3d.get_model(w, h, binning, self.hfov)
        x, y, z = model.unproject(d, self.parallax)
        z[z > self.depth_max] = 0.0
        if amp is not None:
            z, gray = tof3d.amplitude_gate(z, amp, self.amp_frac)
        else:
            gray = np.full((h, w), 180, np.uint8)
        z = tof3d.drop_flying_pixels(z, self.edge_jump)
        valid = z > 0
        if int(valid.sum()) < 100:
            self.lost = True
            return False

        vmap = np.stack([x, y, z], axis=-1)
        nmap, nok = grid_normals(vmap, valid)
        pts = vmap[valid]
        g = gray[valid].astype(np.float32) / 255.0
        pts, g, _ = _voxel_fuse(pts, g, np.ones(len(pts), np.float32),
                                self.voxel)
        pts, g = _drop_outliers(pts, g)
        if len(pts) < 50:
            self.lost = True
            return False

        if self.tracked > 0:
            T = self._track(model, vmap, nmap, valid & nok, pts)
            if T is None:
                self.lost = True
                return False               # keep pose, drop the frame
            self.delta = np.linalg.inv(self.pose) @ T
            self.pose = T
        self.lost = False
        self.tracked += 1

        world = (pts @ self.pose[:3, :3].T + self.pose[:3, 3]).astype(np.float32)
        self.vox.integrate(world, g)
        if (self.tracked - 1) % self.kf_every == 0 and \
                len(self.keyframes) < self.max_keyframes:
            self.keyframes.append((z.copy(), self.pose.copy(), model))
        if self.tracked % 120 == 0:
            self.vox.compact(self.min_cnt, stale=240)
        return True

    # ---- outputs ------------------------------------------------------------
    def map_points(self):
        """(pts Nx3 float32, rgba Nx4 float32) of the confirmed world cloud."""
        pts, gray = self.vox.confirmed(self._confirm_thresh())
        if len(pts) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
        g = np.clip(gray, 0.15, 1.0)[:, None]      # keep dark points visible
        rgba = np.concatenate([g, g, g, np.ones_like(g)], axis=1)
        return pts, rgba.astype(np.float32)

    def _tsdf(self):
        """Fuse keyframes into (tsdf, known, origin, voxel) uniform grids.
        Each keyframe only touches the sub-grid inside its view frustum."""
        pts, _ = self.vox.confirmed(self._confirm_thresh())
        vox = self.voxel
        lo = pts.min(axis=0) - 4 * vox
        hi = pts.max(axis=0) + 4 * vox
        # cap the volume at ~12M voxels by coarsening if needed
        while np.prod(np.ceil((hi - lo) / vox)) > 12e6:
            vox *= 1.26
        dims = np.maximum(np.ceil((hi - lo) / vox).astype(int), 2)
        trunc = 3 * vox
        origin = lo + 0.5 * vox
        tsdf = np.zeros(int(np.prod(dims)), np.float32)
        wsum = np.zeros(int(np.prod(dims)), np.float32)
        for z_img, pose, model in self.keyframes:
            zmax = float(z_img.max())
            if zmax <= 0:
                continue
            tx, ty = model.tangents()
            corners = []
            for (r, c) in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                corners.append([tx[r, c] * zmax, ty[r, c] * zmax, zmax])
            frust = np.array(corners + [[0.0, 0.0, 0.0]], np.float64)
            fw = frust @ pose[:3, :3].T + pose[:3, 3]
            mn = fw.min(axis=0) - trunc
            mx = fw.max(axis=0) + trunc
            i0 = np.clip(np.floor((mn - origin) / vox).astype(int), 0, dims)
            i1 = np.clip(np.ceil((mx - origin) / vox).astype(int) + 1, 0, dims)
            if (i1 <= i0).any():
                continue
            ax = [origin[k] + np.arange(i0[k], i1[k]) * vox for k in range(3)]
            gx, gy, gz = np.meshgrid(*ax, indexing="ij")
            centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
            ii = [np.arange(i0[k], i1[k]) for k in range(3)]
            flat = ((ii[0][:, None, None] * dims[1] + ii[1][None, :, None])
                    * dims[2] + ii[2][None, None, :]).reshape(-1)

            inv = np.linalg.inv(pose)
            pc = centers @ inv[:3, :3].T + inv[:3, 3]
            zc = pc[:, 2]
            m = zc > 1e-3
            u, v = model.project(pc[m].astype(np.float32))
            ur = np.rint(u).astype(np.int64)
            vr = np.rint(v).astype(np.int64)
            ib = np.isfinite(u) & np.isfinite(v) & \
                (ur >= 0) & (ur < model.w) & (vr >= 0) & (vr < model.h)
            d = np.zeros(int(m.sum()), np.float32)
            d[ib] = z_img[vr[ib], ur[ib]]
            sdf = d - zc[m]
            upd = (d > 0) & (sdf > -trunc)
            val = np.clip(sdf[upd] / trunc, -1.0, 1.0)
            mi = flat[np.where(m)[0][upd]]
            tsdf[mi] = (tsdf[mi] * wsum[mi] + val) / (wsum[mi] + 1.0)
            wsum[mi] += 1.0
        # require the same multi-view confirmation as the live map
        known = wsum.reshape(dims) >= min(self.min_cnt, len(self.keyframes))
        return tsdf.reshape(dims), known, origin, vox

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
