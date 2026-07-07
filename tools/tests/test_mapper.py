"""Tests for the 3-D mapper: voxel-hash fusion, grid normals, and projective
ICP tracking on a synthetic corner scene with a known trajectory."""

import numpy as np
import pytest

import tof3d
from mapper import DepthMapper, VoxelMap, grid_normals


# ---- synthetic scene ----------------------------------------------------------
# axis-aligned corner: front wall z=2, left wall x=-0.5, floor y=+0.8 (Y down)
PLANES = ((2, 2.0), (0, -0.5), (1, 0.8))


def render_corner(model, pose):
    """Radial-distance image (mm) of the corner scene from `pose` (cam->world)."""
    tx, ty = model.tangents()
    dirs = np.stack([tx, ty, np.ones_like(tx)], axis=-1)
    dw = dirs @ pose[:3, :3].T
    o = pose[:3, 3]
    best = np.full(tx.shape, np.inf)
    for axis, val in PLANES:
        denom = dw[..., axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - o[axis]) / denom
        hit = np.isfinite(t) & (t > 0.05)
        best = np.where(hit & (t < best), t, best)
    norm = np.sqrt(tx ** 2 + ty ** 2 + 1.0)
    radial = np.where(np.isfinite(best), best * norm * 1000.0, 0.0)
    return radial.astype(np.float32)


def pose_at(k):
    """Ground-truth pose k: slide along +X at 4 mm/frame, yaw 0.15 deg/frame."""
    a = np.radians(0.15 * k)
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(a), 0, np.sin(a)],
                          [0, 1, 0],
                          [-np.sin(a), 0, np.cos(a)]])
    T[0, 3] = 0.004 * k
    return T


# ---- VoxelMap -----------------------------------------------------------------
def test_voxelmap_accumulates():
    vm = VoxelMap(voxel=0.02)
    pts = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]], np.float32)
    gray = np.array([0.2, 0.8], np.float32)
    vm.integrate(pts, gray)
    vm.integrate(pts, gray)
    assert len(vm) == 2
    p, g = vm.confirmed(2)
    assert len(p) == 2
    assert np.allclose(sorted(g.tolist()), [0.2, 0.8])
    assert np.allclose(p[np.argsort(p[:, 0])], pts, atol=1e-6)
    # single-hit voxels are not confirmed yet
    vm.integrate(np.array([[1.0, 1.0, 1.0]], np.float32),
                 np.array([0.5], np.float32))
    assert len(vm.confirmed(2)[0]) == 2
    assert len(vm) == 3


def test_voxelmap_same_voxel_merges():
    vm = VoxelMap(voxel=0.1)
    pts = np.array([[0.01, 0.0, 0.0], [0.03, 0.0, 0.0]], np.float32)
    vm.integrate(pts, np.array([0.0, 1.0], np.float32))
    assert len(vm) == 1
    p, g = vm.confirmed(2)
    assert p[0, 0] == pytest.approx(0.02, abs=1e-6)
    assert g[0] == pytest.approx(0.5, abs=1e-6)


def test_voxelmap_growth_and_compact():
    vm = VoxelMap(voxel=0.02, cap=16)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (500, 3)).astype(np.float32)
    vm.integrate(pts, np.full(500, 0.5, np.float32))     # forces _grow
    n1 = len(vm)
    assert n1 > 16
    # age the single-hit voxels out, keep one confirmed voxel alive
    keep = np.array([[0.0, 0.0, 0.0]], np.float32)
    for _ in range(3):
        vm.integrate(keep, np.array([0.5], np.float32))
    vm.frame += 1000
    vm.compact(min_cnt=2, stale=240)
    assert len(vm) < n1
    p, _ = vm.confirmed(2)
    assert len(p) == 1
    # the rebuilt index still works
    vm.integrate(keep, np.array([0.5], np.float32))
    assert len(vm) == len(p)


# ---- grid normals -------------------------------------------------------------
def test_grid_normals_wall():
    m = tof3d.get_model(54, 42, binning=2)
    tx, ty = m.tangents()
    z = np.full(tx.shape, 1.0, np.float32)
    vmap = np.stack([tx * z, ty * z, z], axis=-1)
    nrm, ok = grid_normals(vmap, z > 0)
    assert ok[1:-1, 1:-1].all()
    # fronto-parallel wall: normals point back at the camera (-Z)
    assert np.allclose(nrm[ok][:, 2], -1.0, atol=1e-3)
    assert not ok[0].any() and not ok[-1].any()          # borders invalid


# ---- end-to-end tracking ------------------------------------------------------
def test_mapper_static_scene():
    model = tof3d.get_model(54, 42, binning=2)
    m = DepthMapper(parallax=False)
    img = render_corner(model, np.eye(4))
    for _ in range(6):
        assert m.add_frame(img, None, binning=2)
    assert m.tracked == 6
    assert not m.lost
    assert m.fitness > 0.8
    assert np.allclose(m.pose, np.eye(4), atol=5e-3)
    pts, rgba = m.map_points()
    assert len(pts) > 300
    assert rgba.shape == (len(pts), 4)
    # every confirmed point lies on one of the three planes
    d = np.stack([np.abs(pts[:, 2] - 2.0),
                  np.abs(pts[:, 0] + 0.5),
                  np.abs(pts[:, 1] - 0.8)]).min(axis=0)
    assert np.percentile(d, 95) < 0.02


def test_mapper_tracks_known_trajectory():
    model = tof3d.get_model(54, 42, binning=2)
    m = DepthMapper(parallax=False)
    n = 25
    for k in range(n):
        ok = m.add_frame(render_corner(model, pose_at(k)), None, binning=2)
        assert ok, f"lost tracking at frame {k} (fit {m.fitness:.2f})"
    gt = pose_at(n - 1)
    t_err = np.linalg.norm(m.pose[:3, 3] - gt[:3, 3])
    R_err = m.pose[:3, :3] @ gt[:3, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
    assert t_err < 0.03, f"translation error {t_err * 1000:.1f} mm"
    assert ang < 2.0, f"rotation error {ang:.2f} deg"


def test_mapper_rejects_garbage_frame():
    model = tof3d.get_model(54, 42, binning=2)
    m = DepthMapper(parallax=False)
    img = render_corner(model, np.eye(4))
    for _ in range(3):
        m.add_frame(img, None, binning=2)
    pose_before = m.pose.copy()
    # an unrelated wall much closer than anything in the map: ICP must reject
    bad = np.full(img.shape, 300.0, np.float32)
    assert not m.add_frame(bad, None, binning=2)
    assert m.lost
    assert np.allclose(m.pose, pose_before)
    # recovery on the next good frame
    assert m.add_frame(img, None, binning=2)
    assert not m.lost


def test_mapper_finish_writes_ply(tmp_path):
    model = tof3d.get_model(54, 42, binning=2)
    m = DepthMapper(parallax=False)
    for k in range(8):
        m.add_frame(render_corner(model, pose_at(k)), None, binning=2)
    out = tmp_path / "map.ply"
    msg = m.finish(str(out))
    written = list(tmp_path.glob("*.ply"))
    assert written, msg
    head = written[0].read_bytes()[:200]
    assert head.startswith(b"ply")
