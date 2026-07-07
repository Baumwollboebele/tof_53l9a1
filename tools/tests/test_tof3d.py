"""Unit tests for the shared 3-D geometry (tof3d.py): the numpy port of ST's
vl53l9-transform-c optical model and the depth filters."""

import numpy as np
import pytest

import tof3d


def wall_radial(model, z0_m):
    """Radial-distance image (mm) of a fronto-parallel wall at z0 metres."""
    tx, ty = model.tangents()
    return (z0_m * np.sqrt(tx ** 2 + ty ** 2 + 1.0) * 1000.0).astype(np.float32)


def test_model_constants():
    m = tof3d.get_model(54, 42, binning=2)
    assert not m.pinhole
    # focal = EFL / (spad_size * spads_per_pixel); 216 SPADs over 54 px -> 4
    assert m.focal == pytest.approx(2428.16 / (10.17 * 4.0), rel=1e-6)
    assert m.cx == pytest.approx(26.5)
    assert m.cy == pytest.approx(20.5)
    # real horizontal FoV is ~49 deg, not the old 60 deg guess
    hfov = 2 * np.degrees(np.arctan(27.0 / m.focal))
    assert 47.0 < hfov < 51.0


def test_model_cache():
    assert tof3d.get_model(54, 42, 2) is tof3d.get_model(54, 42, 2)
    assert tof3d.get_model(54, 42, 2) is not tof3d.get_model(24, 20, 4)


def test_wall_unprojects_flat():
    m = tof3d.get_model(54, 42, binning=2)
    z0 = 1.5
    x, y, z = m.unproject(wall_radial(m, z0), parallax=False)
    assert np.allclose(z, z0, atol=1e-4)
    tx, ty = m.tangents()
    assert np.allclose(x, tx * z0, atol=1e-4)
    assert np.allclose(y, ty * z0, atol=1e-4)


def test_pinhole_override():
    m = tof3d.get_model(54, 42, binning=2, hfov_deg=60.0)
    assert m.pinhole
    z0 = 2.0
    x, y, z = m.unproject(wall_radial(m, z0))
    assert np.allclose(z, z0, atol=1e-4)
    # edge ray at half the FoV: x/z = tan(30 deg) at the outermost column
    assert abs(x[21, 0] / z0) == pytest.approx(np.tan(np.radians(30)) * (26.5 / 27.0),
                                               rel=0.01)


def test_project_unproject_roundtrip():
    m = tof3d.get_model(54, 42, binning=2)
    cc, rr = np.meshgrid(np.arange(54, dtype=np.float32),
                         np.arange(42, dtype=np.float32))
    for z0 in (0.4, 1.0, 3.0):
        x, y, z = m.unproject(wall_radial(m, z0), parallax=True)
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        u, v = m.project(pts, parallax=True)
        assert np.abs(u - cc.ravel()).max() < 0.15
        assert np.abs(v - rr.ravel()).max() < 0.15


def test_invalid_pixels_stay_invalid():
    m = tof3d.get_model(54, 42, binning=2)
    img = wall_radial(m, 1.0)
    img[5:9, 7:11] = 0.0
    _, _, z = m.unproject(img)
    assert (z[5:9, 7:11] == 0).all()
    assert (z[20:25, 20:25] > 0).all()


def test_drop_flying_pixels():
    z = np.full((10, 10), 1.0, np.float32)
    z[:, 5:] = 2.0                       # depth step: the two edge columns fly
    out = tof3d.drop_flying_pixels(z, jump=0.5)
    assert (out[:, 4:6] == 0).all()
    assert (out[:, :4] == 1.0).all() and (out[:, 6:] == 2.0).all()
    # invalid neighbours never trigger the filter
    z2 = np.full((10, 10), 1.0, np.float32)
    z2[:, 5:] = 0.0
    assert (tof3d.drop_flying_pixels(z2, 0.5) == z2).all()


def test_amplitude_gate():
    z = np.full((8, 8), 1.0, np.float32)
    amp = np.full((8, 8), 1000.0, np.float32)
    amp[0, 0] = 5.0
    out, gray = tof3d.amplitude_gate(z, amp, frac=0.05)
    assert out[0, 0] == 0.0
    assert (out[1:] == 1.0).all()
    assert gray.dtype == np.uint8


def test_to_display_frame():
    # camera X right / Y down / Z forward -> GL X right / Y depth / Z up
    pts = np.array([[1.0, 2.0, 3.0]], np.float32)
    assert np.allclose(tof3d.to_display(pts), [[1.0, 3.0, -2.0]])
    assert len(tof3d.to_display(np.zeros((0, 3), np.float32))) == 0


def test_live_pointcloud():
    m = tof3d.get_model(54, 42, binning=2)
    img = wall_radial(m, 1.0)
    amp = np.full(img.shape, 500.0, np.float32)
    pts, rgba = tof3d.live_pointcloud(img, amp, binning=2, dmax_mm=2000,
                                      amp_frac=0.05, edge_jump=0.06)
    assert len(pts) == 54 * 42
    assert rgba.shape == (len(pts), 4)
    # display frame: depth along +Y, wall 1 m away
    assert np.allclose(pts[:, 1], 1.0, atol=1e-3)
    # beyond dmax everything is culled
    pts2, _ = tof3d.live_pointcloud(img, amp, binning=2, dmax_mm=500)
    assert len(pts2) == 0
