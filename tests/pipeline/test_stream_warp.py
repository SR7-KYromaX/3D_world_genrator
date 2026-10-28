"""Depth-image-based novel views for online Difix (hypersplat/stream/warp.py), on the CPU."""

import numpy as np

from hypersplat.stream.warp import composite, warp_to_view

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((H, W, 3)) * 255).astype(np.uint8)


def _c2w(tx=0.0, ty=0.0, tz=0.0):
    m = np.eye(4)
    m[:3, 3] = (tx, ty, tz)
    return m


def test_identity_reproduces_the_photo():
    rgb = _texture()
    out, valid = warp_to_view(rgb, np.full((H, W), 2.0, np.float32), K, _c2w(), _c2w())
    assert valid.all()
    np.testing.assert_array_equal(out, rgb)


def test_sideways_move_shifts_a_plane_by_its_disparity():
    # Plane at z = 2, camera 2/70 * 2 = 0.057 to the right: the image shifts 2 px left
    rgb = _texture(1)
    out, valid = warp_to_view(rgb, np.full((H // 2, W // 2), 2.0, np.float32), K, _c2w(),
                              _c2w(tx=2.0 * 2.0 / 70.0))
    np.testing.assert_array_equal(out[:, 5:-5], rgb[:, 7:-3])
    # The 2 px strip the camera moved into: the 2x2 dots spill one column in, the last stays a hole
    assert not valid[:, -1].any()
    assert valid[:, :-1].all()


def test_nearer_surface_wins_and_disocclusion_is_a_hole():
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, :] = (0, 0, 255)                      # far wall
    rgb[20:40, 30:50] = (255, 0, 0)              # near box
    depth = np.full((H, W), 4.0, np.float32)
    depth[20:40, 30:50] = 1.0
    out, valid = warp_to_view(rgb, depth, K, _c2w(), _c2w(tx=0.1))   # box moves 7 px, wall 1.75
    assert (out[25:35, 27:38] == (255, 0, 0)).all()                   # box in front of the wall
    assert not valid[25:35, 43:48].all()                              # wall behind it never seen
    filled = composite(out, valid, np.full_like(out, 77))
    assert (filled[~valid] == 77).all() and (filled[valid] == out[valid]).all()


def test_invalid_depth_and_points_behind_the_camera_are_dropped():
    rgb = _texture(2)
    depth = np.full((H, W), 2.0, np.float32)
    depth[:, : W // 2] = 0.0
    out, valid = warp_to_view(rgb, depth, K, _c2w(), _c2w())
    assert valid[:, W // 2:].all() and not valid[:, : W // 2 - 1].any()
    _, valid = warp_to_view(rgb, np.full((H, W), 2.0, np.float32), K, _c2w(), _c2w(tz=3.0))
    assert not valid.any()
