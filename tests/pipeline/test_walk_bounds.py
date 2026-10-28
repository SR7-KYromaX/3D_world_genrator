"""Walk-mode bounds for the live viewer (hypersplat/stream/walk_bounds.py)."""
import numpy as np
import pytest

from hypersplat.stream.walk_bounds import WalkBounds, WalkConfig, compute_walk_bounds


def _poses(centers, fwd=(1.0, 0.0, 0.0)):
    """Level OpenCV cameras (+Z forward) at the given centers, all looking along `fwd`."""
    f = np.asarray(fwd, float) / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(f, up)  # camera right
    y = np.cross(f, x)   # camera down
    out = np.tile(np.eye(4), (len(centers), 1, 1))
    out[:, :3, 0], out[:, :3, 1], out[:, :3, 2] = x, y, f
    out[:, :3, 3] = centers
    return out


def _line(n=20, step=0.1, z=1.5, slope=0.0):
    t = np.arange(n) * step
    return np.stack([t, np.zeros(n), z + slope * t], -1)


def test_straight_path_is_a_corridor():
    c = _line()
    b = compute_walk_bounds(_poses(c), scene_scale=1.0)
    r = WalkConfig().radius_mult * b.spacing
    assert b.spacing == pytest.approx(0.1)
    # Whole path is walkable, including midpoints between keyframes
    mids = (c[1:] + c[:-1]) / 2
    assert b.contains(c).all() and b.contains(mids).all()
    # Inside the radius beside the path yes, well outside no
    assert b.contains(np.array([[1.0, 0.8 * r]])).all()
    assert not b.contains(np.array([[1.0, 1.5 * r], [1.0, -1.5 * r], [-1.5 * r, 0.0]])).any()


def test_heights_follow_capture_height_on_a_slope():
    c = _line(slope=0.2)
    b = compute_walk_bounds(_poses(c), scene_scale=1.0)
    ij = np.round((c[:, :2] - np.asarray(b.origin)) / b.cell).astype(int)
    h = b.heights[ij[:, 1], ij[:, 0]]
    # Interior cameras: weighted mean of a linear ramp is ~the ramp itself
    np.testing.assert_allclose(h[3:-3], c[3:-3, 2], atol=0.02)


def test_relocalization_jump_is_not_bridged():
    a = _line(n=10)
    b_ = _line(n=10) + np.array([5.0, 0.0, 0.0])
    b = compute_walk_bounds(_poses(np.concatenate([a, b_])), scene_scale=5.0)
    assert not b.contains(np.array([[3.0, 0.0]])).any()
    assert b.contains(np.array([[0.5, 0.0], [5.5, 0.0]])).all()


def test_l_shaped_path_turns_the_corner():
    leg1 = _line(n=10)
    leg2 = np.stack([np.full(10, 0.9), 0.1 * np.arange(1, 11), np.full(10, 1.5)], -1)
    b = compute_walk_bounds(_poses(np.concatenate([leg1, leg2])), scene_scale=1.0)
    assert b.contains(np.array([[0.9, 0.0], [0.9, 0.5], [0.45, 0.0]])).all()
    assert not b.contains(np.array([[0.0, 1.0]])).any()  # far side of the L stays closed


@pytest.mark.parametrize("n", [1, 2])
def test_few_cameras(n):
    b = compute_walk_bounds(_poses(_line(n=n)), scene_scale=2.0)
    assert b.spacing == pytest.approx(0.1)  # 0.05 * scene_scale fallback
    assert b.contains(_line(n=n)).all()
    assert b.mask.sum() > 0


def test_empty_and_start_pose():
    assert compute_walk_bounds(np.zeros((0, 4, 4)), 1.0) is None
    c = _line()
    b = compute_walk_bounds(_poses(c, fwd=(0.0, 1.0, -0.3)), scene_scale=1.0)
    assert b.start_position == pytest.approx(tuple(c[0]))
    look = np.asarray(b.start_look_at) - c[0]
    assert look[2] == pytest.approx(0.0)  # levelled to the horizon
    assert look[1] > 0


def test_grid_is_capped():
    c = _line(n=400, step=0.05)
    c[:, 1] = np.sin(np.arange(400) * 0.05) * 5
    b = compute_walk_bounds(_poses(c), scene_scale=10.0)
    assert max(b.mask.shape) <= 256


def test_npz_round_trip(tmp_path):
    b = compute_walk_bounds(_poses(_line()), scene_scale=1.0)
    b.save_npz(tmp_path / "w.npz")
    b2 = WalkBounds.load_npz(tmp_path / "w.npz")
    assert b2.origin == pytest.approx(b.origin) and b2.num_views == b.num_views
    np.testing.assert_array_equal(b2.mask, b.mask)
    np.testing.assert_allclose(b2.heights, b.heights)
    assert b2.start_look_at == pytest.approx(b.start_look_at)


def test_clean_view_keeps_flat_surface_splats():
    # Splats on a surface are flat discs (two long axes, one tiny): the old max/min needle
    # test hid 100% of a phone model's visible splats; only max >> middle is a needle
    import sys
    import types
    from pathlib import Path

    import torch

    examples = str(Path(__file__).resolve().parents[2] / "examples")
    if examples not in sys.path:
        sys.path.insert(0, examples)
    st = pytest.importorskip("simple_trainer")
    scales = torch.tensor([[0.3, 0.3, 0.001],     # flat disc on a surface: keep
                           [0.6, 0.01, 0.01],     # needle: hide
                           [0.3, 0.3, 0.001],     # faint: hide
                           [2.0, 2.0, 2.0]])      # blob > 8 voxels: hide
    opa = torch.tensor([0.9, 0.9, 0.005, 0.9])
    cfg = types.SimpleNamespace(viewer_min_opacity=0.02, viewer_max_aniso=30.0, viewer_max_scale_voxels=8.0)
    me = types.SimpleNamespace(cfg=cfg, stream_voxel=0.1,
                               splats={"scales": scales.log(), "opacities": torch.logit(opa)})
    assert st.Runner._viewer_keep(me).tolist() == [1.0, 0.0, 0.0, 0.0]
    me.stream_voxel = None
    assert st.Runner._viewer_keep(me).tolist() == [1.0, 0.0, 0.0, 1.0]


def test_first_bounds_move_viewers_already_connected():
    # viser's initial_camera only reaches clients that connect later; the browser opened at
    # launch must be moved to the start view too
    import contextlib
    import threading
    import types

    from hypersplat.stream.walk_bounds import WalkBoundsPublisher

    class Client:
        def __init__(self):
            self.camera = types.SimpleNamespace(position=None, look_at=None, up_direction=None)

        @contextlib.contextmanager
        def atomic(self):
            yield

    clients = {1: Client(), 2: Client()}
    server = types.SimpleNamespace(initial_camera=types.SimpleNamespace(), get_clients=lambda: clients)
    pub = WalkBoundsPublisher.__new__(WalkBoundsPublisher)
    pub.server, pub.client_ok, pub._placed, pub.version, pub.bounds = server, False, False, 0, None
    pub._lock = threading.Lock()
    pub._info = types.SimpleNamespace(content="")
    b = compute_walk_bounds(_poses(_line()), scene_scale=1.0)
    pub.publish(b)
    assert server.initial_camera.position == b.start_position
    for c in clients.values():
        assert c.camera.position == b.start_position and c.camera.look_at == b.start_look_at
        assert c.camera.up_direction == (0.0, 0.0, 1.0)
    clients[1].camera.position = (9.0, 9.0, 9.0)
    pub.publish(b)                                   # later updates leave the user's view alone
    assert clients[1].camera.position == (9.0, 9.0, 9.0)
