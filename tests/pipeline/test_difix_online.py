"""Reproducible online Difix: per-view seeds and pooled round control (CPU only)."""

import pytest

from hypersplat.pipeline.params import SceneProfile
from hypersplat.pipeline.params.strategies.difix import (derive_max_angle, derive_pseudo_weight, judged,
                                                         round_hallucinating, round_stats)
from hypersplat.stream.difix_online import OnlineDifix


def _difix(**kw):
    return OnlineDifix(profile=SceneProfile(), **kw)


def _round(d, changes, kept, angle=6.0, loop=0):
    stats = round_stats(loop, angle, [angle] * len(changes), changes, 15.0, kept)
    return d._pooled(stats, changes, kept, len(changes), angle)


def test_view_seeds_are_fixed_and_distinct():
    d = _difix(seed=0)
    seeds = {d.view_seed(r, k) for r in range(20) for k in range(16)}
    assert len(seeds) == 320
    assert _difix(seed=0).view_seed(3, 1) == d.view_seed(3, 1)
    assert _difix(seed=1).view_seed(3, 1) != d.view_seed(3, 1)
    assert _difix(seed=None).view_seed(3, 1) is None


def test_one_bad_two_view_round_no_longer_stops_difix():
    # Per round (legacy): 0 of 2 views kept -> "hallucinating", Difix off for the rest of the run
    legacy = round_stats(0, 6.0, [6.0, 6.0], [14.0, 14.5], 15.0, 0)
    assert round_hallucinating(legacy)
    d = _difix(pool_views=8)
    s = _round(d, [14.0, 14.5], 0)
    assert s["judge"] == "pooling" and judged(s) == (None, 0, None)
    assert s["n_views"] == 2 and s["n_kept"] == 0  # the round's own stats are unchanged
    assert round_hallucinating(s) is None
    for _ in range(3):  # three good rounds fill the pool: the bad pair is outvoted
        s = _round(d, [22.0, 23.0], 2)
    assert judged(s)[1:] == (8, 6) and round_hallucinating(s) is None
    assert s["mean_change_psnr"] == pytest.approx(22.5)


def test_pool_slides_and_restarts_on_angle_change():
    d = _difix(pool_views=4)
    for _ in range(5):
        s = _round(d, [20.0, 20.0], 2)
    assert s["pooled_rounds"] == 2 and judged(s)[1] == 4
    s = _round(d, [20.0, 20.0], 2, angle=9.0)
    assert s["judge"] == "pooling"  # change PSNR depends on the angle: start over


def test_weight_and_angle_hold_while_pooling():
    prof = SceneProfile()
    prof.append("difix.rounds", {"max_angle": 6.0, "mean_change_psnr": 22.0, "n_views": 8, "n_kept": 8}, "t")
    w = derive_pseudo_weight(prof, None).value
    prof.append("difix.rounds", {"max_angle": 9.0, "mean_change_psnr": 12.0, "n_views": 2, "n_kept": 0,
                                 "judge": "pooling"}, "t")
    assert derive_pseudo_weight(prof, None).value == w
    assert derive_max_angle(prof, None).value == 9.0


def test_pseudo_share_cap_keeps_real_views_dominant():
    import numpy as np

    from hypersplat.stream.training import StreamTrainSet

    ts = StreamTrainSet(session=None, pseudo_share=0.15)
    ts._pseudo = [False] * 4 + [True] * 4
    w = np.array([0.2] * 4 + [1.0] * 4)  # fresh pseudo views: novelty favours them (83%)
    c = ts._cap_pseudo(w)
    assert c[4:].sum() / c.sum() == pytest.approx(0.15)
    assert np.allclose(c[:4], w[:4])  # real weights untouched
    ts._pseudo = [False] * 8  # no pseudo views: nothing to cap
    assert np.allclose(ts._cap_pseudo(w), w)
    ts.pseudo_share, ts._pseudo = None, [False] * 4 + [True] * 4  # no cap configured
    assert np.allclose(ts._cap_pseudo(w), w)


def test_preload_must_finish_before_modules_are_created(monkeypatch):
    # The Hugging Face load swaps nn.Module.register_parameter process-wide
    # (accelerate.init_empty_weights): a parameter registered on another thread meanwhile
    # lands on the meta device. wait_preload() is the barrier the trainer relies on.
    import threading
    import time

    import torch
    accelerate = pytest.importorskip("accelerate")

    inside = threading.Event()

    def slow_build(device):
        with accelerate.init_empty_weights():
            inside.set()
            time.sleep(0.5)
        return object()

    d = _difix()
    monkeypatch.setattr(d, "_build_fixer", slow_build)
    d.preload()
    inside.wait(5)
    raced = torch.nn.ParameterDict({"means": torch.nn.Parameter(torch.zeros(4, 3))})
    assert raced["means"].device.type == "meta"   # the race the trainer used to hit
    d.wait_preload()
    assert d.fixer is not None and d._preload is None
    safe = torch.nn.ParameterDict({"means": torch.nn.Parameter(torch.zeros(4, 3))})
    assert safe["means"].device.type == "cpu"


def test_failed_preload_is_cleared_by_wait(monkeypatch):
    d = _difix()

    def boom(device):
        raise RuntimeError("no network")

    monkeypatch.setattr(d, "_build_fixer", boom)
    d.preload()
    d.wait_preload()
    assert d.fixer is None and d._preload is None and d._preload_error is None
    d.wait_preload()   # idempotent



def test_round_starts_pseudo_views_from_depth_warped_photos(monkeypatch):
    import time

    import numpy as np

    from hypersplat.stream.difix_online import DifixSources

    H, W = 40, 60
    K = np.array([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]])
    photo = (np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)
    moved = np.eye(4)
    moved[0, 3] = 0.08                                   # 2 px of disparity at z = 2
    src = DifixSources(c2ws=np.stack([np.eye(4), np.eye(4)]), K=K, width=W, height=H,
                       weights=np.ones(2), points=np.zeros((1, 3)), get_image=lambda i: photo,
                       ids=[10, 11], get_depth=lambda i: np.full((H // 2, W // 2), 2.0) if i == 0 else None)
    d = OnlineDifix(profile=SceneProfile(), device="cpu", model_id="nvidia/difix_ref")
    monkeypatch.setattr(d, "_fixer_on_gpu", lambda: (lambda img, ref, scale, seed: img))
    monkeypatch.setattr(d, "_park", lambda: None)
    d._round = {"src": src, "poses": [(0, moved, 1.0), (1, moved, 1.0)], "angle": 2.0, "base_weight": 1.0,
                "renders": [], "fixed": [], "t0": time.time(), "step0": 0, "chunks": 0, "max_chunk_s": 0.0,
                "t": {"render_ms": 0.0, "difix_ms": 0.0, "load_ms": 0.0, "park_ms": 0.0, "warp_ms": 0.0},
                "warp_cover": []}
    gray = np.full((H, W, 3), 77, np.uint8)
    assert d.step_round(lambda c2w, K, w, h: gray, budget_s=None)
    warped, plain = d._round["renders"]
    np.testing.assert_array_equal(warped[:, 3:-3], photo[:, 5:-1])   # the real photo, shifted
    assert (warped[:, -1] == 77).all()                               # the hole: the splat render
    assert (plain == 77).all()                                       # no depth: render only
    assert d._round["warp_cover"] == [pytest.approx(1 - 1 / W)]
    round_state = dict(d._round)
    _, _, stats = d.finish_round(step=100)
    assert stats["warped_views"] == 1 and stats["warp_cover"] == pytest.approx(1 - 1 / W, abs=1e-3)
    d._round = round_state
    d.depth_warp = False
    d._round.update(renders=[], fixed=[])
    d.step_round(lambda c2w, K, w, h: gray, budget_s=None)
    assert (d._round["renders"][0] == 77).all()
