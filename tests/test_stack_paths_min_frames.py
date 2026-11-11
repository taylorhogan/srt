"""nn.stacks.stack_paths: the two-frame floor is the training splits' rule,
and a render may lower it to one so a target with a single Ha frame still
gets its palette picture (IC 405, 2026-11-11)."""
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
stacks = pytest.importorskip("nn.stacks")


def _fake_stacker(monkeypatch, calls):
    import stacking.stacker as stacker

    def fake_stack(paths, **kw):
        calls.append(list(paths))
        return np.ones((4, 4), dtype=np.float32), {"n_frames": len(paths)}

    monkeypatch.setattr(stacker, "stack", fake_stack)
    monkeypatch.setattr(stacker, "calibration_paths_from_config",
                        lambda filt, like=None: ([], [], []))


def test_one_frame_is_refused_by_default(monkeypatch):
    calls = []
    _fake_stacker(monkeypatch, calls)
    assert stacks.stack_paths([Path("a.fits")], "Ha", progress_cb=lambda m: None) is None
    assert calls == []


def test_a_render_may_stack_one_frame(monkeypatch):
    calls = []
    _fake_stacker(monkeypatch, calls)
    meta = {}
    img = stacks.stack_paths([Path("a.fits")], "Ha", progress_cb=lambda m: None,
                             meta_out=meta, min_frames=1)
    assert img is not None and img.shape == (4, 4)
    assert meta["n_frames"] == 1
    assert calls == [[Path("a.fits")]]


def test_no_frames_is_never_stacked(monkeypatch):
    calls = []
    _fake_stacker(monkeypatch, calls)
    assert stacks.stack_paths([], "Ha", progress_cb=lambda m: None, min_frames=1) is None
    assert calls == []
