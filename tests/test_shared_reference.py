"""The shared-reference picker survives a clouded middle frame.

2026-10-18: ngc6946's morning render was refused because the middle of its 41 L
frames fell in a clouded hour (0 stars, sky 2855 ADU against 647) and the
picker tried that one frame only. The quality gate would have dropped it at
stack time; the reference pick was the one place a single bad frame could
stop a whole target.
"""
from pathlib import Path

import pytest

# CI runs on bare Python + pytest (see .github/workflows): guard the science
# stack the way every other test here does, or collection fails and the
# whole gate goes red.
np = pytest.importorskip("numpy")
pytest.importorskip("astropy")

from nn import stacks  # noqa: E402
from stacking import stacker  # noqa: E402


def _paths(n):
    return [Path(f"/x/dso/cdk17/2026-10-15/LIGHT/f{i:03d}.fits") for i in range(n)]


@pytest.fixture
def fake_stacker(monkeypatch):
    """Control points per frame index come from a dict; everything else is stubbed."""
    state = {"points": {}, "loaded": []}
    monkeypatch.setattr(stacker, "calibration_from_config", lambda f: None)
    monkeypatch.setattr(stacker, "_despike", lambda a: a)

    def load(path, cal):
        i = int(path.stem[1:])
        state["loaded"].append(i)
        return np.full((4, 4), i, dtype=np.float32)

    def points(ref):
        n = state["points"].get(int(ref[0, 0]), 0)
        return None if n < 3 else np.zeros((n, 2))

    monkeypatch.setattr(stacker, "_load_calibrated", load)
    monkeypatch.setattr(stacker, "_reference_control_points", points)
    monkeypatch.setattr(stacks, "load_precomputed_fwhm_stars", lambda *a: {}, raising=False)
    return state


def test_middle_frame_is_first_choice(fake_stacker):
    fake_stacker["points"] = {i: 200 for i in range(41)}
    ref = stacks.shared_reference_for(_paths(41), "L", progress_cb=lambda m: None)
    assert ref is not None and len(ref[0]) == 200
    assert fake_stacker["loaded"] == [20]


def test_clouded_middle_moves_on(fake_stacker):
    # frames 17-28 clouded (as on ngc6946), the rest clear
    fake_stacker["points"] = {i: (0 if 17 <= i <= 28 else 200) for i in range(41)}
    msgs = []
    ref = stacks.shared_reference_for(_paths(41), "L", progress_cb=msgs.append)
    assert ref is not None and len(ref[0]) == 200
    first = fake_stacker["loaded"][0]
    assert first == 20 and not (17 <= fake_stacker["loaded"][-1] <= 28)
    assert any("no control points — trying another" in m for m in msgs)
    assert any(m.startswith("    shared reference:") for m in msgs)


def test_thin_cloud_everywhere_takes_the_best(fake_stacker):
    fake_stacker["points"] = {i: 5 + i % 7 for i in range(41)}       # all below REF_MIN_POINTS
    ref = stacks.shared_reference_for(_paths(41), "L", progress_cb=lambda m: None)
    assert ref is not None
    tried = fake_stacker["loaded"]
    assert len(tried) == stacks.REF_CANDIDATES
    assert len(ref[0]) == max(5 + i % 7 for i in tried)


def test_all_clouded_gives_up(fake_stacker):
    fake_stacker["points"] = {}
    msgs = []
    assert stacks.shared_reference_for(_paths(41), "L", progress_cb=msgs.append) is None
    assert len(fake_stacker["loaded"]) == stacks.REF_CANDIDATES
    assert any("per-split refs" in m for m in msgs)


def test_short_run_tries_every_frame_once(fake_stacker):
    fake_stacker["points"] = {}
    stacks.shared_reference_for(_paths(4), "L", progress_cb=lambda m: None)
    assert sorted(fake_stacker["loaded"]) == [0, 1, 2, 3]
