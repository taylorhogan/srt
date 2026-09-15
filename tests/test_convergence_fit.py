"""convergence.decay_fit — the curve's curvature as one number.

The convergence curve is the residual of a k-frame stack against the
all-frames golden, which for independent noise is A * sqrt(1/k - 1/n).
decay_fit fits residual = A * (1/k - 1/n)^(q/2) through every point but the
last (k = n is zero by construction): q = 1 is that ideal, q < 1 is a curve
flattening harder than statistics allow (a correlated term), q > 1 is later
frames worth more than early ones. Pure Python; nothing here touches disk,
numpy, astropy or hardware.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# convergence imports configs.config at load, and that imports the gitignored
# config_private, which CI does not have. Stand in a stub whose data() is
# empty: every threshold the summary reads falls back to its default, and
# the tests below pin the ones they depend on with monkeypatch anyway.
try:
    import configs.config  # noqa: F401
except Exception:  # noqa: BLE001
    import types
    import configs
    _stub = types.ModuleType("configs.config")
    _stub.data = lambda: {}
    sys.modules["configs.config"] = _stub
    configs.config = _stub

from fits_processing import convergence as c  # noqa: E402

COUNTS = [1, 2, 3, 5, 8, 13, 21, 34, 55]
N = COUNTS[-1]


def _curve(q, amp=0.8):
    return [amp * (1 / k - 1 / N) ** (q / 2) for k in COUNTS]


def test_textbook_curve_has_exponent_one_and_every_frame_counts():
    fit = c.decay_fit(COUNTS, _curve(1.0))
    assert fit is not None
    assert abs(fit["exponent"] - 1.0) < 1e-6
    assert abs(fit["amplitude"] - 0.8) < 1e-6
    assert abs(fit["tail_ratio"] - 1.0) < 1e-6
    assert fit["effective_frames"] == N
    assert fit["points"] == len(COUNTS) - 1        # k = n excluded


def test_correlated_curve_is_recovered_and_frames_lose_weight():
    fit = c.decay_fit(COUNTS, _curve(0.7))
    assert abs(fit["exponent"] - 0.7) < 1e-6
    assert fit["tail_ratio"] > 1.5                 # tail sits above the ideal
    assert fit["effective_frames"] < N / 2         # 55 frames average like far fewer
    # tail_ratio is the whole-curve version of decay_ratio's two-point number
    assert abs(fit["tail_ratio"] - c.decay_ratio(COUNTS, _curve(0.7))) < 1e-3   # stored to 3 dp


def test_noise_cannot_make_more_independent_frames_than_frames():
    rng = random.Random(1)
    noisy = [v * (1 + rng.gauss(0, 0.05)) for v in _curve(1.0)]
    fit = c.decay_fit(COUNTS, noisy)
    assert 0.9 < fit["exponent"] < 1.1
    assert fit["effective_frames"] <= N


def test_too_little_to_fit_is_none():
    assert c.decay_fit([1, 2, 3], [1.0, 0.5, 0.0]) is None
    assert c.decay_fit(COUNTS, [0.0] * len(COUNTS)) is None      # no positive residuals
    assert c.decay_fit(COUNTS, _curve(1.0)[:-1]) is None          # length mismatch
    assert c.decay_fit([1], [0.0]) is None


def test_curvature_sentence_names_the_three_regimes():
    assert c.curvature_sentence(None, 55) == ""
    slow = c.curvature_sentence(c.decay_fit(COUNTS, _curve(0.7)), N)
    assert "harder than photon statistics" in slow and "average like" in slow
    fair = c.curvature_sentence(c.decay_fit(COUNTS, _curve(1.0)), N)
    assert "as independent frames should" in fair
    fast = c.curvature_sentence(c.decay_fit(COUNTS, _curve(1.3)), N)
    assert "later frames are pulling more weight" in fast


def test_summary_carries_the_curvature_and_tempers_the_recommendation(monkeypatch):
    monkeypatch.setattr(c, "_threshold", lambda: 0.4)
    monkeypatch.setattr(c, "_min_frames", lambda: 16)
    text = c.progress_summary("O-III", COUNTS, _curve(0.7), -0.9, 3.0, 60)
    assert "Curvature:" in text
    assert "find the correlated term" in text
    assert "Caveat:" not in text                    # the fit replaced the two-point caveat
    text = c.progress_summary("Ha", COUNTS, _curve(1.0), -0.9, 3.0, 60)
    assert "as independent frames should" in text
    assert "keep shooting this filter" in text


def test_summary_below_min_frames_stays_silent_on_curvature(monkeypatch):
    monkeypatch.setattr(c, "_threshold", lambda: 0.4)
    monkeypatch.setattr(c, "_min_frames", lambda: 100)
    text = c.progress_summary("L", COUNTS, _curve(0.7), -0.9, 3.0, 60)
    assert "Too early to judge" in text and "Curvature:" not in text
