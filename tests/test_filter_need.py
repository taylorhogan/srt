"""filter_need — the §2c need-weighted split, tested from dict fixtures.

Every test drives the pure core with an injected convergence slice and
explicit thresholds; nothing here touches config, disk, astropy, or hardware,
so CI's bare pytest runner covers the whole policy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fits_processing.filter_need import (  # noqa: E402
    counts_from_weights, filter_need)

# Thresholds pinned explicitly so config changes cannot silently move tests.
THR = dict(slope_threshold=0.40, rmse_threshold=5.0, min_frames=30)


def _entry(slope, frames, total=None, rmse=1.0, calibrated=True):
    e = {"tail_slope_pct": slope, "frame_count": frames,
         "final_rmse_pct": rmse, "calibrated": calibrated}
    if total is not None:
        e["total_frames"] = total
    return e


CONVERGED = _entry(-0.10, 60)
IMPROVING = _entry(-1.20, 50, total=60)      # needs ~ 50*(3-1)*(60/50) = 120


def test_no_history_returns_none():
    assert filter_need(None, "nebula", **THR) is None
    assert filter_need({}, "nebula", **THR) is None


def test_converged_filter_gets_zero_share():
    w = filter_need({"Ha": CONVERGED, "O-III": IMPROVING, "S-II": IMPROVING},
                    "nebula", **THR)
    assert w["Ha"] == 0.0
    assert w["O-III"] > 0 and w["S-II"] > 0


def test_missing_filter_gets_the_largest_share():
    """The zero-S-II rule: a never-imaged filter of the set dominates."""
    w = filter_need({"Ha": CONVERGED, "O-III": IMPROVING}, "nebula", **THR)
    assert w["S-II"] == max(w.values())
    assert w["S-II"] >= w["O-III"]


def test_missing_filter_share_floored_at_min_frames():
    """With every measured filter converged, the missing one still gets a
    full min_frames worth of need — not a share of nothing."""
    w = filter_need({"Ha": CONVERGED, "O-III": CONVERGED}, "nebula", **THR)
    assert w == {"Ha": 0.0, "O-III": 0.0, "S-II": 30.0}


def test_uncalibrated_entry_treated_as_missing():
    """Old pedestal-scale slopes are unknowable, not converged."""
    old = _entry(-0.01, 200, calibrated=False)
    w = filter_need({"Ha": CONVERGED, "O-III": old, "S-II": IMPROVING},
                    "nebula", **THR)
    assert w["O-III"] == max(w.values())


def test_all_converged_returns_empty_dict():
    w = filter_need({"Ha": CONVERGED, "O-III": CONVERGED, "S-II": CONVERGED},
                    "nebula", **THR)
    assert w == {}


def test_need_proportional_to_slope_and_keep_rate():
    barely = _entry(-0.80, 40, total=40)      # 40*(2-1)        = 40
    badly = _entry(-1.60, 40, total=80)       # 40*(4-1)*(80/40) = 240
    w = filter_need({"Ha": barely, "O-III": badly, "S-II": CONVERGED},
                    "nebula", **THR)
    assert w["S-II"] == 0.0
    assert abs(w["Ha"] - 40.0) < 1e-9
    assert abs(w["O-III"] - 240.0) < 1e-9


def test_below_min_frames_need_floored_at_reaching_min():
    """A 5-frame filter's flat slope is noise; it needs at least 25 more."""
    early = _entry(-0.05, 5, rmse=1.0)        # slope formula would say 0
    w = filter_need({"Ha": early, "O-III": CONVERGED, "S-II": CONVERGED},
                    "nebula", **THR)
    assert w["Ha"] >= 25.0


def test_high_rmse_is_not_converged():
    noisy = _entry(-0.10, 60, rmse=40.0)
    w = filter_need({"Ha": noisy, "O-III": CONVERGED, "S-II": CONVERGED},
                    "nebula", **THR)
    assert w["Ha"] > 0


def test_filter_names_matched_case_insensitively():
    w = filter_need({"ha": CONVERGED, "o-iii": IMPROVING, "s-ii": IMPROVING},
                    "nebula", **THR)
    assert w["Ha"] == 0.0


def test_galaxy_and_unknown_use_lrgb_set():
    entries = {"L": IMPROVING}
    for t in ("galaxy", "unknown", "somethingelse"):
        w = filter_need(entries, t, **THR)
        assert set(w) == {"L", "R", "G", "B"}


def test_counts_split_seconds_proportionally():
    counts = counts_from_weights({"O-III": 60.0, "Ha": 20.0, "S-II": 0.0},
                                 8 * 3600, 300)
    assert counts["S-II"] == 0                 # explicit zero, zeroes the block
    assert counts["O-III"] == 3 * counts["Ha"]
    total_s = sum(c * 300 for c in counts.values())
    assert total_s <= 8 * 3600


def test_counts_degenerate_inputs():
    assert counts_from_weights({}, 3600, 300) == {}
    assert counts_from_weights({"Ha": 0.0}, 3600, 300) == {}
    assert counts_from_weights({"Ha": 1.0}, 0, 300) == {}
    assert counts_from_weights({"Ha": 1.0}, 3600, 0) == {}
