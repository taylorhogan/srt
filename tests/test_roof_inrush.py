"""The roof plug's first reading of a move can carry the motor's start-up inrush
(2026-11-21): the plug computes a reading about once a second, so when its
window opens on the breakaway the first number is 400-760 W against 310-350 W
running, held for one second, gone by the next. 53 of 187 moves June-November
2026 showed it, always the first reading, never mid-travel. It is sampling, not
the roof: the peak is scored after that window and the reading is kept as start_w."""
import pytest

np = pytest.importorskip("numpy")                               # the bare CI env has none
rcs = pytest.importorskip("sentry.roof_current_signature")
from sentry.roof_move_report import line  # noqa: E402


def _trace(first_w, run_w=340.0, spike_at=None, hz=20, idle=5.0, travel=11.0, tail=5.0):
    """A 20 Hz capture: idle, one second at first_w, the travel at run_w, idle.
    The plug holds each reading for a second, so values repeat 20 times."""
    t, p = [], []
    k = 0
    for sec in np.arange(0, idle + travel + tail, 1.0):
        if sec < idle or sec >= idle + travel:
            w = 5.0
        elif sec < idle + 1.0:
            w = first_w
        elif spike_at is not None and sec == idle + spike_at:
            w = first_w                      # a second reading at the breakaway level, mid-travel
        else:
            w = run_w
        for _ in range(hz):
            t.append(k / hz); p.append(w); k += 1
    c = [w / 120.0 for w in p]
    return {"direction": "open", "timestamp": "2026-11-21T16:00:00",
            "samples": {"t": t, "power": p, "current": c, "voltage": [120.0] * len(p)}}


def test_the_first_reading_is_start_w_and_the_peak_is_judged_after_it():
    f = rcs.extract_features(_trace(578.0))
    assert f["valid"] and f["features_version"] == rcs.FEATURES_VERSION
    assert f["start_w"] == 578.0 and f["peak_w"] == 340.0 and f["running_w"] == 340.0
    assert 10.5 <= f["move_duration_s"] <= 11.5
    plain = rcs.extract_features(_trace(340.0))
    assert plain["start_w"] == 340.0 and plain["peak_w"] == 340.0


def test_a_spike_mid_travel_still_counts_as_the_peak():
    f = rcs.extract_features(_trace(600.0, spike_at=4.0))
    assert f["start_w"] == 600.0 and f["peak_w"] == 600.0      # the 4 s reading is real load


def test_compare_rescores_a_library_saved_before_the_rule():
    """Stored features from the old extractor carry first-reading peaks; compare
    judges every signature at the current version, so a plain move is not 'low'
    against them and a breakaway reading is not 'high'."""
    lib = []
    for w in (578.0, 340.0, 612.0, 345.0, 338.0):
        s = _trace(w); f = rcs.extract_features(s)
        f = dict(f, peak_w=w); f.pop("features_version")        # as the old code stored it
        s["features"] = f; lib.append(s)
    assert all(rcs.features_of(s)["peak_w"] == 340.0 for s in lib)
    res = rcs.compare(_trace(700.0), library=lib)
    assert not res["is_anomaly"], res["reasons"]
    assert res["detail"]["peak_w"]["mean"] == 340.0
    bad = rcs.compare(_trace(340.0, run_w=520.0), library=lib)
    assert bad["is_anomaly"] and any(k.startswith("peak_w") or k.startswith("running_w") for k in bad["reasons"])


def test_the_report_line_shows_the_start_reading_without_judging_it():
    feats = {"valid": True, "move_duration_s": 11.0, "peak_w": 349.0, "running_w": 346.0,
             "start_w": 578.0, "returned_to_baseline": True}
    ok = {"is_anomaly": False, "reasons": [], "golden_ok": True, "golden_reasons": []}
    assert line("open", feats, ok) == "Roof open: 11.0 s, peak 349 W / running 346 W, start 578 W (typical), audio not captured, north tag not read yet."
    feats["start_w"] = 352.0
    assert "start" not in line("open", feats, ok)
