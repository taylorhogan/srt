"""C3 of docs/AUTOMATION_PLAN.md (2026-10-30): the emission level in the convergence record.

The record builders and the ratio reader are pure (fits_processing.convergence
imports configs.config, which CI stubs); the golden measurement needs numpy.
"""
import pytest

from fits_processing import convergence as C

MEASURE = {"sky_adu": 1.5, "noise_adu": 0.4, "level_adu": 6.0, "block_px": 3, "scale": 11.7, "frames": 40}


def test_emission_record_is_per_second_with_the_frames_facts():
    em = C.emission_record(MEASURE, {"exposure_s": 300.0, "gain": 26})
    assert em["level_adu_s"] == pytest.approx(0.02) and em["sky_adu_s"] == pytest.approx(0.005)
    assert em["exposure_s"] == 300.0 and em["gain"] == 26 and em["block_px"] == 3 and em["frames"] == 40
    assert C.emission_record(None, {"exposure_s": 300.0}) is None
    assert C.emission_record(MEASURE, {"exposure_s": None}) is None          # nothing to divide by
    assert C.emission_record({}, {"exposure_s": 300.0}) is None


def test_filter_entry_is_the_one_shape_for_both_writers():
    counts, resid = [1, 2, 3, 5, 8], [0.9, 0.6, 0.4, 0.2, 0.0]
    e = C.filter_entry(counts, resid, -0.1234567, 3.21, 10, True, MEASURE, {"exposure_s": 300.0, "gain": 26})
    assert e["frame_count"] == 8 and e["total_frames"] == 10 and e["calibrated"] is True
    assert e["tail_slope_pct"] == -0.123457 and e["final_rmse_pct"] == 3.21
    assert "decay_exponent" in e and "effective_frames" in e and "updated" in e
    assert e["emission"]["level_adu_s"] == pytest.approx(0.02)
    assert "emission" not in C.filter_entry(counts, resid, -0.1, 3.0, 10, False)   # no measurement: as before


def _entry(level, gain=26, exp=300.0):
    return {"emission": {"level_adu_s": level, "exposure_s": exp, "gain": gain}}


def test_emission_ratios_against_ha_at_one_gain():
    filters = {"Ha": _entry(0.02), "S-II": _entry(0.0012), "O-III": _entry(0.009),
               "L": _entry(0.05, gain=0), "R": {"tail_slope_pct": -0.1}}       # L at another gain, R unmeasured
    r = C.emission_ratios(filters)
    assert r == {"S-II": 0.06, "O-III": 0.45}
    assert C.emission_ratios({"S-II": _entry(0.001)}) == {}                   # no reference
    assert C.emission_ratios({"Ha": _entry(0.0), "S-II": _entry(0.001)}) == {}
    assert C.emission_ratios({"Ha": _entry(0.02), "S-II": _entry(0.001, gain=0)}) == {}   # gains differ
    # the 2024 archive's spelling reads as the canonical name
    assert C.emission_ratios({"Ha": _entry(0.02), "O2": _entry(0.01)}) == {"O-III": 0.5}
    assert C.emission_ratios({}) == {} and C.emission_ratios(None) == {}


def test_golden_measure_is_the_nebula_not_the_stars():
    np = pytest.importorskip("numpy")
    from stacking import stacker as S
    rng = np.random.default_rng(0)
    g = 1.5 + rng.normal(0, 0.05, (512, 512)).astype("float32")             # sky 1.5 ADU
    g[100:300, 100:300] += 4.0                                                 # a nebula, 15 % of the field
    for _ in range(300):                                                       # stars: bright, tiny
        y, x = rng.integers(2, 510, 2)
        g[y - 1:y + 2, x - 1:x + 2] += 500.0
    m = S.golden_measure(g, 1.5, 0.05, native_width=6000, n_frames=40)
    assert m["block_px"] == 3 and m["scale"] == pytest.approx(11.72, abs=0.01) and m["frames"] == 40
    # the level is the nebula's ~4 ADU above sky, not the stars' hundreds
    assert 3.5 < m["level_adu"] < 4.5
    assert m["sky_adu"] == 1.5
