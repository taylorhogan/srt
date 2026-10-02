"""fits_processing/night_contribution: the summary, the verdict and the chat line."""
from fits_processing.night_contribution import describe, night_of, summarize, verdict


def _rows(night, n, a=1.0, sigma=1.0):
    return [{"night": night, "a": a, "sigma": sigma} for _ in range(n)]


def test_night_is_the_session_folder():
    p = r"C:\T\ngc7380\cdk17\2026-10-01\LIGHT\2026-10-02_01-54-41_S-II_2_300.00s_0021.fits"
    assert night_of(p) == "2026-10-01"


def test_equal_frames_add_as_root_n():
    rows = _rows("A", 16) + _rows("B", 9)
    r = summarize("x", "L", "B", rows, {"A": 16, "B": 9}, {})
    assert r["gain_optimal_pct"] == 25.0          # sqrt(25/16) - 1
    assert r["gain_equal_pct"] == 25.0
    assert r["worth_prior_frames"] == 9.0
    assert verdict(r) == "helped"


def test_weak_frames_hurt_a_plain_mean_but_not_a_weighted_stack():
    rows = _rows("A", 10) + _rows("B", 10, a=0.2, sigma=1.0)
    r = summarize("x", "Ha", "B", rows, {"A": 10, "B": 10}, {})
    assert r["gain_optimal_pct"] > 0
    assert r["gain_equal_pct"] < 0
    assert verdict(r) == "hurt"
    assert "HURT" in describe(r)


def test_nothing_through_the_gate_and_a_first_night():
    r = summarize("x", "S-II", "B", _rows("A", 5), {"A": 5, "B": 22}, {})
    assert verdict(r) == "nothing usable" and "0 of 22" in describe(r)
    r = summarize("x", "S-II", "B", _rows("B", 5), {"B": 5}, {})
    assert verdict(r) == "first night"


def test_fwhm_and_ratios_in_the_line():
    rows = _rows("A", 4) + _rows("B", 4, a=0.8, sigma=1.1)
    r = summarize("ngc7380", "Ha", "B", rows, {"A": 4, "B": 4},
                  {"A": [1.9, 1.9], "B": [2.1, 2.1]})
    line = describe(r)
    assert "signal 0.80x, noise 1.10x" in line and 'FWHM 2.10" vs 1.90"' in line
