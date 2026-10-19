"""The ISS recorder arms only for passes the sky camera can actually see.

2026-10-19: the ISS was in Earth's shadow from rise through its 78-deg peak and
lit only for its last 35 s, low in the north-east -- x = 76 -> -266 px on a
2560-wide frame. The flat 40-deg floor armed it anyway and the analyser, rightly,
found nothing. Now a pass needs MIN_VISIBLE_S of samples that are sunlit, over a
dark site, and inside the frame by the plate solution.
"""
import pytest

sw = pytest.importorskip("sentry.station_watch")


def test_visible_seconds_counts_only_lit_dark_in_frame_samples():
    s = [(True, True, True)] * 4 + [(True, True, False)] * 3 + [(False, True, True)] * 5 \
        + [(True, False, True)] * 2
    assert sw.visible_seconds(s, step_s=2.0) == 8.0


def test_the_10_19_pass_is_skipped_and_a_found_pass_still_arms():
    # 3 s of 10-19 in frame against 15 s for the tightest found pass (Tiangong 10-03).
    morning = [(True, True, True)] * 2 + [(False, True, False)] * 60
    tight_found = [(True, True, True)] * 8 + [(True, True, False)] * 30
    assert sw.visible_seconds(morning, 2.0) < sw.MIN_VISIBLE_S
    assert sw.visible_seconds(tight_found, 2.0) >= sw.MIN_VISIBLE_S


def test_frame_test_uses_the_plate_solution_and_its_margin(monkeypatch):
    np = pytest.importorskip("numpy")
    from sentry import plate_solve

    def fake(sol, alt, az):              # x follows azimuth, y altitude: a toy camera
        return np.array([az * 10.0]), np.array([1440.0 - alt * 15.0])
    monkeypatch.setattr(plate_solve, "altaz_to_pixel", fake)
    inside = sw.frame_test(sol={"stub": True}, frame=(2560, 1440), margin=25)
    assert inside(60.0, 100.0)               # x 1000, y 540
    assert not inside(60.0, 2.0)             # x 20: inside the 25 px margin
    assert not inside(-5.0, 100.0)           # below the horizon
    monkeypatch.setattr(plate_solve, "load", lambda *a, **k: None)   # no solution stored
    flat = sw.frame_test()
    assert flat(sw.MIN_ALT_DEG + 1, 0) and not flat(sw.MIN_ALT_DEG - 1, 0)
