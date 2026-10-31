"""The sky page's "catalogue stars seen" counts both sides over the same stars.

2026-10-31: the page read 113 % of a clear sky. plate_solve.verify matched the
whole catalogue (to V 6) and the denominator counted only V <= 5, so forty-odd
faint matches were never in the total. verify(mag_limit=) cuts the catalogue
before matching; the page's line is now plate_solve.seen_to_mag over the
completeness table's rows, so it is literally rows 2 to 5 added up.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from sentry import plate_solve as ps   # noqa: E402


def _sol():
    return {"R": np.eye(3).tolist(), "cx": 1280.0, "cy": 720.0, "F": 600.0,
            "model": "equi", "flip": True, "profile": "sky camera"}


def _catalogue():
    # Six stars near the zenith, three bright and three faint; all land in frame.
    alt = np.array([85.0, 80.0, 75.0, 82.0, 78.0, 70.0])
    az = np.array([10.0, 100.0, 200.0, 300.0, 40.0, 250.0])
    vmag = np.array([2.0, 3.5, 4.9, 5.2, 5.6, 5.9])
    return alt, az, vmag, np.zeros(6), np.zeros(6)


def test_mag_limit_counts_only_the_bright_matches(monkeypatch, tmp_path):
    alt, az, vmag, ra, dec = _catalogue()
    sol = _sol()
    monkeypatch.setattr(ps.bright_stars, "above_horizon", lambda when: (alt, az, vmag, ra, dec))
    monkeypatch.setattr(ps.star_count, "_load", lambda frame: np.zeros((1440, 2560)))
    x, y, _ = ps.cam_to_pix(ps.unit_from_altaz(az, alt) @ np.asarray(sol["R"]).T,
                            sol["cx"], sol["cy"], sol["F"], sol["model"], sol["flip"])
    dets = [{"x": float(a), "y": float(b)} for a, b in zip(x, y)]     # every star detected
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"")
    assert ps.verify(sol, frame, when="x", dets=dets)["matches"] == 6
    assert ps.verify(sol, frame, when="x", dets=dets, mag_limit=5.0)["matches"] == 3
    # A faint star found and a bright one missed: the V 5 count drops, the full count does not.
    dets2 = dets[1:]
    assert ps.verify(sol, frame, when="x", dets=dets2)["matches"] == 5
    assert ps.verify(sol, frame, when="x", dets=dets2, mag_limit=5.0)["matches"] == 2


def test_the_sky_line_is_the_tables_rows_two_to_five_added_up():
    # The owner's table of 2026-10-31 02:30: rows 2-5 sum to 110 of 193, row 6 is left out.
    rows = [{"vmag": 2.0, "n": 5, "found": 5}, {"vmag": 3.0, "n": 11, "found": 11},
            {"vmag": 4.0, "n": 44, "found": 42}, {"vmag": 5.0, "n": 133, "found": 52},
            {"vmag": 6.0, "n": 145, "found": 10}]
    assert ps.seen_to_mag(rows, 5.0) == {"n": 193, "found": 110}
    assert ps.seen_to_mag([], 5.0) == {"n": 0, "found": 0}
