"""`seen` draws the site's number (seen_pct) where the sky monitor wrote it (2026-11-01)."""
from datetime import datetime, timedelta

from scripts.sky_transparency_report import night_points

T0 = datetime(2026, 11, 1, 5, 0)


def test_a_night_with_seen_pct_uses_it_and_only_those_rows():
    rows = [(T0, {"solve_matches": 150}),                                   # before the field
            (T0 + timedelta(minutes=5), {"solve_matches": 150, "seen_pct": 63}),
            (T0 + timedelta(minutes=10), {"solve_matches": 154, "seen_pct": 61})]
    pts, site = night_points(rows, lambda w: 190)
    assert site is True
    assert [p for _w, _r, p in pts] == [63.0, 61.0]


def test_an_older_night_keeps_the_old_index():
    rows = [(T0, {"solve_matches": 95}), (T0 + timedelta(minutes=5), {"solve_matches": 10})]
    pts, site = night_points(rows, lambda w: 190 if w == T0 else 0)
    assert site is False
    assert [round(p) for _w, _r, p in pts] == [50]
