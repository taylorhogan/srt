"""scripts/live_skymap: the Live tab's "Every frame" panel gets a second row for
tonight's frames (owner, 2026-11-18). The counting is pure: per filter, biggest
first, and with a night only that observing night's frames, nights being the
frame time shifted back twelve hours so a run past midnight is one night."""
from datetime import date, datetime

import pytest

pytest.importorskip("numpy")          # live_skymap draws with matplotlib; the CI interpreter is pytest-only
pytest.importorskip("matplotlib")

from scripts import live_skymap as ls  # noqa: E402


def _f(t, filt):
    return {"time": t, "filter": filt}


FRAMES = [_f("2026-11-10T23:05:00", "Ha"), _f("2026-11-11T00:40:00", "S-II"),          # night of the 10th
          _f("2026-11-16T23:23:00", "Ha"), _f("2026-11-17T03:33:00", "Ha"),            # night of the 16th
          _f("2026-11-17T01:36:00", "S-II"), _f("2026-11-17T02:50:00", "O-III"),
          _f("2026-11-18T04:00:00", "Ha"), _f("2026-11-18T07:38:00", "O-III"),         # night of the 17th
          {"time": None, "filter": "Ha"}, {"time": "2026-11-18T05:00:00", "filter": ""}]


def test_counts_all_and_by_night_biggest_first():
    assert ls.frame_counts(FRAMES) == {"Ha": 5, "S-II": 2, "O-III": 2}
    assert ls.frame_counts(FRAMES, date(2026, 11, 16)) == {"Ha": 2, "S-II": 1, "O-III": 1}
    assert ls.frame_counts(FRAMES, date(2026, 11, 17)) == {"Ha": 1, "O-III": 1}
    assert ls.frame_counts(FRAMES, date(2026, 11, 18)) == {}


def test_tonight_is_the_twelve_hour_shifted_night():
    assert ls.tonight_night(datetime(2026, 11, 18, 7, 43)) == date(2026, 11, 17)      # 02:43 EST, the 17th's night
    assert ls.tonight_night(datetime(2026, 11, 18, 13, 0)) == date(2026, 11, 18)      # the next evening
    assert ls._night_of("2026-11-18T07:38:01.731000") == date(2026, 11, 17)


def test_the_row_shows_last_night_through_the_day_and_hands_over_at_the_first_frame():
    # 02:43 EST on the 18th: the night of the 17th is in progress and has frames
    assert ls.latest_night_counts(FRAMES, datetime(2026, 11, 18, 7, 43)) == \
        ("tonight", date(2026, 11, 17), {"Ha": 1, "O-III": 1})
    # noon on the 18th: "tonight" is the 18th's night, no frame yet -> last night's row
    assert ls.latest_night_counts(FRAMES, datetime(2026, 11, 18, 17, 0)) == \
        ("last night", date(2026, 11, 17), {"Ha": 1, "O-III": 1})
    # the 18th's first frame hands the row over
    more = FRAMES + [_f("2026-11-18T23:10:00", "S-II")]
    assert ls.latest_night_counts(more, datetime(2026, 11, 19, 4, 11)) == \
        ("tonight", date(2026, 11, 18), {"S-II": 1})
    # a week of cloud: neither night has a frame, no row
    assert ls.latest_night_counts(FRAMES, datetime(2026, 11, 25, 17, 0)) == ("tonight", date(2026, 11, 25), {})
