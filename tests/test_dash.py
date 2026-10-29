"""The Core2 desk display's status (cmd_processing/dash.py, 2026-10-22).

Read-only by construction; these pin down what the small screen is told,
above all that it never says SHUT or OPEN on a read the gate would not trust.
"""
from datetime import datetime
from types import SimpleNamespace

from cmd_processing import dash

SUMMARY = "Tonight: m76\nRoof opens ~17:52, closes ~06:05\nm76 22:00"
PLAN = {"noon": {"plan": {"will_image": True, "slots": [
    {"dso": "ngc7790", "start": "2026-10-22T20:00-04:00"},
    {"dso": "m76", "start": "2026-10-22T22:00-04:00"}]}}, "sunset": None}


def _roof(v, at="2026-10-22 09:00"):
    return {"v": v, "at": datetime.fromisoformat(at)}


def _sum(now="2026-10-22 13:00", roof=None, scope="?", state="ARMED", disk=388.0, live=None):
    return dash.summarize(datetime.fromisoformat(now), roof, scope,
                          {"state": state} if state else None, PLAN, SUMMARY, live, disk)


def test_the_newest_read_decides_and_needs_every_frame():
    names = ["084605_0_shut.jpg", "090017_0_shut.jpg", "090017_1_shut_ir.jpg", "090017_2_shut.jpg"]
    r = dash.roof_from_names("2026-10-22", names)
    assert r["v"] == "SHUT" and r["at"] == datetime(2026, 10, 22, 9, 0, 17)
    mixed = dash.roof_from_names("2026-10-22", ["090017_0_shut.jpg", "090017_1_absent.jpg"])
    assert mixed["v"] == "?"                                     # one frame did not see it
    assert dash.roof_from_names("2026-10-22", ["090017_0_absent.jpg"])["v"] == "?"
    assert dash.roof_from_names("2026-10-22", ["notes.txt"]) is None


def test_scope_is_read_never_connected():
    def st(conn, slew=False, track=False):
        return SimpleNamespace(mount=SimpleNamespace(is_connected=conn, is_slewing=slew, is_tracking=track))
    assert dash.scope_from_status(None) == "?"
    assert dash.scope_from_status(st(False)) == "OFF"
    assert dash.scope_from_status(st(True, slew=True)) == "SLEWING"
    assert dash.scope_from_status(st(True, track=True)) == "TRACKING"
    assert dash.scope_from_status(st(True)) == "STOPPED"


def test_a_quiet_afternoon_is_all_green():
    d = _sum(roof=_roof("SHUT"))
    assert d["level"] == "g" and d["roof"] == {"v": "SHUT", "age": 240, "c": "g"}
    assert d["scope"] == {"v": "PWI4 OFF", "c": "g"}            # normal by day
    assert d["next"] == {"dso": "ngc7790", "at": "20:00"}
    assert d["tonight"]["opens"] == "17:52" and d["tonight"]["closes"] == "06:05"


def test_open_inside_the_window_is_fine_outside_it_is_flagged():
    assert _sum("2026-10-22 23:00", _roof("OPEN", "2026-10-22 22:50"), "TRACKING")["roof"]["c"] == "g"
    assert _sum("2026-10-23 02:00", _roof("OPEN", "2026-10-23 01:50"), "TRACKING")["roof"]["c"] == "g"
    d = _sum("2026-10-23 06:20", _roof("OPEN", "2026-10-23 06:15"))
    assert d["roof"]["c"] == "a" and d["level"] == "a" and "roof open off-plan" in d["alerts"]
    d = _sum("2026-10-23 08:00", _roof("OPEN", "2026-10-23 07:55"))     # 2 h past the close
    assert d["roof"]["c"] == "r" and d["level"] == "r" and "roof open late" in d["alerts"]
    assert _sum("2026-10-22 15:00", _roof("OPEN", "2026-10-22 14:55"))["roof"]["c"] == "r"
    assert _sum("2026-10-22 15:00", _roof("OPEN", "2026-10-22 14:55"), state="MANUAL_OPEN")["roof"]["c"] == "a"


def test_a_hold_promises_no_next_slot():
    assert _sum("2026-10-22 19:00", state="ESTOP")["next"] is None
    assert _sum("2026-10-22 19:00")["next"] == {"dso": "ngc7790", "at": "20:00"}


def test_no_pwi4_at_night_is_worth_a_look():
    assert _sum("2026-10-22 23:00", _roof("OPEN", "2026-10-22 22:50"))["scope"]["c"] == "a"


def test_holds_and_a_dead_conductor_are_red():
    assert _sum(state="ESTOP")["level"] == "r"
    d = _sum(state=None)
    assert d["night"] == {"v": "NO CONDUCTOR", "c": "r"} and d["level"] == "r"


def test_disk_follows_the_morning_check_thresholds():
    assert _sum(disk=150.0)["disk"]["c"] == "g"
    assert _sum(disk=60.0)["disk"]["c"] == "a"
    assert _sum(disk=20.0)["level"] == "r"


def test_after_the_last_slot_starts_there_is_no_next():
    assert _sum("2026-10-22 23:00")["next"] is None


def test_unknown_roof_is_amber():
    assert _sum(roof=None)["roof"] == {"v": "?", "age": None, "c": "a"}


# Pages 2-4 (2026-10-25): the night's FWHM, the newest picture, last night's summary.
def _row(dso, night, stamp, filt, fw, stars=400):
    return {"path": "C:/T/%s/cdk17/%s/LIGHT/%s_%s_1_300.00s_0001.fits" % (dso, night, stamp, filt),
            "filter": filt, "fwhm_arcsec": fw, "star_count": stars}


ROWS = {"m76": [_row("m76", "2026-10-24", "2026-10-24_22-05-00", "Ha", 1.80),
                _row("m76", "2026-10-24", "2026-10-25_01-45-00", "O-III", 1.45),
                _row("m76", "2026-10-23", "2026-10-23_23-00-00", "Ha", 1.90),    # the night before
                "not a dict", {"path": "x", "fwhm_arcsec": None}],
        "m1": [_row("m1", "2026-10-24", "2026-10-25_04-10-00", "S-II", 2.10)]}


def test_night_frames_keeps_the_nights_own_frames_in_time_order():
    f = dash.night_frames(ROWS, "2026-10-24")
    assert [(x[0].strftime("%d %H:%M"), x[1], x[2]) for x in f] == [
        ("24 22:05", "m76", "Ha"), ("25 01:45", "m76", "O-III"), ("25 04:10", "m1", "S-II")]


def test_fwhm_series_is_compact_and_marks_the_slots():
    s = dash.fwhm_series(dash.night_frames(ROWS, "2026-10-24"))
    assert s["start"] == "22:05" and s["f"] == ["Ha", "O-III", "S-II"]
    assert s["p"] == [[0, 180, 0], [220, 145, 1], [365, 210, 2]]
    assert s["slots"] == [[0, "m76"], [365, "m1"]] and s["med"] == 180 and s["last"] == [210, "S-II"]
    assert dash.fwhm_series([]) is None


def test_fwhm_series_thins_a_long_night():
    from datetime import timedelta
    t0 = datetime(2026, 10, 24, 20, 0)
    frames = [(t0 + timedelta(minutes=5 * i), "m76", "Ha", 2.0, 300) for i in range(400)]
    assert len(dash.fwhm_series(frames)["p"]) <= dash.MAX_POINTS


def test_night_summary_counts_filters_and_names_the_sharpest_frame():
    s = dash.night_summary(dash.night_frames(ROWS, "2026-10-24"), "2026-10-24")
    assert s["total"] == 3 and [t["dso"] for t in s["targets"]] == ["m76", "m1"]
    assert s["targets"][0]["n"] == {"Ha": 1, "O-III": 1}
    assert s["best"] == {"fwhm": 1.45, "dso": "m76", "filter": "O-III", "at": "01:45"}


# The sky camera line (2026-10-25).
def test_sky_label_reads_a_clear_sky_and_admits_what_the_moon_hides():
    from datetime import timezone, timedelta
    now = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)
    row = {"captured": "2026-10-25T05:58:00+00:00", "night": True, "solve_matches": 120}
    assert dash.sky_label(row, now, 200, 0.75, False) == {"v": "80%", "c": "g", "moon": False}
    assert dash.sky_label(dict(row, solve_matches=60), now, 200, 0.75, False)["c"] == "a"
    assert dash.sky_label(dict(row, solve_matches=10), now, 200, 0.75, False)["c"] == "x"
    assert dash.sky_label(dict(row, solve_matches=60), now, 200, 0.75, True) == {"v": "40%", "c": "w", "moon": True}
    assert dash.sky_label(dict(row, night=False), now, 200, 0.75, False)["v"] == "day"
    assert dash.sky_label(row, now + timedelta(minutes=30), 200, 0.75, False)["v"] == "no sky cam"
    assert dash.sky_label(None, now, 200, 0.75, False) is None


# The big-type imaging pages (2026-10-26).
def test_imaging_filters_count_and_judge_the_running_target_only():
    rows = {"m1": [_row("m1", "2026-10-25", "2026-10-26_02-10-00", "S-II", 1.90),
                   _row("m1", "2026-10-25", "2026-10-26_02-20-00", "Ha", 1.80),
                   _row("m1", "2026-10-25", "2026-10-26_02-30-00", "Ha", 1.60),
                   _row("m1", "2026-10-25", "2026-10-26_02-40-00", "Ha", 1.70)],
            "melotte15": [_row("melotte15", "2026-10-25", "2026-10-25_23-10-00", "Ha", 2.0)]}
    f = dash.imaging_filters(dash.night_frames(rows, "2026-10-25"), "m1")
    assert f == [{"f": "Ha", "n": 3, "med": 1.7, "last": 1.7},
                 {"f": "S-II", "n": 1, "med": 1.9, "last": 1.9}]
    assert dash.imaging_filters([], "m1") == []


def test_flats_and_stale_frames_are_not_imaging():
    # 2026-10-27 06:43: IN_FLATS, watcher still beating, newest light 69 min old.
    from datetime import timedelta
    now = datetime(2026, 10, 27, 10, 43)
    assert not dash.frames_coming_in("IN_FLATS", now - timedelta(minutes=69), now)
    assert not dash.frames_coming_in("IN_MAIN", now - timedelta(minutes=45), now)     # nothing for 45 min
    assert dash.frames_coming_in("IN_MAIN", now - timedelta(minutes=4), now)
    assert dash.frames_coming_in("DONE_MAIN", now - timedelta(minutes=10), now)      # slot hand-over
    assert not dash.frames_coming_in("IN_MAIN", None, now)


def test_sky_night_summary_gives_the_median_and_the_newest():
    assert dash.sky_night_summary([], False) is None
    assert dash.sky_night_summary([40, 55, 61, 52, 48], True) == {"med": 52, "last": 48, "n": 5, "moon": True}


def test_the_night_runs_from_noon_to_noon():
    # 2026-10-28: readings after midnight were dropped; the page froze at 22/25%.
    from datetime import date
    ev = date(2026, 10, 27)
    assert dash.in_night(datetime(2026, 10, 27, 21, 0), ev)
    assert dash.in_night(datetime(2026, 10, 28, 3, 15), ev)            # after midnight: still tonight
    assert not dash.in_night(datetime(2026, 10, 28, 12, 0), ev)
    assert not dash.in_night(datetime(2026, 10, 27, 11, 59), ev)


def test_sky_view_picks_the_newest_frame_in_either_folder(tmp_path):
    """The display's sky page (2026-10-29): bursts move frames to keep/."""
    from cmd_processing import dash
    (tmp_path / "keep").mkdir()
    for n in ("sky_20261028T235958Z.jpg", "note.txt"):
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "keep" / "sky_20261029T010458Z.jpg").write_bytes(b"x")
    p = dash.newest_sky_frame(tmp_path)
    assert p.name == "sky_20261029T010458Z.jpg" and p.parent.name == "keep"
    info = dash.sky_view_info(p)
    assert info["id"] == "20261029T010458" and len(info["at"]) == 5
    assert dash.newest_sky_frame(tmp_path / "missing") is None
    assert dash.sky_view_info(None) is None


def test_sky_view_jpeg_is_square_and_stretched(tmp_path):
    import io
    import numpy as np
    from PIL import Image
    from cmd_processing import dash
    a = np.full((1440, 2560, 3), 20, np.uint8)
    a[700:720, 1270:1290] = 60                          # a faint blob on a dark sky
    p = tmp_path / "sky_20261029T010458Z.jpg"
    Image.fromarray(a).save(p)
    im = Image.open(io.BytesIO(dash.view_jpeg(p)))
    assert im.size == (240, 240)
    assert np.asarray(im).max() > 200                    # stretched: the blob comes up bright
