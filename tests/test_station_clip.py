"""The clip cut from a station recording: its window, its crop, its caption.

Pure arithmetic on the fit, so no video is needed; the module imports cv2 at
the top, so this skips on the bare CI runner like the other camera tests.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("numpy")
pytest.importorskip("cv2")
import station_track as st  # noqa: E402


# ------------------------------------------------------------- clip_window

def test_window_pads_both_sides_at_the_frame_rate():
    a, b = st.clip_window(t_first=40.0, t_last=63.0, fps=15.0, nframes=3000, pad_s=5.0)
    assert a == 35 * 15
    assert b == 68 * 15 + 1


def test_window_is_clamped_to_the_recording():
    a, b = st.clip_window(2.0, 10.0, 15.0, nframes=200, pad_s=5.0)
    assert a == 0 and b == 200


def test_long_track_is_trimmed_about_its_middle():
    a, b = st.clip_window(10.0, 170.0, 15.0, nframes=3000, pad_s=5.0, max_s=60.0)
    assert b - a == 60 * 15
    assert abs(((a + b) / 2.0) / 15.0 - 90.0) < 1.0


# ---------------------------------------------------------------- crop_box

def test_crop_is_square_holds_the_track_and_stays_inside_the_frame():
    x, y, side = st.crop_box(2381.7, 665.9, 2334.4, 1151.7, 2560, 1440)
    assert side == 800                      # short track: the minimum size wins
    assert x + side <= 2560 and y + side <= 1440 and x >= 0 and y >= 0
    for px, py in ((2381.7, 665.9), (2334.4, 1151.7)):
        assert x <= px <= x + side and y <= py <= y + side


def test_long_track_grows_the_crop_up_to_the_frame():
    x, y, side = st.crop_box(100, 700, 2400, 720, 2560, 1440)
    assert side == 1440                     # wants 2540 but the frame is 1440 tall
    assert y == 0


# ------------------------------------------------------------ summary_line

FOUND = {"sat": "Tiangong", "peak": "2026-09-08T20:48:17-04:00", "peak_alt_deg": 50.3,
         "is_satellite": True, "frames_agreeing": 281, "deg_per_s": 0.698, "arc_deg": 16.09,
         "alt_first": 49.6, "az_first": 214.2, "alt_last": 50.9, "az_last": 189.0,
         "checked_against_tle": True, "miss_deg_median": 3.5}


def test_found_line_carries_the_verdict_and_the_numbers():
    s = st.summary_line(FOUND)
    assert s.startswith("Tiangong pass 2026-09-08 20:48, peak alt 50 deg: FOUND.")
    assert "281 frames agree" in s and "0.70 deg/s" in s and "16 deg arc" in s
    assert "within 4 deg of the TLE" in s


def test_rejected_line_says_no_and_why():
    s = st.summary_line({**FOUND, "is_satellite": False,
                         "reject_reason": "0.145 deg/s is too slow for anything in low orbit"})
    assert "no Tiangong found -- 0.145 deg/s is too slow" in s
    assert "FOUND" not in s


def test_no_track_dict_is_handled():
    s = st.summary_line({"sat": "ISS", "peak": "2026-09-23T05:10:00-04:00",
                         "peak_alt_deg": 67.0, "is_satellite": False})
    assert "no ISS found -- no track." in s
