"""Golden-reference drift detection for the roof.

The rolling good libraries measurably follow the roof (logged good peak_w
mean rose 351 W -> 398 W in twelve days), so slow degradation never trips
them. These tests pin the two pieces that make drift visible: the golden
envelope comparison (a frozen library fed to the existing compare()) and the
morning report's verdict line, whose defining case is "rolling says fine,
golden says drifted".
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.shadow_report import _drift_lines


def _rows(golden_oks, rolling_oks, kind="current", direction="close"):
    now = datetime.now().astimezone()
    return [{"t": (now - timedelta(hours=i)).isoformat(timespec="seconds"),
             "kind": kind, "direction": direction,
             "rolling_ok": r, "golden_ok": g,
             "summary": "peak_w 412 vs golden 354±14"}
            for i, (g, r) in enumerate(reversed(list(zip(golden_oks, rolling_oks))))]


def _write(tmp_path, rows):
    p = tmp_path / "roof_drift.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_healthy_moves_report_ok(tmp_path):
    p = _write(tmp_path, _rows([True] * 5, [True] * 5))
    lines = _drift_lines(p)
    assert any("5/5 vs golden OK" in ln for ln in lines)


def test_slow_degradation_reads_drift_not_fault(tmp_path):
    """The defining case: every move passes the ROLLING check (the library
    has followed the roof down) while failing golden -- that is DRIFT."""
    p = _write(tmp_path, _rows([True, False, False, False, False], [True] * 5))
    lines = _drift_lines(p)
    assert any("DRIFT" in ln for ln in lines)
    assert not any("FAULTY" in ln for ln in lines)


def test_failing_both_references_reads_faulty(tmp_path):
    p = _write(tmp_path, _rows([False] * 5, [False] * 5))
    lines = _drift_lines(p)
    assert any("FAULTY" in ln for ln in lines)


def test_no_golden_library_rows_are_skipped(tmp_path):
    rows = _rows([None] * 5, [True] * 5)
    p = _write(tmp_path, rows)
    assert _drift_lines(p) == []


def test_missing_file_is_silent(tmp_path):
    assert _drift_lines(tmp_path / "absent.jsonl") == []


def test_golden_envelope_flags_what_the_drifted_library_absorbs():
    """compare() against a frozen library must flag a value the accumulated
    library would accept -- the numeric heart of the whole feature."""
    np = pytest.importorskip("numpy")  # noqa: F841 — module needs it
    from sentry.roof_current_signature import compare

    def sig(peak):
        return {"direction": "close",
                "features": {"valid": True, "peak_w": peak,
                             "running_w": peak * 0.8, "running_a": peak / 120.0,
                             "move_duration_s": 42.0, "energy_ws": peak * 40.0,
                             "returned_to_baseline": True}}
    golden = [sig(350), sig(352), sig(354), sig(351)]     # healthy era
    drifted = [sig(v) for v in (350, 370, 390, 405, 415)]  # absorbed the drift
    probe = sig(415)
    assert compare(probe, library=drifted)["is_anomaly"] is False
    res = compare(probe, library=golden)
    assert res["is_anomaly"] is True
    assert any("peak_w" in r for r in res["reasons"])
