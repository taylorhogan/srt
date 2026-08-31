"""Registry audit — pure-fixture tests of the Phase 1 gate's second half.

No config, no disk, no network: every case injects queue rows, a registry
snapshot, and converged/frames answers as plain data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iris.conductor.audit import (  # noqa: E402
    audit_registry, expected_state, find_inconsistencies, is_resolved, render)


def _row(dso, status="waiting", **kw):
    return {"dso": dso, "status": status, **kw}


NEVER = lambda _:_ and False       # noqa: E731  converged_of / has_frames_of
ALWAYS = lambda _: True            # noqa: E731


# ------------------------------------------------------------ expected_state

def test_lifecycle_precedence():
    r = _row("x", ra_deg=1.0, dec_deg=2.0)
    assert expected_state(_row("x", "completed"), True, True) == "RETIRED"
    assert expected_state(r, True, True) == "CONVERGED"
    assert expected_state(r, False, True) == "ACQUIRING"
    assert expected_state(r, False, False) == "QUEUED"
    assert expected_state(_row("x"), False, False) == "WISHED"


def test_zero_above_horizon_is_not_resolved():
    """'0' is the resolver FAILING, not a resolution — the truthiness trap."""
    assert not is_resolved(_row("x", above_horizon="0"))
    assert is_resolved(_row("x", above_horizon="5:33:20"))
    assert is_resolved(_row("x", ra_deg=317.9, dec_deg=59.9))


# ------------------------------------------------------------ audit_registry

def test_clean_registry_audits_clean():
    rows = [_row("a", ra_deg=1, dec_deg=2), _row("b", "completed")]
    reg = {"a": {"state": "QUEUED"}, "b": {"state": "RETIRED"}}
    out = audit_registry(rows, reg, NEVER, NEVER)
    assert out["mismatches"] == [] and out["missing"] == [] and out["extra"] == []
    assert out["counts"] == {"QUEUED": 1, "RETIRED": 1}


def test_mismatch_missing_and_extra_reported():
    rows = [_row("a", ra_deg=1, dec_deg=2), _row("b")]
    reg = {"a": {"state": "WISHED"}, "ghost": {"state": "QUEUED"}}
    out = audit_registry(rows, reg, NEVER, NEVER)
    assert out["mismatches"] == [("a", "WISHED", "QUEUED")]
    assert out["missing"] == ["b"]
    assert out["extra"] == ["ghost"]


# ------------------------------------------------------ find_inconsistencies

def test_auto_stop_candidate_flagged():
    rows = [_row("done1", "waiting", above_horizon="1:00:00")]
    msgs = find_inconsistencies(rows, {}, ALWAYS)
    assert any("AUTO-STOP" in m for m in msgs)


def test_retired_early_flagged_only_when_calibrated():
    conv = {"early": {"Ha": {"calibrated": True}}}
    msgs = find_inconsistencies([_row("early", "completed")], conv, NEVER)
    assert any("retired early" in m for m in msgs)
    conv_old = {"early": {"Ha": {}}}          # pedestal-scale entry: unknowable
    msgs = find_inconsistencies([_row("early", "completed")], conv_old, NEVER)
    assert not any("retired early" in m for m in msgs)


def test_duplicates_and_orphans_flagged():
    rows = [_row("sh2-129", above_horizon="1:00:00"),
            _row("sh2-129", above_horizon="1:00:00")]
    conv = {"neverqueued": {"Ha": {"calibrated": True}}}
    msgs = find_inconsistencies(rows, conv, NEVER)
    assert any("duplicate" in m for m in msgs)
    assert any("neverqueued" in m and "no queue row" in m for m in msgs)


def test_narrowband_history_without_obj_type_flagged():
    conv = {"wizard": {"Ha": {"calibrated": True}, "S-II": {"calibrated": True}}}
    msgs = find_inconsistencies([_row("wizard", above_horizon="1:00:00")],
                                conv, NEVER)
    assert any("obj_type" in m for m in msgs)
    ok = _row("wizard", above_horizon="1:00:00", obj_type="nebula")
    msgs = find_inconsistencies([ok], conv, NEVER)
    assert not any("obj_type" in m for m in msgs)


def test_unresolvable_flagged():
    msgs = find_inconsistencies([_row("mystery", above_horizon="0")], {}, NEVER)
    assert any("unresolvable" in m for m in msgs)


# ------------------------------------------------------------------- render

def test_render_verdicts_and_ascii():
    rows = [_row("a", ra_deg=1, dec_deg=2)]
    clean = audit_registry(rows, {"a": {"state": "QUEUED"}}, NEVER, NEVER)
    txt = render(clean, [], "test")
    assert "Registry verdict: CLEAN" in txt
    bad = audit_registry(rows, {"a": {"state": "WISHED"}}, NEVER, NEVER)
    txt = render(bad, ["a finding"], "test")
    assert "Registry verdict: MISMATCHED" in txt and "a finding" in txt
    assert txt.encode("ascii", errors="strict")   # cp1252-console safe
