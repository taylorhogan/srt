"""Disk space on the observatory PC (2026-10-11): alarms, the master-cache prune, `purge bad`.

On 2026-10-11 C: had 377 GB free with ~60-90 GB written a month and nothing
pruning the raw frames, the stacking cache (47 GB) or rejected frames (20 GB).
"""
import os
import time
from pathlib import Path

import pytest

from utils import disk_space as D

DAY = 86400.0


def test_thresholds_default_and_configured():
    assert D.thresholds({}) == (100.0, 25.0)
    assert D.thresholds({"disk": {"warn_free_gb": 80, "refuse_free_gb": 30}}) == (80.0, 30.0)
    assert D.bad_keep_days({}) == 30.0 and D.bad_keep_days({"disk": {"bad_keep_days": 7}}) == 7.0


def test_data_path_walks_up_to_an_existing_directory(tmp_path):
    assert D.data_path({"nina": {"image_dir": str(tmp_path / "not" / "yet")}}) == str(tmp_path)
    assert os.path.isdir(D.data_path({}))                      # falls back to the repo's drive


def test_usage(tmp_path):
    u = D.usage(tmp_path)
    assert u["free_gb"] > 0 and u["total_gb"] >= u["free_gb"]
    assert D.usage(tmp_path / "missing") is None


def _u(free, total=930.0):
    return {"path": "C:/", "free_gb": free, "total_gb": total}


def test_morning_status():
    assert D.morning_status(_u(377), 100, 25) == (None, None)
    prob, warn = D.morning_status(_u(80), 100, 25)
    assert prob is None and "80 GB free" in warn
    prob, warn = D.morning_status(_u(20), 100, 25)
    assert "nearly full" in prob and warn is None
    prob, warn = D.morning_status(None, 100, 25)
    assert prob is None and "could not read" in warn


def test_image_refusal():
    assert D.image_refusal(_u(377), 25) is None
    assert D.image_refusal(None, 25) is None                   # unreadable never blocks a night
    assert "only 20 GB free" in D.image_refusal(_u(20), 25)


def test_check_image_run_reads_the_configured_drive(tmp_path, monkeypatch):
    seen = []

    def fake(path):
        seen.append(path)
        return type("U", (), {"free": 10e9, "total": 930e9, "used": 920e9})()
    monkeypatch.setattr(D.shutil, "disk_usage", fake)
    why = D.check_image_run({"nina": {"image_dir": str(tmp_path)}})
    assert seen == [str(tmp_path)] and "only 10 GB free" in why


def _touch(p: Path, age_days: float, size=10):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    t = time.time() - age_days * DAY
    os.utime(p, (t, t))


def test_stale_bad_frames_only_old_rejected_lights(tmp_path):
    old = tmp_path / "m31" / "2026-08-01" / "LIGHT" / "a.fits.bad"
    new = tmp_path / "m31" / "2026-10-10" / "LIGHT" / "b.fits.bad"
    flat = tmp_path / "cdk17" / "2026-08-01" / "FLAT" / "f.fits.bad"     # not a light
    _touch(old, 40)
    _touch(new, 2)
    _touch(flat, 40)
    assert [p for p, _ in D.stale_bad_frames(tmp_path, 30)] == [old]


# ------------------------------------------------------------ the morning check

def test_morning_check_reports_the_disk():
    mc = pytest.importorskip("scripts.morning_check")
    vision = {"camera": True, "closed": True, "parked": True, "scope_verdict": "safe",
              "frames": 3, "scope_tag_frames": 3}
    kasa = {"resolved": {n: 0 for n in mc.KASA_DEVICES}, "error": None}
    peg = {1: 0, 2: 0, 3: 0}
    assert mc.evaluate(vision, kasa, peg) == ([], [])          # not measured: silent
    disk = {"usage": _u(20), "warn_gb": 100.0, "refuse_gb": 25.0}
    problems, _ = mc.evaluate(vision, kasa, peg, disk=disk)
    assert any("nearly full" in p for p in problems)
    disk["usage"] = _u(80)
    problems, warnings = mc.evaluate(vision, kasa, peg, disk=disk)
    assert problems == [] and any("disk getting full" in w for w in warnings)


# ------------------------------------------------------------ the master cache

def test_master_cache_prunes_only_unused_masters(tmp_path):
    stacker = pytest.importorskip("stacking.stacker")
    stale = tmp_path / "master_flat_L_aaaa.npy"
    used = tmp_path / "master_bias_bbbb.npy"
    fresh = tmp_path / "master_dark_cccc.npy"
    other = tmp_path / "stack_m31_L.npy"                       # not a cached master
    for p in (stale, used, other):
        _touch(p, 40)
    _touch(fresh, 3)
    stacker._cache_hit(used)                                   # reused today: in service
    n, freed = stacker.prune_master_cache(tmp_path, keep_days=30)
    assert n == 1 and freed == 10 and not stale.exists()
    assert used.exists() and fresh.exists() and other.exists()


# ------------------------------------------------------------ image!! and purge bad

def _suc():
    return pytest.importorskip("cmd_processing.super_user_commands")


def test_image_cmd_refuses_on_a_full_disk(monkeypatch):
    suc = _suc()
    posted, pushed = [], []
    monkeypatch.setattr(suc, "is_imaging", lambda: False)
    monkeypatch.setattr(suc, "is_nina_running", lambda: False)
    monkeypatch.setattr(D, "check_image_run", lambda cfg: "only 10 GB free on the image drive")
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: posted.append(m))
    monkeypatch.setattr(suc.pushover, "push_message", lambda m: pushed.append(m))
    monkeypatch.setattr(suc.jobs, "spawn", lambda fn: pytest.fail("must not launch"))
    assert suc.image_cmd(["", "image!!", "1"], "iris") is False
    assert "REFUSED" in posted[0] and "10 GB free" in posted[0] and pushed


def test_purge_bad_is_a_dry_run_until_go(tmp_path, monkeypatch):
    suc = _suc()
    old = tmp_path / "m31" / "2026-08-01" / "LIGHT" / "a.fits.bad"
    new = tmp_path / "m31" / "2026-10-10" / "LIGHT" / "b.fits.bad"
    _touch(old, 40)
    _touch(new, 2)
    posted = []
    monkeypatch.setattr(suc.config, "data", lambda: {"nina": {"image_dir": str(tmp_path)}})
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: posted.append(m))
    monkeypatch.setattr(suc.jobs, "get_current_job", lambda: None)
    monkeypatch.setattr(suc.jobs, "raise_if_cancelled", lambda job: None)
    suc._purge_bad_run(["", "purge", "bad"])
    assert old.exists() and "m31: 1 frame" in posted[-1] and "purge bad go" in posted[-1]
    suc._purge_bad_run(["", "purge", "bad", "go"])
    assert not old.exists() and new.exists() and "Deleted 1/1" in posted[-1]
