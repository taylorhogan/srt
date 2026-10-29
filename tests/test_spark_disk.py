"""The Spark's disk floor and the stale-stack prune (docs/AUTOMATION_PLAN.md B6).

utils.disk_space.band is the one word everything decides on; morning_trigger's
verdict and once-a-day memory are pure; spark_prune deletes only old cached
stacks and refuses anything that looks like data.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import disk_space as D  # noqa: E402
from scripts import spark_prune as P  # noqa: E402
from scripts import morning_trigger as T  # noqa: E402
from scripts import morning_watch as W  # noqa: E402


def test_band_is_one_word():
    assert D.band({"free_gb": 150.0, "total_gb": 3700.0}, 400, 200) == "refuse"
    assert D.band({"free_gb": 300.0, "total_gb": 3700.0}, 400, 200) == "warn"
    assert D.band({"free_gb": 2500.0, "total_gb": 3700.0}, 400, 200) == "ok"
    assert D.band(None, 400, 200) == "unknown"


def test_the_spark_floors_come_from_the_machine_block(monkeypatch):
    """Pure on the overlay: CI runs with the config stubbed, so the live
    config.data() is not consulted here."""
    from configs import config
    cfg = {"disk": {"warn_free_gb": 100, "refuse_free_gb": 25},
           "machine": {"spark-3129": {"disk_warn_free_gb": 400, "disk_refuse_free_gb": 200,
                                      "stack_keep_days": 45},
                       "iris-pc": {"image_dir": "C:/x"}}}
    monkeypatch.setattr(config.socket, "gethostname", lambda: "spark-3129")
    config._apply_machine_overrides(cfg)
    assert D.thresholds(cfg) == (400.0, 200.0) and cfg["disk"]["stack_keep_days"] == 45
    pc = {"disk": {"warn_free_gb": 100, "refuse_free_gb": 25}, "machine": cfg["machine"]}
    monkeypatch.setattr(config.socket, "gethostname", lambda: "iris-pc")
    config._apply_machine_overrides(pc)
    assert D.thresholds(pc) == (100.0, 25.0) and "stack_keep_days" not in pc["disk"]


def test_the_trigger_verdict_says_what_is_skipped():
    band, msg = T.disk_verdict({"free_gb": 150.0, "total_gb": 3700.0}, 400, 200)
    assert band == "refuse" and "SKIPPED" in msg and "spark_prune" in msg
    band, msg = T.disk_verdict({"free_gb": 300.0, "total_gb": 3700.0}, 400, 200)
    assert band == "warn" and "getting full" in msg
    assert T.disk_verdict(None, 400, 200)[0] == "unknown"
    assert T.disk_verdict({"free_gb": 2500.0, "total_gb": 3700.0}, 400, 200)[1].startswith("disk ok")


def test_once_today_remembers_across_runs(tmp_path):
    p = tmp_path / "state.json"
    assert T.once_today(p, "disk:warn", "2026-10-29") is True
    assert T.once_today(p, "disk:warn", "2026-10-29") is False
    assert T.once_today(p, "disk:refuse", "2026-10-29") is True
    assert T.once_today(p, "disk:warn", "2026-10-30") is True
    assert json.loads(p.read_text())["disk:refuse"] == "2026-10-29"


def _stack(d, name, built_days_ago, now, sidecar=True):
    npy = d / f"{name}_raw.npy"
    npy.write_bytes(b"0" * 1000)
    stamp = now - built_days_ago * 86400
    if sidecar:
        from datetime import datetime
        (d / f"{name}_raw.json").write_text(json.dumps(
            {"built": datetime.fromtimestamp(stamp).strftime("%Y-%m-%dT%H:%M"), "frames": 10}))
    else:
        os.utime(npy, (stamp, stamp))
    return npy


def test_prune_lists_only_stacks_older_than_the_keep(tmp_path):
    now = time.time()
    d = tmp_path / "local" / "n2n_lrgb_render"
    d.mkdir(parents=True)
    old = _stack(d, "ngc5907_L", 60, now)
    _stack(d, "ic410_Ha", 2, now)
    nosc = _stack(d, "m1_O-III", 90, now, sidecar=False)
    (d / "ic410_SHO_raw.jpg").write_bytes(b"x")
    stale = P.prune_stale_stacks(d, 45, now=now)
    assert [p.name for p, _, _ in stale] == ["m1_O-III_raw.npy", "ngc5907_L_raw.npy"]
    assert all(age > 45 for _, _, age in stale)
    freed = P.delete_stack(old) + P.delete_stack(nosc)
    assert freed > 2000 and not old.exists() and not (d / "ngc5907_L_raw.json").exists()
    assert (d / "ic410_Ha_raw.npy").exists()


def test_prune_refuses_data_paths(tmp_path):
    import pytest
    light = tmp_path / "Targets" / "m1" / "2026-10-29" / "LIGHT"
    light.mkdir(parents=True)
    f = light / "m1_raw.npy"
    f.write_bytes(b"x")
    assert P.protected(f) is True
    assert P.protected(tmp_path / "local" / "models" / "x_raw.npy") is True
    assert P.protected(tmp_path / "local" / "n2n_lrgb_render" / "x_raw.npy") is False
    assert P.protected(tmp_path / "elsewhere" / "x_raw.npy", image_dir=tmp_path / "elsewhere") is True
    with pytest.raises(PermissionError):
        P.delete_stack(f)
    assert P.prune_stale_stacks(light, 0, now=time.time() + 1e6) == []
    assert P.prune_stale_stacks(tmp_path / "nowhere", 45) == []


def test_the_report_names_what_is_never_pruned(tmp_path):
    text = P.report([(tmp_path / "a_raw.npy", 2e9, 60.0)], 45, "/home/taylor/Desktop/Targets")
    assert "1 stale stack" in text and "2.00 GB" in text and "never pruned" in text


def test_morning_watch_reads_the_run_record():
    from datetime import date
    today = date(2026, 10, 29)
    assert W.evaluate(None, today) and "no record" in W.evaluate(None, today)[0]
    assert "from 2026-10-28" in W.evaluate({"date": "2026-10-28", "render_rc": 0}, today)[0]
    assert W.evaluate({"date": "2026-10-29", "render_rc": 0, "sync_ok": True}, today) == []
    p = W.evaluate({"date": "2026-10-29", "render_rc": 1, "sync_ok": False}, today)
    assert len(p) == 2 and "exited 1" in p[0] and "sync failed" in p[1]
    p = W.evaluate({"date": "2026-10-29", "skipped": "disk nearly full"}, today)
    assert p == ["the morning run was skipped: disk nearly full"]
