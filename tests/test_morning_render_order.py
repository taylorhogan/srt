"""spark_morning_render's finals (D1, 2026-10-30): a converged target's one final render
goes first, outside the cap, and is redone only when its frames changed. Pure; stdlib.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import spark_morning_render as r  # noqa: E402

FULL = {"Ha": 44, "S-II": 39, "O-III": 41}


def test_converged_targets_reads_the_mirror(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"dso": "IC 410", "status": "converged"}, {"dso": "m1", "status": "waiting"},
                             {"dso": "m31", "status": "completed"}, {"status": "converged"}]))
    assert r.converged_targets(str(q)) == ["ic410"]
    assert r.converged_targets(str(tmp_path / "missing.json")) == []


def test_needs_final_only_without_a_current_record():
    rec = {"recipe": "SHO", "channels": {"Ha": {"frames": 44}, "S-II": {"frames": 39}, "O-III": {"frames": 41}}}
    assert r.needs_final(None, "SHO", FULL)
    assert r.needs_final(rec, "HOO", FULL)                               # another recipe
    assert not r.needs_final(rec, "SHO", FULL)
    assert r.needs_final(rec, "SHO", {**FULL, "Ha": 50})                  # new frames since
    assert r.needs_final({"recipe": "SHO", "channels": {}}, "SHO", FULL)


def test_plan_finals_prefers_one_recipe_and_skips_current_records():
    counts = {"ic410": FULL, "m1": {"Ha": 30, "O-III": 30, "S-II": 30}, "thin": {"Ha": 5}}
    records = {"ic410": None,
               "m1": {"recipe": "HOO", "channels": {"Ha": {"frames": 30}, "O-III": {"frames": 30}}}}
    emission = {"m1": {"S-II": 0.06}}                                     # weak S-II: HOO is m1's recipe
    assert r.plan_finals(["ic410", "m1", "thin", "nope"], counts, records, emission) == [("ic410", "SHO")]
    # m1 with new frames is due again, as HOO
    counts["m1"]["Ha"] = 36
    assert r.plan_finals(["m1"], counts, records, emission) == [("m1", "HOO")]


def test_render_command_carries_final():
    cmd = r.render_command("IC 410", "SHO", first=True, exptime=300, galaxy=False, final=True)
    assert cmd[-2:] == ["--force", "--final"] and "--gain" in cmd
    assert "--final" not in r.render_command("ic410", "SHO", first=False, exptime=300, galaxy=False)


def test_final_record_path():
    assert r.final_record_path("IC 410", Path("/d")) == Path("/d/Iris/ic410/final/final.json")


def test_order_plan_is_longest_waiting_first():
    plan = [("deep", ["LRGB"], 200, ""), ("never", ["SHO"], 30, ""), ("old", ["HOO"], 60, ""), ("recent", ["LRGB"], 90, "")]
    times = {"deep": 1000.0, "old": 500.0, "recent": 2000.0}             # "never" has no render
    assert [r[0] for r in r.order_plan(plan, times)] == ["never", "old", "deep", "recent"]
    # two never rendered: the deeper first
    assert [x[0] for x in r.order_plan([("a", [], 10, ""), ("b", [], 50, "")], {})] == ["b", "a"]
    assert r.order_plan([], {}) == []


def test_last_render_time_reads_the_meta_files(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "_root", str(tmp_path))
    d = tmp_path / "local" / "n2n_lrgb_render"
    d.mkdir(parents=True)
    assert r.last_render_time("ic410") is None
    (d / "meta_ic410_SHO.json").write_text("{}")
    (d / "meta_ic410_HOO.json").write_text("{}")
    assert r.last_render_time("IC 410") == max(p.stat().st_mtime for p in d.glob("meta_ic410_*.json"))
