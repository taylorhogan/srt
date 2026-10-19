"""Raw light frames leave iris-pc only when the Spark provably holds them.

utils/light_archive.plan is pure: local frame records + the Spark's manifest.
The rule (owner, 2026-10-11): no light frame is deleted without a verified
copy elsewhere -- here, the same relative path with the same SHA-256 in a
fresh manifest -- and only for targets idle past the keep window.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import light_archive as la  # noqa: E402

DAY = 86400.0
NOW = datetime(2026, 10, 19, 12, tzinfo=timezone.utc).timestamp()


def rec(target, name, age_days, sha="aa", size=100):
    return {"rel": "%s/cdk17/2026-06-01/LIGHT/%s" % (target, name), "path": "/x/" + name,
            "target": target, "size": size, "mtime": NOW - age_days * DAY, "sha256": sha}


def manifest(files, age_days=0.5):
    gen = datetime.fromtimestamp(NOW - age_days * DAY, tz=timezone.utc).isoformat()
    return {"generated": gen, "files": {r["rel"]: {"size": r["size"], "sha256": r["sha256"]} for r in files}}


def test_only_hash_matched_frames_of_idle_targets_are_deletable():
    old = [rec("ngc5907", "a.fits", 120), rec("ngc5907", "b.fits", 125)]
    fresh = [rec("ngc6946", "c.fits", 1)]
    p = la.plan(old + fresh, manifest(old + fresh), NOW)
    assert p["refused"] is None
    assert [r["rel"] for r in p["deletable"]] == [r["rel"] for r in old]
    assert p["targets"]["ngc6946"]["eligible"] is False


def test_a_frame_the_spark_lacks_or_holds_differently_stays():
    a, b, c = rec("m92", "a.fits", 120), rec("m92", "b.fits", 120), rec("m92", "c.fits", 120)
    m = manifest([a, b])
    m["files"][b["rel"]]["sha256"] = "different"
    p = la.plan([a, b, c], m, NOW)
    assert [r["rel"] for r in p["deletable"]] == [a["rel"]]
    t = p["targets"]["m92"]
    assert (t["verified"], t["mismatch"], t["missing"]) == (1, 1, 1)


def test_an_unhashed_local_frame_is_never_verified():
    a = rec("m13", "a.fits", 200)
    m = manifest([a])
    a = dict(a, sha256=None)
    assert la.plan([a], m, NOW)["deletable"] == []


def test_a_target_with_one_recent_frame_is_not_idle():
    recs = [rec("bubble", "old.fits", 300), rec("bubble", "new.fits", 10)]
    p = la.plan(recs, manifest(recs), NOW)
    assert p["deletable"] == [] and p["targets"]["bubble"]["eligible"] is False


def test_no_manifest_or_a_stale_one_refuses_everything():
    a = rec("m13", "a.fits", 200)
    assert la.plan([a], None, NOW)["refused"]
    stale = la.plan([a], manifest([a], age_days=5), NOW)
    assert stale["refused"] and "days old" in stale["refused"] and stale["deletable"] == []


def test_naming_a_target_limits_the_plan():
    recs = [rec("m13", "a.fits", 200), rec("m92", "b.fits", 200)]
    p = la.plan(recs, manifest(recs), NOW, only="M92")
    assert list(p["targets"]) == ["m92"] and len(p["deletable"]) == 1


def test_the_report_says_what_stays_and_why():
    a, b = rec("m92", "a.fits", 120), rec("m92", "b.fits", 120)
    p = la.plan([a, b], manifest([a]), NOW, queue_status={"m92": "waiting"})
    text = la.describe(p, 90, go=False)
    assert "1 not on the Spark" in text and "Would delete 1" in text and "queue: waiting" in text


def test_spark_manifest_keys_match_the_local_keys(tmp_path):
    """Both sides key a frame by its path under Targets with forward slashes."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import spark_light_manifest as sm
    light = tmp_path / "ngc5907" / "cdk17" / "2026-06-01" / "LIGHT"
    light.mkdir(parents=True)
    (light / "f1.fits").write_bytes(b"x" * 10)
    (light / "f2.fits.bad").write_bytes(b"y")
    files, cache, hashed = sm.build(str(tmp_path), {}, log=lambda m: None)
    local = la.local_frames(tmp_path)
    assert set(files) == {r["rel"] for r in local} == {"ngc5907/cdk17/2026-06-01/LIGHT/f1.fits"}
    assert files["ngc5907/cdk17/2026-06-01/LIGHT/f1.fits"]["sha256"] == la.sha256_of(local[0]["path"])
    again, _c, hashed2 = sm.build(str(tmp_path), cache, log=lambda m: None)
    assert hashed == 1 and hashed2 == 0 and again == files


def test_delete_removes_logs_and_tidies(tmp_path, monkeypatch):
    light = tmp_path / "m13" / "cdk17" / "2026-06-19" / "LIGHT"
    light.mkdir(parents=True)
    f = light / "a.fits"
    f.write_bytes(b"z" * 5)
    monkeypatch.setattr(la, "_local_dir", lambda: tmp_path / "local")
    (tmp_path / "local").mkdir()
    r = {"rel": "m13/cdk17/2026-06-19/LIGHT/a.fits", "path": str(f), "size": 5, "sha256": "s"}
    n, freed, failed = la.delete([r], tmp_path)
    assert (n, freed, failed) == (1, 5, [])
    assert not (tmp_path / "m13").exists()                    # emptied dirs tidied
    assert "a.fits" in (tmp_path / "local" / la.PURGE_LOG_NAME).read_text()
