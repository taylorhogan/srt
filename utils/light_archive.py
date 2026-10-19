"""Which raw light frames iris-pc may delete: only ones the Spark provably holds.

    plan(...)   -> per-target report + the deletable files            pure
    survey(cfg) -> the same from disk, the manifest and the queue       live

Raw lights are the only irreplaceable data, so the rule (owner, 2026-10-11)
is that none is deleted without a VERIFIED copy elsewhere. Built 2026-10-19:
the Spark, after its 06:00 pull of the Targets tree, hashes every LIGHT frame
it holds (scripts/spark_light_manifest.py) and pushes the manifest to
local/spark_light_manifest.json here. A frame here is deletable only when

  * the manifest lists the same relative path with the SAME SHA-256 as the
    file on this disk (size alone is not proof),
  * its target has had no new light frame for disk.light_keep_days (90), and
  * the manifest is fresh (disk.light_manifest_max_age_days, 3) -- an old one
    may predate files the Spark has since lost.

Anything unverified is reported and left alone. The sync has no --delete, so
removing a frame here never removes the Spark's copy. Deletion itself is the
`purge lights [dso] [go]` command, dry run unless `go`.
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LIGHT_KEEP_DAYS = 90.0
MANIFEST_MAX_AGE_DAYS = 3.0
MANIFEST_NAME = "spark_light_manifest.json"
HASH_CACHE_NAME = "light_hash_cache.json"
PURGE_LOG_NAME = "light_purge_log.jsonl"
_CHUNK = 8 * 1024 * 1024


# --------------------------------------------------------------------------- pure

def manifest_age_days(manifest: dict, now: float) -> Optional[float]:
    try:
        t = datetime.fromisoformat(manifest["generated"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return None
    return (now - t) / 86400.0


def plan(local: list, manifest: Optional[dict], now: float,
         keep_days: float = LIGHT_KEEP_DAYS,
         manifest_max_age_days: float = MANIFEST_MAX_AGE_DAYS,
         queue_status: Optional[dict] = None, only: Optional[str] = None) -> dict:
    """What may go, and why everything else stays. Pure.

    *local*: [{"rel", "target", "size", "mtime", "sha256" (None = not hashed)}]
    for every light frame on this disk. *manifest*: the Spark's, or None.
    -> {"refused": reason or None, "targets": {target: {...}}, "deletable": [rec]}
    """
    queue_status = queue_status or {}
    out = {"refused": None, "targets": {}, "deletable": []}
    age = manifest_age_days(manifest, now) if manifest else None
    if manifest is None:
        out["refused"] = "no manifest from the Spark yet (%s)" % MANIFEST_NAME
    elif age is None:
        out["refused"] = "the Spark's manifest has no readable date"
    elif age > manifest_max_age_days:
        out["refused"] = ("the Spark's manifest is %.1f days old (limit %.0f): it may predate "
                          "files the Spark has since lost" % (age, manifest_max_age_days))
    spark = (manifest or {}).get("files") or {}

    by_target: dict = {}
    for r in local:
        by_target.setdefault(r["target"], []).append(r)
    for target, recs in sorted(by_target.items()):
        if only and target.lower() != only.lower():
            continue
        last = max(r["mtime"] for r in recs)
        idle = (now - last) / 86400.0
        t = {"frames": len(recs), "bytes": sum(r["size"] for r in recs),
             "last_frame": datetime.fromtimestamp(last).strftime("%Y-%m-%d"),
             "idle_days": idle, "queue": queue_status.get(target.lower()),
             "verified": 0, "verified_bytes": 0, "missing": 0, "mismatch": 0,
             "eligible": idle >= keep_days}
        out["targets"][target] = t
        if not t["eligible"]:
            continue
        if out["refused"]:
            t["unchecked"] = len(recs)      # no usable manifest: nothing was compared
            continue
        for r in recs:
            s = spark.get(r["rel"])
            if s is None:
                t["missing"] += 1
            elif r.get("sha256") is None or s.get("sha256") != r["sha256"] \
                    or int(s.get("size", -1)) != int(r["size"]):
                t["mismatch"] += 1
            else:
                t["verified"] += 1
                t["verified_bytes"] += r["size"]
                if out["refused"] is None:
                    out["deletable"].append(r)
    return out


def describe(p: dict, keep_days: float, go: bool) -> str:
    lines = []
    if p["refused"]:
        lines.append("Light purge refused: " + p["refused"])
    elig = {k: v for k, v in p["targets"].items() if v["eligible"]}
    lines.append("Light frames idle %.0f+ days, checked against the Spark's copy:" % keep_days)
    if not elig:
        lines.append("  none -- every target has a frame newer than that")
    for name, t in sorted(elig.items(), key=lambda kv: -kv[1]["bytes"]):
        q = (" [queue: %s]" % t["queue"]) if t["queue"] else ""
        bad = []
        if t.get("unchecked"):
            bad.append("%d not checked" % t["unchecked"])
        if t["missing"]:
            bad.append("%d not on the Spark" % t["missing"])
        if t["mismatch"]:
            bad.append("%d differ" % t["mismatch"])
        lines.append("  %s: %d frames, %.1f GB, last %s (%d d)%s -- %d verified%s"
                     % (name, t["frames"], t["bytes"] / 1e9, t["last_frame"], t["idle_days"], q,
                        t["verified"], ("; " + ", ".join(bad) + ", kept") if bad else ""))
    if any(t["queue"] == "waiting" for t in elig.values()):
        lines.append("  Targets still waiting in the queue keep imaging; after a purge, iris-pc's "
                     "`process` sees only new frames, while the Spark holds them all.")
    n = len(p["deletable"])
    gb = sum(r["size"] for r in p["deletable"]) / 1e9
    if n:
        lines.append("%s %d verified frames, %.1f GB." % ("Deleted" if go else "Would delete", n, gb))
        if not go:
            lines.append("Re-run with `go` to delete them. Only frames whose SHA-256 matches the "
                         "Spark's copy are touched; the Spark keeps its copy.")
    elif not p["refused"]:
        lines.append("Nothing verified to delete.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- live

def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _local_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "local"


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def local_frames(image_dir: Path) -> list:
    """Every *.fits in a LIGHT directory under *image_dir*, unhashed."""
    out = []
    for dirpath, _dirs, files in os.walk(image_dir):
        if os.path.basename(dirpath).upper() != "LIGHT":
            continue
        for f in files:
            if not f.lower().endswith(".fits"):
                continue
            ap = os.path.join(dirpath, f)
            try:
                st = os.stat(ap)
            except OSError:
                continue
            rel = os.path.relpath(ap, image_dir).replace(os.sep, "/")
            out.append({"rel": rel, "path": ap, "target": rel.split("/")[0],
                        "size": st.st_size, "mtime": st.st_mtime, "sha256": None})
    return out


def survey(cfg: dict, only: Optional[str] = None, now: Optional[float] = None,
           progress=None) -> tuple:
    """-> (plan, keep_days). Hashes the eligible targets' local frames (cached by
    size + mtime in local/light_hash_cache.json) and nothing else."""
    now = time.time() if now is None else now
    disk = cfg.get("disk") or {}
    keep = float(disk.get("light_keep_days", LIGHT_KEEP_DAYS))
    max_age = float(disk.get("light_manifest_max_age_days", MANIFEST_MAX_AGE_DAYS))
    image_dir = Path(cfg["nina"]["image_dir"])
    manifest = _read_json(_local_dir() / MANIFEST_NAME, None)
    queue = {}
    try:
        inst = _read_json(Path(__file__).resolve().parent.parent / cfg["location"]["instructions"], [])
        for e in inst if isinstance(inst, list) else []:
            queue[str(e.get("dso", "")).lower().replace(" ", "")] = e.get("status")
    except Exception:
        queue = {}
    local = local_frames(image_dir)
    # Hash only what could be deleted: frames of targets idle past the limit.
    last = {}
    for r in local:
        last[r["target"]] = max(last.get(r["target"], 0), r["mtime"])
    eligible = {t for t, m in last.items() if (now - m) / 86400.0 >= keep
                and (not only or t.lower() == only.lower())}
    # No usable manifest: plan() refuses regardless, so do not spend minutes
    # hashing frames that cannot be verified anyway.
    m_age = manifest_age_days(manifest, now) if manifest else None
    if m_age is None or m_age > max_age:
        eligible = set()
    cache_path = _local_dir() / HASH_CACHE_NAME
    cache = _read_json(cache_path, {})
    hashed = 0
    for r in local:
        if r["target"] not in eligible:
            continue
        key = "%d:%d" % (r["size"], int(r["mtime"]))
        c = cache.get(r["rel"])
        if c and c.get("key") == key:
            r["sha256"] = c["sha256"]
            continue
        r["sha256"] = sha256_of(r["path"])
        cache[r["rel"]] = {"key": key, "sha256": r["sha256"]}
        hashed += 1
        if progress and hashed % 100 == 0:
            progress("hashed %d frames..." % hashed)
    if hashed:
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, cache_path)
    return plan(local, manifest, now, keep, max_age,
                {k: v for k, v in queue.items()}, only), keep


def delete(records: list, image_dir: Path) -> tuple:
    """Delete verified frames; log each; tidy empty LIGHT and night dirs. -> (n, bytes, failures)."""
    log_path = _local_dir() / PURGE_LOG_NAME
    n, freed, failed = 0, 0, []
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log_path, "a", encoding="utf-8") as log:
        for r in records:
            try:
                os.remove(r["path"])
            except OSError as exc:
                failed.append((r["rel"], str(exc)))
                continue
            n += 1
            freed += r["size"]
            log.write(json.dumps({"time": stamp, "rel": r["rel"], "size": r["size"],
                                  "sha256": r["sha256"]}) + "\n")
    for d in sorted({os.path.dirname(r["path"]) for r in records}, key=len, reverse=True):
        p = Path(d)
        while p != image_dir and p.is_dir() and not any(p.iterdir()):
            p.rmdir()
            p = p.parent
    return n, freed, failed
