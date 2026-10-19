#!/usr/bin/env python3
"""The Spark's proof that it holds each light frame: a SHA-256 per frame.

    python scripts/spark_light_manifest.py \
        --root /home/taylor/Desktop/Targets \
        --out  /home/taylor/Documents/srt/local/spark_light_manifest.json

Runs on the Spark right after sync_nina_targets_to_spark.bsh has pulled the
Targets tree, which then pushes the manifest to iris-pc's local/. There,
utils/light_archive compares it with the observatory's own frames: a light
frame becomes deletable on iris-pc ONLY when the Spark's copy hashes the same
(`purge lights`). The sync has no --delete, so a frame removed on iris-pc
stays on the Spark.

Every *.fits in a LIGHT directory is listed, keyed by its path relative to
--root with forward slashes (the same key iris-pc builds). Hashes are cached
by (size, mtime) in --cache, so a morning hashes only the night's new frames;
the first run hashes everything (~400 GB, some minutes of CPU). Standard
library only: the Spark's cron Python needs nothing installed.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

CHUNK = 8 * 1024 * 1024


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def light_files(root: str):
    """(relpath with '/', abspath) for every *.fits in a LIGHT directory."""
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath).upper() != "LIGHT":
            continue
        for f in files:
            if f.lower().endswith(".fits"):
                ap = os.path.join(dirpath, f)
                yield os.path.relpath(ap, root).replace(os.sep, "/"), ap


def build(root: str, cache: dict, log=print) -> tuple:
    """-> (files {rel: {size, sha256}}, new cache, n_hashed)."""
    files, new_cache, hashed = {}, {}, 0
    for rel, ap in light_files(root):
        try:
            st = os.stat(ap)
        except OSError:
            continue
        key = "%d:%d" % (st.st_size, int(st.st_mtime))
        c = cache.get(rel)
        if c and c.get("key") == key:
            digest = c["sha256"]
        else:
            digest = sha256_of(ap)
            hashed += 1
            if hashed % 200 == 0:
                log("  hashed %d frames..." % hashed)
        new_cache[rel] = {"key": key, "sha256": digest}
        files[rel] = {"size": st.st_size, "sha256": digest}
    return files, new_cache, hashed


def _atomic_write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="the Spark's copy of N.I.N.A/Targets")
    ap.add_argument("--out", required=True, help="manifest JSON to write")
    ap.add_argument("--cache", default="", help="hash cache (default: <out>.cache)")
    args = ap.parse_args()
    cache_path = args.cache or args.out + ".cache"
    try:
        with open(cache_path) as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    if not os.path.isdir(args.root):
        print("no such root: %s" % args.root)
        return 1
    t0 = time.time()
    files, new_cache, hashed = build(args.root, cache)
    manifest = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": socket.gethostname(), "root": args.root,
                "count": len(files), "bytes": sum(f["size"] for f in files.values()),
                "files": files}
    _atomic_write(args.out, manifest)
    _atomic_write(cache_path, new_cache)
    print("manifest: %d light frames, %.1f GB, %d hashed this run, %.0f s -> %s"
          % (len(files), manifest["bytes"] / 1e9, hashed, time.time() - t0, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
