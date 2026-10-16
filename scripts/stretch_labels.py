#!/usr/bin/env python3
"""Stretch labels: the stretches the owner actually chose, measured portably.

    python scripts/stretch_labels.py            # (re)build local/stretch_labels.json
    python scripts/stretch_labels.py --only ngc6946

Phase 0 of docs/AUTO_STRETCH_PLAN.md. An automatic stretch is scored against
what the owner picked, so the picks have to exist as data. Two sources:

  BACKFILL  picks made before the log could say so, identified by hand from
            each target's process_log.json (2026-10-16). Joined by time only.
  the logs  every render or publish since then that carries "kept" (the
            `keep` command), "from_sweep" (picked off a sweep sheet) or is a
            publish (always a choice off a sheet).

Each pick is re-rendered from the cached channels exactly as it was made and
measured with color_process.measure_stretch, which records what the stretch
does to the field (black in sigma below sky, output level of the faint signal,
what clips) instead of its percentiles, which mean nothing across targets.
Later picks of the same (dso, tag) supersede earlier ones.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

OUT = Path(_root) / "local" / "stretch_labels.json"

# Picks found in the logs on 2026-10-16. "who": owner = the owner chose it;
# claude = chosen by Claude and shown to the owner, not yet confirmed (weight
# it lower until `keep` confirms it).
BACKFILL = [
    {"dso": "bubble", "tag": "HOO", "time": "2026-08-16T14:38:01-04:00", "who": "owner",
     "how": "publish 19 of a 27-variant sweep",
     "opts": {"black_pct": 65.0, "white_pct": 99.0, "softening": 0.025}},
    {"dso": "ic1396", "tag": "HOO", "time": "2026-08-19T08:27:46-04:00", "who": "owner",
     "how": "publish 1, 13 min after publishing 19",
     "opts": {"black_pct": 45.0, "white_pct": 99.0, "softening": 0.025}},
    {"dso": "sh2-92", "tag": "HOO", "time": "2026-08-03T14:06:32-04:00", "who": "owner",
     "how": "reuse render after sweeps",
     "opts": {"black_pct": 55.0, "white_pct": 99.5, "softening": 0.05}},
    {"dso": "m33", "tag": "HALRGB", "time": "2026-09-24T09:20:31-04:00", "who": "owner",
     "how": "reuse render after sweeps",
     "opts": {"black_pct": 55.0, "white_pct": 99.9, "softening": 0.025, "white_balance": "stars"}},
    {"dso": "ngc7320", "tag": "LRGB", "time": "2026-10-04T03:08:36-04:00", "who": "owner",
     "how": "reuse render after sweeps",
     "opts": {"black_pct": 60.0, "white_pct": 99.9, "softening": 0.025, "white_balance": "stars"}},
    {"dso": "ngc6946", "tag": "HALRGB", "time": "2026-10-16T11:30:00-04:00", "who": "claude",
     "how": "panel 5 of an 8-variant sweep, sent to the owner",
     "opts": {"black_pct": 60.0, "white_pct": 99.9, "softening": 0.05, "white_balance": "stars"}},
    # The 2026-09-12 by-hand NGC 7380 deep render (n2n pipeline): per-channel
    # black margins below sky in sigma and white at p99.95, re-applied here to
    # the process cache, so only approximately the same pixels.
    {"dso": "ngc7380", "tag": "HOO", "time": "2026-09-12T12:00:00-04:00", "who": "owner",
     "how": "by-hand n2n render (auto_stretch_compare HAND), re-applied",
     "opts": {"white_pct": 99.95, "softening": 0.006},
     "margins": {"R": 0.5, "G": 0.0, "B": 0.0}},
]

_STRETCH_KEYS = ("black_pct", "white_pct", "softening")


def log(m=""):
    print(m, flush=True)


def from_logs(image_root: Path) -> list[dict]:
    """Picks the logs can vouch for: kept, from_sweep, or publish."""
    out = []
    for p in sorted(image_root.glob("*/process_log.json")):
        try:
            entries = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict) or e.get("kind") not in ("render", "publish"):
                continue
            if not (e.get("kept") or e.get("from_sweep") or e.get("kind") == "publish"):
                continue
            opts = dict(e.get("options") or e.get("settings") or {})
            if (e.get("scale") or 1) != 1 and e.get("kind") == "render" and not e.get("kept"):
                continue                 # a quick look, not a pick
            how = ("kept" if e.get("kept") else
                   "publish %s" % e.get("panel") if e.get("kind") == "publish" else
                   "panel %s of a sweep" % e.get("panel"))
            out.append({"dso": p.parent.name, "tag": e.get("tag") or e.get("recipe"),
                        "time": e.get("time"), "who": "owner", "how": how, "opts": opts,
                        "kept": bool(e.get("kept"))})
    return out


def measure(label: dict, results_dir) -> dict:
    import numpy as np
    from stacking import color_process as cp

    out_dir = results_dir(label["dso"])
    ch = cp.load_cached_channels(out_dir, label["dso"], label["tag"])
    if ch is None:
        raise FileNotFoundError("no cached channels for %s %s" % (label["dso"], label["tag"]))
    o = cp.effective_options(**label["opts"])
    subbed, white = cp._prepare(ch, o["subtract_background"], o["mesh"], o["white_pct"],
                                o["ha_gain"], o["white_balance"])
    if "margins" in label:
        blacks = {}
        for c, a in subbed.items():
            if c in cp.DERIVED_PLANES:
                continue
            d = cp.sky_noise_model(a)
            blacks[c] = d["sky"] - label["margins"].get(c, 0.0) * d["sigma"]
    else:
        blacks = cp.blacks_from(subbed, o["black_pct"])
    m = cp.measure_stretch(subbed, blacks, white, o["softening"],
                           lum="L" if "L" in subbed else None)
    m["white_adu"] = white
    m["blacks_adu"] = {c: float(v) for c, v in blacks.items()}
    m["shape"] = list(next(iter(ch.values())).shape)
    del ch, subbed
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="measure just this dso")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from configs import config
    from stacking import stacker
    image_root = Path(config.data()["nina"]["image_dir"]) / "Iris"

    picks = BACKFILL + from_logs(image_root)
    # Latest pick per (dso, tag) wins; a kept one beats an unkept later one.
    best: dict[tuple, dict] = {}
    for p in sorted(picks, key=lambda p: (bool(p.get("kept")), str(p["time"]))):
        best[(p["dso"], str(p["tag"]).upper())] = p
    labels = [p for k, p in sorted(best.items()) if not args.only or k[0] == args.only]

    previous = {}
    if args.only and args.out.exists():
        previous = {(x["dso"], x["tag"]): x for x in json.loads(args.out.read_text(encoding="utf-8"))}
    for lab in labels:
        t0 = time.perf_counter()
        try:
            lab["measured"] = measure(lab, stacker.results_dir)
        except Exception as exc:  # noqa: BLE001
            lab["error"] = "%s: %s" % (type(exc).__name__, exc)
            log("%-10s %-7s  FAILED  %s" % (lab["dso"], lab["tag"], lab["error"]))
            previous[(lab["dso"], lab["tag"])] = lab
            continue
        m = lab["measured"]
        margins = " ".join("%s %+.2f" % (c, v) for c, v in sorted(m["margin_sigma"].items()))
        log("%-10s %-7s %-6s  margin(σ) %s  ext p10→%s p50→%s  bright→%s  ext clip %s  star clip %s  (%.0fs)" % (
            lab["dso"], lab["tag"], lab["who"], margins,
            "%.2f" % m["ext_p10_out"] if m["ext_p10_out"] is not None else "-",
            "%.2f" % m["ext_p50_out"] if m["ext_p50_out"] is not None else "-",
            "%.2f" % m["bright_out"] if m["bright_out"] is not None else "-",
            "%.2f%%" % (100 * m["ext_clip"]) if m["ext_clip"] is not None else "-",
            "%.0f%%" % (100 * m["star_clip"]) if m["star_clip"] is not None else "-",
            time.perf_counter() - t0))
        previous[(lab["dso"], lab["tag"])] = lab

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(list(previous.values()), indent=1, default=str), encoding="utf-8")
    os.replace(tmp, args.out)
    log("wrote %d labels to %s" % (len(previous), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
