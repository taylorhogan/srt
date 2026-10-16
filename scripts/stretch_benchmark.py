#!/usr/bin/env python3
"""Score automatic stretches against the owner's picks, leave-one-out.

    python scripts/stretch_benchmark.py                 # all labels, bin 2
    python scripts/stretch_benchmark.py --bin 1         # full resolution (slow)

Phase 1 of docs/AUTO_STRETCH_PLAN.md. For every label in
local/stretch_labels.json (scripts/stretch_labels.py) the pick is re-made from
the cached channels and measured with color_process.measure_stretch, then three
candidates are made on the same field and measured the same way:

  v2       auto_stretch_v2 with targets taken ONLY from the OTHER owner labels
           of the same class (broadband / narrowband). A target may never
           tune itself; with no other label in its class, the code defaults.
  auto     the 2026-09-12 auto_stretch (white at p99.95, as n2n uses it).
  default  the process command's fixed p65 / p99 / 0.025.

Each candidate is compared with the pick in output units: black margin (mean
over R, G, B, in sigma), object clip fraction, and the output level of the
bright p90. PASS = |margin| <= 0.3 sigma, |bright| <= 0.12, and object clip
within 3x (or 0.2 percentage points) of the pick. Writes
local/stretch_benchmark/{results.json, sheet.jpg}: one row per label, the pick
beside v2 and the default, so the numbers can be checked by eye.

Also prints each class's targets measured over ALL owner labels: the values
for color_process.STRETCH_TARGETS.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from stacking import color_process as cp  # noqa: E402

LABELS = Path(_root) / "local" / "stretch_labels.json"
OUT = Path(_root) / "local" / "stretch_benchmark"
TOL_MARGIN = 0.3
TOL_BRIGHT = 0.12
CLIP_FACTOR = 3.0
CLIP_FLOOR = 0.002


def log(m=""):
    print(m, flush=True)


def rgb_margin(m: dict) -> float:
    return float(np.mean([m["margin_sigma"][c] for c in ("R", "G", "B") if c in m["margin_sigma"]]))


def summary(m: dict) -> dict:
    return {"margin": rgb_margin(m), "ext_clip": m["ext_clip"], "bright_out": m["bright_out"]}


def targets_from(labels: list[dict]) -> dict:
    """Median targets over measured labels (their summary())."""
    s = [x["summary"] for x in labels]
    out = {"margin": float(np.median([v["margin"] for v in s]))}
    clips = [v["ext_clip"] for v in s if v["ext_clip"] is not None]
    brights = [v["bright_out"] for v in s if v["bright_out"] is not None]
    if clips:
        out["ext_clip"] = float(np.median(clips))
    if brights:
        out["bright_out"] = float(np.median(brights))
    return out


def scores(cand: dict, pick: dict) -> dict:
    dm = cand["margin"] - pick["margin"]
    db = (None if cand["bright_out"] is None or pick["bright_out"] is None
          else cand["bright_out"] - pick["bright_out"])
    pc, cc = pick["ext_clip"], cand["ext_clip"]
    clip_ok = (pc is None or cc is None or abs(cc - pc) <= CLIP_FLOOR
               or (pc > 0 and 1 / CLIP_FACTOR <= cc / pc <= CLIP_FACTOR))
    ok = abs(dm) <= TOL_MARGIN and (db is None or abs(db) <= TOL_BRIGHT) and clip_ok
    return {"d_margin": dm, "d_bright": db, "clip_ok": clip_ok, "pass": ok}


def prepared(label: dict, results_dir, bin_factor: int):
    """(subbed, label blacks, label white, label softening, lum) for one label."""
    ch = cp.load_cached_channels(results_dir(label["dso"]), label["dso"], label["tag"])
    if ch is None:
        raise FileNotFoundError("no cached channels for %s %s" % (label["dso"], label["tag"]))
    if bin_factor > 1:
        ch = {c: cp._block_mean(a, bin_factor).astype(np.float32) for c, a in ch.items()}
    o = cp.effective_options(**label["opts"])
    subbed, white = cp._prepare(ch, o["subtract_background"], o["mesh"], o["white_pct"],
                                o["ha_gain"], o["white_balance"])
    lum = "L" if "L" in subbed else None
    if "margins" in label:
        fm = cp.field_model(subbed, lum)
        blacks = {c: d["sky"] - label["margins"].get(c, 0.0) * d["sigma"]
                  for c, d in fm["diag"].items()}
    else:
        blacks = cp.blacks_from(subbed, o["black_pct"])
    return subbed, blacks, white, o["softening"], lum


def panel(subbed, blacks, white, soft, width=520):
    from PIL import Image
    from stacking import stacker
    rgb = cp.compose_prepared(subbed, blacks, white, softening=soft)
    arr = stacker.sky_parity((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im = Image.fromarray(arr)
    return im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", type=int, default=2, help="block-average the channels first")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--no-sheet", action="store_true")
    args = ap.parse_args()

    from stacking import stacker
    labels = [x for x in json.loads(args.labels.read_text(encoding="utf-8")) if "error" not in x]
    for lab in labels:
        lab["cls"] = cp.stretch_class(lab["tag"])

    # Pass 1: measure every pick on the benchmark's own grid.
    log("pass 1: measuring %d picks at bin %d" % (len(labels), args.bin))
    for lab in labels:
        t0 = time.perf_counter()
        subbed, blacks, white, soft, lum = prepared(lab, stacker.results_dir, args.bin)
        lab["pick"] = cp.measure_stretch(subbed, blacks, white, soft, lum=lum)
        lab["summary"] = summary(lab["pick"])
        log("  %-9s %-7s %-10s margin %+.2f  clip %s  bright %s  (%.0fs)" % (
            lab["dso"], lab["tag"], lab["cls"], lab["summary"]["margin"],
            "%.2f%%" % (100 * lab["summary"]["ext_clip"]) if lab["summary"]["ext_clip"] is not None else "-",
            "%.2f" % lab["summary"]["bright_out"] if lab["summary"]["bright_out"] is not None else "-",
            time.perf_counter() - t0))
        del subbed

    # Pass 2: candidates, leave-one-out.
    log("\npass 2: candidates (v2 targets from the OTHER owner labels of the class)")
    rows, results = [], []
    for lab in labels:
        others = [x for x in labels if x is not lab and x["who"] == "owner" and x["cls"] == lab["cls"]]
        loo = targets_from(others) if others else {}
        subbed, blacks, white, soft, lum = prepared(lab, stacker.results_dir, args.bin)
        fm = cp.field_model(subbed, lum)
        v2 = cp.auto_stretch_v2(subbed, lab["cls"], loo, lum=lum, fm=fm)
        old_white = cp.shared_white(subbed, 99.95)
        old = cp.auto_stretch(subbed, old_white, lum=lum)
        cands = {
            "v2": (v2["blacks"], v2["white"], v2["softening"]),
            "auto": (old["blacks"], old_white, old["softening"]),
            "default": (cp.blacks_from(subbed, cp.BLACK_PCT), cp.shared_white(subbed, cp.WHITE_PCT),
                        cp.SOFTENING),
        }
        res = {"dso": lab["dso"], "tag": lab["tag"], "cls": lab["cls"], "who": lab["who"],
               "pick": lab["summary"], "loo_targets": loo, "v2_confident": v2["confident"],
               "v2_why": v2["why"], "candidates": {}}
        for name, (b, w, s) in cands.items():
            m = summary(cp.measure_stretch(subbed, b, w, s, lum=lum, fm=fm))
            res["candidates"][name] = {**m, **scores(m, lab["summary"]), "softening": s}
        results.append(res)
        line = "  %-9s %-7s" % (lab["dso"], lab["tag"])
        for name in ("v2", "auto", "default"):
            c = res["candidates"][name]
            line += "  %s %s (dm %+.2f db %s%s)" % (
                name, "PASS" if c["pass"] else "fail", c["d_margin"],
                "%+.2f" % c["d_bright"] if c["d_bright"] is not None else "-",
                "" if c["clip_ok"] else " clip")
        if not v2["confident"]:
            line += "  [v2 not confident: %s]" % "; ".join(v2["why"])
        log(line)
        if not args.no_sheet:
            rows.append((lab, [panel(subbed, blacks, white, soft),
                               panel(subbed, *cands["v2"]), panel(subbed, *cands["default"])]))
        del subbed, fm

    n = len(results)
    for name in ("v2", "auto", "default"):
        k = sum(r["candidates"][name]["pass"] for r in results)
        log("%-8s passes %d/%d" % (name, k, n))
    log("\nclass targets over ALL owner labels (for STRETCH_TARGETS):")
    for cls in ("broadband", "narrowband"):
        own = [x for x in labels if x["who"] == "owner" and x["cls"] == cls]
        if own:
            log("  %-10s %s  (n=%d)" % (cls, json.dumps({k: round(v, 4) for k, v in targets_from(own).items()}),
                                         len(own)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    if rows:
        from PIL import Image, ImageDraw
        w = sum(p.width for p in rows[0][1]) + 20
        head = 28
        h = sum(max(p.height for p in ps) + head for _, ps in rows)
        sheet = Image.new("RGB", (w, h), (0, 0, 0))
        dr = ImageDraw.Draw(sheet)
        y = 0
        for lab, ps in rows:
            x = 0
            for title, p in zip(("pick (%s)" % lab["who"], "v2 (leave-one-out)", "default p65/p99"), ps):
                dr.text((x + 6, y + 6), "%s %s - %s" % (lab["dso"], lab["tag"], title), fill=(255, 210, 0))
                sheet.paste(p, (x, y + head))
                x += p.width + 10
            y += max(p.height for p in ps) + head
        sheet.save(OUT / "sheet.jpg", quality=90)
        log("sheet: %s" % (OUT / "sheet.jpg"))
    log("results: %s" % (OUT / "results.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
