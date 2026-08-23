#!/usr/bin/env python3
"""Render the targets that gained frames last night.

    python scripts/spark_morning_render.py --since-hours 30

Runs after the 05:00 rsync. For each DSO whose LIGHT frames changed recently it
picks a recipe from the filters present and runs
`n2n_lrgb_render.py routine`, which writes a mono JPEG per channel, a denoised
set, and both colour composites into `<image_dir>/Iris/<dso>/`.

**Stacks are always rebuilt (`--force`).** The routine path caches stacks by
filename, and after a night of new subs that cache is stale by exactly the
frames this job exists to include. Reusing it would silently render yesterday's
image and report success.

**Both composites are always written and neither is chosen.** Whether the
denoised version is better depends on surface brightness, not object class: NGC
6888's bright filaments survive denoising and its denoised SHO is the better
image, while ic1396's 1-5 sigma emission is destroyed by the same model (lab
manual steps 23, 29). Until that threshold is calibrated against more targets,
the choice needs an eye and this job does not make it.

Bounded on purpose. Stacking is 20-60 min per target and peak RSS is ~64 GB, so
`--max-targets` caps a run and targets are processed one at a time. A morning
with four new targets renders the deepest and defers the rest to the next run
rather than running the Spark out of memory at 6am.
"""

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Recipes in preference order: the richest palette the filters support.
RECIPE_RULES = [
    # HSO before SHO for a three-filter narrowband target. Both are honest —
    # compose holds one brightness scale across channels either way — but on
    # NGC 6888 the nebula measures Ha 4.20 ADU, S-II 2.62, O-III 0.36, so SHO
    # puts the two strong channels on adjacent primaries (red/green, Ha
    # dominant) and washes to yellow-green, while HSO puts them on opposing
    # primaries and their ratio reads directly as hue: green where S-II
    # dominates the rim, orange where Ha and S-II overlap.
    #
    # Caveat worth knowing when reading the output: green IS the S-II channel,
    # and S-II is the channel the denoiser under-retains most (0.572 against
    # Ha's 0.954, step 31). HSO therefore shows that defect plainly rather than
    # burying it in a mixture — better to look at, less forgiving.
    ("HSO", {"Ha", "S-II", "O-III"}),
    ("SHO", {"S-II", "Ha", "O-III"}),
    ("HOO", {"Ha", "O-III"}),
    ("LRGB", {"L", "R", "G", "B"}),
]
MIN_FRAMES = 12          # below this a stack is not worth the wall time


def log(m: str = "") -> None:
    print(m, flush=True)


def recent_targets(subs_dir: Path, since_hours: float, exptime: int):
    """(dso, {filter: n}) for targets whose LIGHT frames changed recently.

    Recency is by file mtime, which rsync sets from the source, so a target that
    was synced but not re-imaged does not re-render.
    """
    from astropy.io import fits

    from stacking.color_process import canonical_filter

    cutoff = time.time() - since_hours * 3600
    fresh = defaultdict(lambda: defaultdict(int))
    touched = set()
    for fp in subs_dir.rglob("*.fits"):
        if fp.parent.name.upper() != "LIGHT":
            continue
        try:
            if fp.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        try:
            dso = fp.relative_to(subs_dir).parts[0]
        except ValueError:
            continue
        touched.add(dso)
    # Only the touched targets get their headers read — a full header scan of
    # the archive costs minutes and nothing here needs it.
    for dso in sorted(touched):
        for fp in (subs_dir / dso).rglob("*.fits"):
            if fp.parent.name.upper() != "LIGHT":
                continue
            try:
                h = fits.getheader(fp)
                if round(float(h.get("EXPTIME", 0))) != exptime:
                    continue
                c = canonical_filter(str(h.get("FILTER", "")).strip())
            except Exception:
                continue
            if c:
                fresh[dso][c] += 1
    return fresh


def pick_recipe(counts: dict):
    have = {f for f, n in counts.items() if n >= MIN_FRAMES}
    for name, need in RECIPE_RULES:
        if need <= have:
            return name
        if name == "LRGB" and len(need & have) >= 3:
            return name
    return None


def main() -> int:
    import socket

    from configs import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=float, default=30.0)
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    if "subs_dir" not in machine:
        log(f"machine {socket.gethostname()} has no subs_dir — this runs on the Spark only")
        return 1
    subs = Path(machine["subs_dir"])

    fresh = recent_targets(subs, args.since_hours, args.exptime)
    if not fresh:
        log(f"no LIGHT frames newer than {args.since_hours:g} h — nothing to render")
        return 0

    plan = []
    for dso, counts in fresh.items():
        recipe = pick_recipe(counts)
        summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        if not recipe:
            log(f"{dso}: {summary} — no recipe reaches {MIN_FRAMES} frames, skipped")
            continue
        plan.append((dso, recipe, sum(counts.values()), summary))

    # Deepest first: if the cap bites, the best data gets rendered.
    plan.sort(key=lambda r: -r[2])
    for dso, recipe, total, summary in plan:
        log(f"{dso}: {summary} -> {recipe} ({total} frames)")
    if len(plan) > args.max_targets:
        log(f"capping at {args.max_targets}; the rest re-qualify on the next run")
        plan = plan[:args.max_targets]

    rc = 0
    for dso, recipe, _total, _s in plan:
        lum = "Ha" if recipe in ("SHO", "HOO") else "L"
        cmd = [sys.executable, os.path.join(_root, "scripts", "n2n_lrgb_render.py"),
               "routine", "--dso", dso.lower().replace(" ", ""), "--recipe", recipe,
               "--lum", lum, "--exptime", str(args.exptime), "--force"]
        if recipe in ("SHO", "HOO"):
            cmd += ["--white-pct", "99", "--black-pct", "65"]
        log(f"\n$ {' '.join(cmd[1:])}")
        if args.dry_run:
            continue
        t0 = time.time()
        # Per-target isolation: one target failing to stack must not stop the
        # others, and the traceback belongs in this job's log.
        p = subprocess.run(cmd, cwd=_root, capture_output=True, text=True)
        for line in (p.stdout or "").splitlines():
            if line.strip():
                log(f"  {line}")
        if p.returncode != 0:
            rc = 1
            log(f"  *** {dso} failed (rc={p.returncode}) ***")
            for line in (p.stderr or "").splitlines()[-15:]:
                log(f"  ! {line}")
        log(f"  {dso} finished in {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
