#!/usr/bin/env python3
"""Render the targets that gained frames last night.

    python scripts/spark_morning_render.py --since-hours 16

Runs after the 05:00 rsync. For each DSO whose LIGHT frames changed recently it
picks EVERY recipe the filters support and runs `n2n_lrgb_render.py routine`
once per recipe, which writes a mono JPEG per channel, a denoised set, and both
colour composites into `<image_dir>/Iris/<dso>/`. A three-filter narrowband
target therefore gets HSO, SHO and HOO in one morning; the palettes disagree
about presentation, not data, so all of them get to exist and the eye chooses.

**Stacks are rebuilt (`--force`) only for a target's FIRST render of the run.**
The routine path caches stacks per (dso, filter) — recipes share them — and
after a night of new subs that cache is stale by exactly the frames this job
exists to include, so the first recipe rebuilds it. The remaining recipes reuse
the stacks that first render just wrote: they cost a denoise+compose, not a
20-60 minute stack. If the first render fails, the rest of that target's
recipes are skipped rather than composed from a half-rebuilt cache.

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

# Recipes in preference order. Every rule the filters satisfy is rendered
# (three-filter narrowband -> HSO, SHO and HOO); the order decides which
# recipe stacks first — the rest reuse its channel stacks.
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
# Narrowband recipes: no L channel, so the shared reference comes from Ha, and
# they take the nebula stretch rather than the galaxy one. Kept as a set because
# adding HSO to RECIPE_RULES without adding it here silently gave that recipe
# `--lum L` (a filter it does not have) and the wrong stretch.
NARROWBAND = {"SHO", "HSO", "HOO"}


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


def already_rendered(dso: str, recipe: str, counts: dict) -> bool:
    """True when the last render of this DSO used the same frame counts.

    The time window alone cannot be trusted: it misfires if the job runs late,
    if the clock skews, or if an rsync touches mtimes without new data. This is
    the exact test — the previous render wrote its per-channel frame counts to
    meta_<dso>_<recipe>.json, so if they match there is nothing new to stack.

    Compares FRAMES OFFERED, not frames accepted: the quality gate is free to
    accept a different number from the same input (it depends on the measured
    FWHM distribution), and a gate decision changing is not new data.
    """
    import json
    meta = (Path(_root) / "local" / "n2n_lrgb_render"
            / f"meta_{dso.lower().replace(' ', '')}_{recipe}.json")
    try:
        prev = json.loads(meta.read_text())["channels"]
    except (OSError, ValueError, KeyError):
        return False
    was = {k: int(v.get("frames", -1)) for k, v in prev.items()}
    now = {k: int(v) for k, v in counts.items() if k in was}
    return bool(was) and was == now


def pick_recipes(counts: dict) -> list[str]:
    """Every recipe the qualifying filters support, in stack-first order."""
    have = {f for f, n in counts.items() if n >= MIN_FRAMES}
    out = []
    for name, need in RECIPE_RULES:
        if need <= have:
            out.append(name)
        elif name == "LRGB" and len(need & have) >= 3:
            out.append(name)
    return out


def main() -> int:
    import socket

    from configs import config

    ap = argparse.ArgumentParser()
    # 16 h, not 30. A window longer than a day necessarily spans two mornings:
    # a frame taken at 03:14 is 3.8 h old at that morning's 07:00 run and 27.8 h
    # old at the next one, so anything imaged after ~01:00 qualified twice and
    # the target re-rendered on identical data. Observed 2026-08-27 on trunk,
    # which burned 23 min of stacking and pushed a notification implying new
    # data. 16 h reaches back to 15:00 the previous afternoon, covering any full
    # night, while last night's small hours fall outside.
    ap.add_argument("--since-hours", type=float, default=16.0)
    ap.add_argument("--exptime", type=int, default=300)
    ap.add_argument("--rerender", action="store_true",
                    help="render even if the frame counts are unchanged")
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
        recipes = pick_recipes(counts)
        summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        if not recipes:
            log(f"{dso}: {summary} — no recipe reaches {MIN_FRAMES} frames, skipped")
            continue
        if not args.rerender:
            todo = [r for r in recipes if not already_rendered(dso, r, counts)]
            if not todo:
                log(f"{dso}: {summary} — unchanged since the last render, skipped "
                    f"(--rerender to force)")
                continue
            recipes = todo
        plan.append((dso, recipes, sum(counts.values()), summary))

    # Deepest first: if the cap bites, the best data gets rendered.
    plan.sort(key=lambda r: -r[2])
    for dso, recipes, total, summary in plan:
        log(f"{dso}: {summary} -> {' + '.join(recipes)} ({total} frames)")
    if len(plan) > args.max_targets:
        log(f"capping at {args.max_targets} targets; the rest re-qualify on the next run")
        plan = plan[:args.max_targets]

    rc = 0
    for dso, recipes, _total, _s in plan:
        for i, recipe in enumerate(recipes):
            lum = "Ha" if recipe in NARROWBAND else "L"
            cmd = [sys.executable, os.path.join(_root, "scripts", "n2n_lrgb_render.py"),
                   "routine", "--dso", dso.lower().replace(" ", ""), "--recipe", recipe,
                   "--lum", lum, "--exptime", str(args.exptime)]
            # Rebuild the shared channel stacks once per target, on its first
            # recipe; the rest reuse them (see module docstring).
            if i == 0:
                cmd.append("--force")
            # No narrowband stretch override. These used to be forced to
            # --white-pct 99 --black-pct 65, and on the 2026-08-24 trunk render that
            # put white at 4.71 ADU (clipping everything above ~3 sigma into flat
            # red slabs) and black at 0.572 ADU — only 0.36 sigma above sky, i.e.
            # inside the noise. Raw sky pixels dithered above that line and read as
            # a veil while denoised sky correctly fell below it, so the denoiser
            # looked as though it had eaten the nebulosity. It had not: tile
            # photometry measured 104-117% flux retention at every brightness. The
            # renderer's own defaults (white p99.95, black at sky - 0.5 sigma) are
            # the ones that render this field correctly.
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
                log(f"  *** {dso} {recipe} failed (rc={p.returncode}) ***")
                for line in (p.stderr or "").splitlines()[-15:]:
                    log(f"  ! {line}")
                if i == 0 and len(recipes) > 1:
                    # The stack rebuild did not finish; composing the remaining
                    # recipes from a half-fresh cache would silently mix nights.
                    log(f"  skipping {' + '.join(recipes[1:])} — stack rebuild incomplete")
                    break
            log(f"  {dso} {recipe} finished in {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
