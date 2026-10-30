#!/usr/bin/env python3
"""Render the targets that gained frames last night.

    python scripts/spark_morning_render.py --since-hours 16

Runs after the 05:00 rsync. For each DSO whose LIGHT frames changed recently it
picks the recipes the filters support and runs `n2n_lrgb_render.py routine`
once per recipe, which writes a mono JPEG per channel, a denoised set, and three
colour composites -- raw, denoised, learned -- into `<image_dir>/Iris/<dso>/`,
and pushes ONE picture, the learned composite. A narrowband target gets ONE
palette (owner, 2026-10-28: "only hoo or sho with the hubble formula"): SHO
with `--gain hubble` when S-II has the frames, HOO otherwise. Until then a
three-filter target got HSO, SHO and HOO in one morning, on the grounds that
the palettes disagree about presentation, not data; three pushes of the same
nebula was the cost, and HSO never earned its slot by eye. A galaxy still
gets HALRGB and LRGB when it has the Ha.

**Stacks are rebuilt (`--force`) only for a target's FIRST render of the run.**
The routine path caches stacks per (dso, filter) — recipes share them — and
after a night of new subs that cache is stale by exactly the frames this job
exists to include, so the first recipe rebuilds it. The remaining recipes reuse
the stacks that first render just wrote: they cost a denoise+compose, not a
20-60 minute stack. If the first render fails, the rest of that target's
recipes are skipped rather than composed from a half-rebuilt cache.

**All three composites are always written and none is chosen.** Whether the
denoised version is better depends on surface brightness, not object class: NGC
6888's bright filaments survive denoising and its denoised SHO is the better
image, while ic1396's 1-5 sigma emission is destroyed by the same model (lab
manual steps 23, 29). Until that threshold is calibrated against more targets,
the choice needs an eye and this job does not make it. The learned composite
(the third since the afternoon of 2026-10-21; the starless `deblurred` one
held that slot from 2026-10-19) is the denoised set with its luminance through
the learned deconvolver, `nn/deblur_model`; a field it refuses still gets the
other two.

Bounded on purpose. Stacking is 20-60 min per target and peak RSS is ~64 GB, so
`--max-targets` caps a run and targets are processed one at a time. A morning
with four new targets renders the deepest and defers the rest to the next run
rather than running the Spark out of memory at 6am.
"""

import argparse
import os
from typing import Optional
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Recipes in preference order. Every rule the filters satisfy is rendered
# (pick_recipes): a narrowband target gets SHO and HOO both when S-II has
# the frames (owner, 2026-10-29: "add the HOO to the morning renders too"),
# HOO alone otherwise. The order decides which recipe stacks first — the
# rest reuse its channel stacks.
RECIPE_RULES = [
    # SHO, rendered with the Hubble gain (PALETTE_GAIN), when S-II has the
    # frames; HOO beside it. HSO (2026-09 to 2026-10-28) put Ha and S-II on
    # opposing primaries so their ratio read as hue, which was honest and
    # also what SHO with gain=hubble now does with the classic colours: on an
    # HII region a true-ratio SHO is green (Ha 3-5x S-II and O-III), and the
    # gain scales each filter's nebula level to the strongest's, stars kept
    # neutral (stacking/color_process.PALETTE_GAIN).
    ("SHO", {"S-II", "Ha", "O-III"}),
    ("HOO", {"Ha", "O-III"}),
    # HALRGB before LRGB when the target has Ha: it stacks first and LRGB
    # reuses its channels. Ha is a detail channel like L (MIN_FRAMES), not a
    # colour (MIN_COLOUR_FRAMES): the excess map is added at full contrast.
    ("HALRGB", {"L", "R", "G", "B", "Ha"}),
    ("LRGB", {"L", "R", "G", "B"}),
]
MIN_FRAMES = 12          # below this a stack is not worth the wall time
# LRGB's colour channels are binned and blurred by the stretch and carry no
# detail of their own, so they need far fewer frames than the luminance that
# does. 2026-09-15: m33 sat at L 18 / R 4 / G 4 / B 3 after a two-hour second
# slot whose plan gives each colour two frames a night; under the 12-frame
# rule for every channel it would not have rendered for six more nights.
MIN_COLOUR_FRAMES = 3
# Narrowband recipes: no L channel, so the shared reference comes from Ha, and
# they take the nebula stretch rather than the galaxy one. Kept as a set because
# adding HSO to RECIPE_RULES without adding it here silently gave that recipe
# `--lum L` (a filter it does not have) and the wrong stretch.
NARROWBAND = {"SHO", "HSO", "HOO"}
# Broadband recipes of a galaxy are white-balanced on the field's stars
# (owner, 2026-10-16). Unbalanced, ngc6946 came out yellow: ~1 mag of Milky
# Way dust reddens the galaxy and every star in the field, and wb=stars
# measured B x1.83 to take it out. Never for narrowband: those are palettes.
WB_STARS_RECIPES = {"LRGB", "HALRGB"}
# The palette gain a recipe is rendered with (n2n_lrgb_render --gain).
PALETTE_GAIN = {"SHO": "hubble"}


def is_galaxy(dso: str) -> bool:
    """The filter planner's own classifier (queued obj_type, then the grid's
    override table / type cache / SIMBAD), so a target is a galaxy here
    exactly when it was planned as one. Unknown is not a galaxy."""
    try:
        from nina_gen.nina_sequence_gen import _classify_object_type
        return _classify_object_type(dso) == "galaxy"
    except Exception as exc:  # noqa: BLE001 -- a lookup failure must not stop the render
        log(f"{dso}: object type unknown ({type(exc).__name__}: {exc}); no white balance")
        return False


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


# A measured weak line (C3, 2026-10-30): SHO is not rendered when the PC's
# convergence record says S-II is below WEAK_LINE_RATIO of Ha (the same rule
# the filter advisor stops shooting it by; fits_processing.filter_advisor).
WEAK_LINE_RATIO = 0.1
MIRROR_CONVERGENCE = os.path.expanduser("~/Desktop/iris_state/live/local/convergence.json")


def emission_for(dso: str, path: str = MIRROR_CONVERGENCE) -> dict:
    """{filter: level / Ha level} from the PC's mirrored convergence record;
    {} when the mirror or the entry is missing. Read only: the PC measures."""
    try:
        import json
        from fits_processing.convergence import emission_ratios
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return emission_ratios(data.get(dso.lower().replace(" ", ""), {}))
    except Exception:  # noqa: BLE001
        return {}


def pick_recipes(counts: dict, emission: Optional[dict] = None,
                 weak_line_ratio: float = WEAK_LINE_RATIO) -> list[str]:
    """The recipes the qualifying filters earn, in stack-first order: every
    rule satisfied (SHO and HOO both on a full narrowband set; HSO is gone).

    Narrowband recipes need MIN_FRAMES in every filter they use. LRGB needs
    MIN_FRAMES of L and at least two colours with MIN_COLOUR_FRAMES each (the
    old rule's tolerance of one missing filter, kept). *emission* ({filter:
    ratio to Ha}, emission_for) drops SHO when S-II is measured weak: a
    palette channel of noise is not worth the stack.
    """
    have = {f for f, n in counts.items() if n >= MIN_FRAMES}
    weak = (emission or {}).get("S-II")
    if weak is not None and float(weak) < weak_line_ratio:
        have.discard("S-II")
    colours = {f for f in ("R", "G", "B") if counts.get(f, 0) >= MIN_COLOUR_FRAMES}
    out = []
    for name, need in RECIPE_RULES:
        if name == "LRGB":
            if "L" in have and len(colours) >= 2:
                out.append(name)
        elif name == "HALRGB":
            if "L" in have and "Ha" in have and len(colours) >= 2:
                out.append(name)
        elif need <= have:
            out.append(name)
    return out


def render_command(dso: str, recipe: str, first: bool, exptime: int,
                   galaxy: bool, final: bool = False) -> list[str]:
    """The n2n_lrgb_render.py routine invocation for one recipe of one target.

    Every routine render writes the raw, denoised and learned composites. A
    galaxy's broadband recipes get --wb stars, measured on the raw set and
    pinned on the denoised one. Only the target's first recipe rebuilds the
    shared channel stacks (--force); the rest reuse them (module docstring).
    *final* (D1) adds --final: the auto stretch, the QC verdict and
    <dso>/final/final.json.
    """
    lum = "Ha" if recipe in NARROWBAND else "L"
    cmd = [sys.executable, os.path.join(_root, "scripts", "n2n_lrgb_render.py"),
           "routine", "--dso", dso.lower().replace(" ", ""), "--recipe", recipe,
           "--lum", lum, "--exptime", str(exptime)]
    if galaxy and recipe in WB_STARS_RECIPES:
        cmd += ["--wb", "stars"]
    if recipe in PALETTE_GAIN:
        cmd += ["--gain", PALETTE_GAIN[recipe]]
    if first:
        cmd.append("--force")
    if final:
        cmd.append("--final")
    return cmd


# ------------------------------------------------------------------ final renders (D1)
# docs/AUTOMATION_PLAN.md D1 (2026-10-30). A target the PC has retired as
# `converged` (control/auto_stop; the queue reaches the Spark in the 11:40
# mirror) gets ONE final render: its preferred recipe with --final, which
# writes <image_dir>/Iris/<dso>/final/final.json with the QC verdict. Finals
# go first and outside --max-targets, and are redone only when the frame
# counts have changed since the record was written. The outcome is reported
# to the conductor's journal (RENDER_DONE, over ssh); nothing is published
# here -- that is D2-D4 behind publish.gate.
MIRROR_QUEUE = os.path.expanduser("~/Desktop/iris_state/live/local/my_instructions.json")
FINAL_MODES = ("off", "shadow")


def final_mode() -> str:
    """cfg publish.final_mode: off / shadow (default shadow)."""
    try:
        from configs import config
        m = str(((config.data().get("publish") or {}).get("final_mode") or "shadow")).lower()
        return m if m in FINAL_MODES else "shadow"
    except Exception:  # noqa: BLE001
        return "shadow"


def converged_targets(path: str = MIRROR_QUEUE) -> list[str]:
    """The queue's `converged` rows, as the Spark sees them; [] without the mirror."""
    try:
        import json
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        return [str(r.get("dso") or "").lower().replace(" ", "")
                for r in rows if str(r.get("status") or "").lower() == "converged" and r.get("dso")]
    except Exception:  # noqa: BLE001
        return []


def final_record_path(dso: str, image_dir: Path) -> Path:
    return Path(image_dir) / "Iris" / dso.lower().replace(" ", "") / "final" / "final.json"


def needs_final(record: Optional[dict], recipe: str, counts: dict) -> bool:
    """Is a final render due? Yes without a record, with another recipe, or
    when the frame counts the record saw differ from today's. Pure."""
    if not record or record.get("recipe") != recipe:
        return True
    was = {k: int((v or {}).get("frames") or 0) for k, v in (record.get("channels") or {}).items()}
    now = {k: int(v) for k, v in counts.items() if k in was}
    return not was or was != now


def plan_finals(converged: list, counts_by_dso: dict, records: dict, emission_by_dso: dict) -> list:
    """[(dso, recipe)] the finals to render, in queue order. Pure.

    *counts_by_dso* {dso: {filter: n}} (count_frames), *records* {dso: final.json
    or None}, *emission_by_dso* {dso: emission_for(dso)}. The preferred recipe
    is pick_recipes' first (SHO over HOO, HALRGB over LRGB); a target whose
    frames earn no recipe, or whose record is current, is skipped.
    """
    out = []
    for dso in converged:
        counts = counts_by_dso.get(dso) or {}
        recipes = pick_recipes(counts, emission_by_dso.get(dso))
        if not recipes:
            continue
        if needs_final(records.get(dso), recipes[0], counts):
            out.append((dso, recipes[0]))
    return out


def count_frames(subs_dir: Path, dso: str, exptime: int) -> dict:
    """{filter: n} LIGHT frames of one target at *exptime* (recent_targets' count, for one)."""
    from astropy.io import fits

    from stacking.color_process import canonical_filter
    counts: dict = defaultdict(int)
    d = subs_dir / dso
    if not d.is_dir():
        return {}
    for fp in d.rglob("*.fits"):
        if fp.parent.name.upper() != "LIGHT":
            continue
        try:
            h = fits.getheader(fp)
            if round(float(h.get("EXPTIME", 0))) != exptime:
                continue
            c = canonical_filter(str(h.get("FILTER", "")).strip())
        except Exception:  # noqa: BLE001
            continue
        if c:
            counts[c] += 1
    return dict(counts)


def post_pc_event(event: str, data: dict, kind: str = "note") -> bool:
    """Report an outcome to the conductor's journal through the PC
    (scripts/post_event.py over ssh, data as base64 JSON so nothing is
    quoted through cmd.exe). False when the PC did not answer."""
    import base64
    import json
    try:
        import morning_trigger as mt
    except Exception:  # noqa: BLE001
        return False
    b64 = base64.b64encode(json.dumps(data).encode("utf-8")).decode()
    script = ("Set-Location '%s'; "
              "$py = if (Test-Path '.venv\\Scripts\\python.exe') { '.venv\\Scripts\\python.exe' } else { 'python' }; "
              "& $py 'scripts\\post_event.py' '%s' --source spark --kind %s --b64 '%s'"
              % (mt.SRT, event, kind, b64))
    out = mt._ps(script, timeout=60)
    return out is not None


def run_finals(subs: Path, exptime: int, log=log) -> int:
    """Render the due finals (plan_finals) and report each to the journal.
    -> the number that failed."""
    mode = final_mode()
    if mode == "off":
        return 0
    conv = converged_targets()
    if not conv:
        return 0
    import json
    counts = {d: count_frames(subs, d, exptime) for d in conv}
    records = {}
    for d in conv:
        try:
            records[d] = json.loads(final_record_path(d, subs).read_text())
        except (OSError, ValueError):
            records[d] = None
    todo = plan_finals(conv, counts, records, {d: emission_for(d) for d in conv})
    log(f"converged in the queue: {', '.join(conv)}; finals due: "
        + (", ".join(f"{d} {r}" for d, r in todo) or "none"))
    failed = 0
    for dso, recipe in todo:
        galaxy = is_galaxy(dso) if recipe in WB_STARS_RECIPES else False
        cmd = render_command(dso, recipe, first=True, exptime=exptime, galaxy=galaxy, final=True)
        log(f"FINAL {dso} {recipe}: {' '.join(cmd[2:])}")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        for line in (r.stdout or "").splitlines()[-8:]:
            log(f"  | {line}")
        rec = None
        try:
            rec = json.loads(final_record_path(dso, subs).read_text())
        except (OSError, ValueError):
            pass
        qc = (rec or {}).get("qc") or {}
        ok = r.returncode == 0 and rec is not None
        failed += not ok
        log(f"FINAL {dso} {recipe}: rc={r.returncode} in {time.time() - t0:.0f}s -- "
            + ("QC PASS" if qc.get("pass") else "QC FAIL: " + "; ".join(qc.get("reasons") or ["no record"])))
        posted = post_pc_event("RENDER_DONE", {
            "dso": dso, "recipe": recipe, "ok": ok, "product": (rec or {}).get("product"),
            "file": (rec or {}).get("file"), "sha256": (rec or {}).get("sha256"),
            "qc_pass": bool(qc.get("pass")), "qc_reasons": qc.get("reasons") or [], "mode": mode})
        log(f"  RENDER_DONE {'journaled' if posted else 'NOT journaled (PC unreachable)'}")
    return failed


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
    # Three, not two: a two-slot night (2026-09-08 onward) fills two by itself,
    # and a cap that exactly fits the normal case defers any incidental third
    # target to the next morning for no reason.
    ap.add_argument("--max-targets", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = config.data()
    machine = cfg.get("machine", {}).get(socket.gethostname()) or {}
    if "subs_dir" not in machine:
        log(f"machine {socket.gethostname()} has no subs_dir — this runs on the Spark only")
        return 1
    subs = Path(machine["subs_dir"])

    # docs/AUTOMATION_PLAN.md B6: a stack written into a full disk dies
    # half-way and leaves a truncated cache; refuse before the first one.
    try:
        from utils import disk_space
        warn_gb, refuse_gb = disk_space.thresholds(cfg)
        u = disk_space.usage(disk_space.data_path(cfg))
        band = disk_space.band(u, warn_gb, refuse_gb)
        if u:
            log(f"disk: {u['free_gb']:.0f} GB free of {u['total_gb']:.0f} GB ({band})")
        if band == "refuse":
            log(f"disk in the refuse band (below {refuse_gb:.0f} GB) — not rendering")
            return 2
    except Exception as exc:  # noqa: BLE001
        log(f"disk check failed ({exc}); rendering anyway")

    # Finals first (D1): a converged target's one publishable render, outside
    # the cap. A failure here never stops the morning's routine renders.
    try:
        finals_failed = run_finals(subs, args.exptime)
    except Exception as exc:  # noqa: BLE001
        log(f"finals failed: {exc}")
        finals_failed = 0

    fresh = recent_targets(subs, args.since_hours, args.exptime)
    if not fresh:
        log(f"no LIGHT frames newer than {args.since_hours:g} h — nothing to render")
        return 1 if finals_failed else 0

    plan = []
    for dso, counts in fresh.items():
        emission = emission_for(dso)
        recipes = pick_recipes(counts, emission)
        summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        if emission.get("S-II") is not None and float(emission["S-II"]) < WEAK_LINE_RATIO \
                and counts.get("S-II", 0) >= MIN_FRAMES:
            log(f"{dso}: S-II measured {emission['S-II']:.2f} of Ha (PC record) — SHO not rendered")
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
        galaxy = is_galaxy(dso) if WB_STARS_RECIPES & set(recipes) else False
        if galaxy:
            log(f"{dso}: galaxy — {' + '.join(r for r in recipes if r in WB_STARS_RECIPES)} "
                f"white-balanced on the field's stars (wb=stars)")
        for i, recipe in enumerate(recipes):
            cmd = render_command(dso, recipe, first=(i == 0), exptime=args.exptime,
                                 galaxy=galaxy)
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
    return rc or (1 if finals_failed else 0)


if __name__ == "__main__":
    sys.exit(main())
