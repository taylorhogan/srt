"""Run the blind transit / variable-star search on the Spark and post results
to the web chat.

Usage:
    python scripts/spark_transit_search.py <dso> [filter]
    python scripts/spark_transit_search.py --auto [--dry-run] [--since-days N]

    <dso>     target name as it appears under the frame root (e.g. m92)
    [filter]  filter letter (R, B, L, ...); default * = all LIGHT frames

    --auto        morning-job mode: find LIGHT frames that arrived since the
                  last auto run (marker file; first run = last 24 h), group
                  them by (dso, filter), and search each group that has
                  enough total frames. Never mixes filters in one search —
                  a star's flux relative to the comparison ensemble is
                  colour-dependent, so a mixed series reads as variability.
    --dry-run     with --auto: print the plan, run nothing, leave the marker.
    --since-days N  with --auto: look back N days instead of the marker.

Designed as the offline analysis node's entry point (see
docs/GPU_TRANSIT_SEARCH_HANDOFF.md): frames arrive via the morning rsync from
the observatory (scripts/spark_morning_search.bsh chains this after the
sync), this runs the all-star search, and the summary + plots are posted into
the one web chat over the Tailnet (utils/webchat_client). Urgent single-night
detections still go via Pushover from inside the search itself.

The __main__ guard is load-bearing: run_transit_search uses a spawn-mode
ProcessPoolExecutor, and spawn workers re-import __main__ — unguarded
top-level code would launch a full search per worker.
"""

import os
import socket
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Default frame root on the Spark: destination of sync_nina_targets_to_spark.bsh
FRAME_ROOT = Path(os.environ.get("SRT_FRAME_ROOT", "/home/taylor/Desktop/Targets"))
OUT_DIR = Path(os.environ.get("SRT_TRANSIT_OUT", "local/spark_transit"))

# Announced in every chat post so it's obvious which machine is speaking.
HOST = socket.gethostname()


def _fmt_candidate(c: dict) -> str:
    gaia = c.get("gaia_source_id")
    ident = f"Gaia {gaia} (G={c.get('gaia_g_mag')})" if gaia else "no Gaia match"
    period = c.get("bls_period_d")
    period_txt = f"  P={period:.3f} d" if period else ""
    return (f"score {c.get('score', 0):.1f}  depth {c.get('transit_depth', 0)*100:.1f}%"
            f"{period_txt}  ({c['x']:.0f},{c['y']:.0f})  {ident}")


def _vsx_label(v: dict) -> str:
    """Known VSX variable → 'V1673 Her (RRAB, P=0.673 d)'; new → '★ NEW'.

    Absence of the vsx_name key (VSX query failed) is 'unknown', not new.
    """
    if "vsx_name" not in v:
        return "VSX: unchecked"
    if v["vsx_name"] is None:
        return "★ NEW (not in VSX)"
    typ = v.get("vsx_type") or "?"
    per = v.get("vsx_period")
    per_txt = f", P={per:.3f} d" if per else ""
    return f"{v['vsx_name']} ({typ}{per_txt})"


def _fmt_variable(v: dict) -> str:
    crowd = "  ⚠blended" if v.get("blended") else ""
    return (f"P={v['ls_period_d']:.3f} d  power {v['ls_power']:.2f}  "
            f"amp {v['amp_pp']*100:.0f}%  ({v['x']:.0f},{v['y']:.0f})  "
            f"→ {_vsx_label(v)}{crowd}")


def _bucket(v: dict) -> str:
    """Which report group a variable belongs to.

    A ★ NEW that is *blended* (aperture overlaps a neighbour) is quarantined:
    in a crowded field the "new" detection is most likely a neighbour's flux
    leaking in, not a real uncatalogued variable, so it goes to 'crowded' for
    vetting rather than being announced as a discovery.
    """
    if "vsx_name" not in v:
        return "unchecked"
    if v["vsx_name"] is not None:
        return "known"
    return "crowded" if v.get("blended") else "new"


def search_and_post(dso: str, filter_name: str) -> int:
    """Run one (dso, filter) search and post summary + plots to the web chat."""
    from transit_search import transit
    from utils.webchat_client import post_to_webchat

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / f"{dso.lower().replace(' ', '')}_{filter_name}_transit.png"

    t0 = time.time()

    def progress(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {dso} [{filter_name}] {msg}", flush=True)

    try:
        entry = transit.run_transit_search(
            dso_name=dso, filter_name=filter_name,
            image_dir=FRAME_ROOT, output_plot_path=plot_path,
            progress_cb=progress,
        )
    except Exception as exc:
        post_to_webchat(f"🔭 [{HOST}] search failed for {dso} [{filter_name}]: {exc}")
        raise

    mins = (time.time() - t0) / 60
    lines = [
        f"🔭 [{HOST}] transit/variable search — {dso} [{filter_name}]",
        f"{entry['frame_count']} frames over {entry['baseline_days']:.2f} d, "
        f"{entry['n_stars_searched']}/{entry['n_stars']} stars searched "
        f"({mins:.0f} min)",
    ]
    candidates = entry.get("candidates", [])
    if candidates:
        lines.append("Top transit candidate:")
        lines.append("  " + _fmt_candidate(candidates[0]))

    variables = entry.get("variables", [])
    if variables:
        groups = {"known": [], "new": [], "crowded": [], "unchecked": []}
        for v in variables:
            groups[_bucket(v)].append(v)
        tally = f"{len(groups['known'])} known, {len(groups['new'])} new"
        if groups["crowded"]:
            tally += f", {len(groups['crowded'])} crowded"
        if groups["unchecked"]:
            tally += f", {len(groups['unchecked'])} unchecked"
        lines.append(f"Variables (Lomb–Scargle, {len(variables)}): {tally}")
        if groups["known"]:
            lines.append("Known (VSX):")
            lines += ["  " + _fmt_variable(v) for v in groups["known"]]
        if groups["new"]:
            lines.append("★ Candidate NEW variables (not in VSX, uncrowded):")
            lines += ["  " + _fmt_variable(v) for v in groups["new"]]
        if groups["crowded"]:
            lines.append("Crowded — needs vetting (not in VSX but aperture "
                         "blended, likely a neighbour):")
            lines += ["  " + _fmt_variable(v) for v in groups["crowded"]]
        if groups["unchecked"]:
            lines.append("Unchecked (VSX query unavailable):")
            lines += ["  " + _fmt_variable(v) for v in groups["unchecked"]]
    else:
        lines.append("No significant variables found.")

    post_to_webchat("\n".join(lines),
                    image_path=plot_path if plot_path.exists() else None)
    vplot = entry.get("variables_plot")
    if vplot and Path(vplot).exists():
        post_to_webchat(f"{dso} [{filter_name}] — variable light curves "
                        f"(raw + phase-folded)", image_path=Path(vplot))
    print("posted to web chat")
    return 0


def _light_frames(root: Path):
    """Every LIGHT frame under the tree: <object>/<scope>/<date>/LIGHT/*.fits."""
    for f in root.rglob("*.fits"):
        if f.parent.name.upper() == "LIGHT":
            yield f


def _frame_filter(path: Path) -> str:
    """FILTER from the FITS header (header-only read), '' when absent."""
    try:
        from astropy.io import fits
        return str(fits.getheader(path).get("FILTER", "") or "")
    except Exception:
        return ""


def auto_mode(dry_run: bool, since_days: float | None) -> int:
    """Morning-job mode: search every (dso, filter) with newly arrived frames.

    "New" = frame mtime after the marker file's mtime (rsync -a preserves
    source mtimes, so this is the observation night, not the copy time).
    The marker is touched only after every search finishes, so a crashed
    morning reruns tomorrow instead of losing the night.
    """
    from configs import config

    marker = OUT_DIR / "last_auto_run"
    if since_days is not None:
        cutoff = time.time() - since_days * 86400
    elif marker.exists():
        cutoff = marker.stat().st_mtime
    else:
        cutoff = time.time() - 86400

    from utils.webchat_client import post_to_webchat

    new_by_group: dict[tuple[str, str], int] = {}
    for f in _light_frames(FRAME_ROOT):
        if f.stat().st_mtime <= cutoff:
            continue
        dso = f.relative_to(FRAME_ROOT).parts[0]
        filt = _frame_filter(f)
        if filt:
            new_by_group[(dso, filt)] = new_by_group.get((dso, filt), 0) + 1

    since_txt = time.strftime("%Y-%m-%d %H:%M", time.localtime(cutoff))

    if not new_by_group:
        msg = (f"🔭 [{HOST}] morning search: no new LIGHT frames since "
               f"{since_txt} — nothing to do")
        print(msg)
        if not dry_run:
            post_to_webchat(msg)
        return 0

    min_frames = int(config.data().get("transit", {}).get("min_frames", 20))
    plan: list[tuple[str, str, int, int]] = []
    for (dso, filt), n_new in sorted(new_by_group.items()):
        total = sum(
            1 for f in _light_frames(FRAME_ROOT / dso)
            if _frame_filter(f).lower() == filt.lower()
        )
        plan.append((dso, filt, n_new, total))

    report = [f"🔭 [{HOST}] morning search — new frames since {since_txt}:"]
    runnable = []
    for dso, filt, n_new, total in plan:
        if total >= min_frames:
            status = "searching"
            runnable.append((dso, filt))
        else:
            status = f"skip (only {total} total, need {min_frames})"
        report.append(f"  {dso} [{filt}]: {n_new} new / {total} total → {status}")
    print("\n".join(report))

    if dry_run:
        print("auto: dry run — no searches started, marker untouched")
        return 0
    post_to_webchat("\n".join(report))

    failures = 0
    for dso, filt in runnable:
        try:
            search_and_post(dso, filt)
        except Exception:
            import traceback
            traceback.print_exc()
            failures += 1

    if failures == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    return 1 if failures else 0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--auto" in flags:
        since_days = None
        if "--since-days" in flags:
            i = sys.argv.index("--since-days")
            since_days = float(sys.argv[i + 1])
            args = [a for a in args if a != sys.argv[i + 1]]
        return auto_mode(dry_run="--dry-run" in flags, since_days=since_days)

    if not args:
        print(__doc__)
        return 2
    return search_and_post(args[0], args[1] if len(args) > 1 else "*")


if __name__ == "__main__":
    sys.exit(main())
