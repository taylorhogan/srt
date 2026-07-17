"""Run the blind transit / variable-star search on the Spark and post results
to the web chat.

Usage:
    python scripts/spark_transit_search.py <dso> [filter]

    <dso>     target name as it appears under the frame root (e.g. m92)
    [filter]  filter letter (R, B, L, ...); default * = all LIGHT frames

Designed as the offline analysis node's entry point (see
docs/GPU_TRANSIT_SEARCH_HANDOFF.md): frames arrive via the morning rsync from
the observatory, this runs the all-star search, and the summary + plots are
posted into the one web chat over the Tailnet (utils/webchat_client). Urgent
single-night detections still go via Pushover from inside the search itself.

The __main__ guard is load-bearing: run_transit_search uses a spawn-mode
ProcessPoolExecutor, and spawn workers re-import __main__ — unguarded
top-level code would launch a full search per worker.
"""

import os
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


def _fmt_candidate(c: dict) -> str:
    gaia = c.get("gaia_source_id")
    ident = f"Gaia {gaia} (G={c.get('gaia_g_mag')})" if gaia else "no Gaia match"
    period = c.get("bls_period_d")
    period_txt = f"  P={period:.3f} d" if period else ""
    return (f"score {c.get('score', 0):.1f}  depth {c.get('transit_depth', 0)*100:.1f}%"
            f"{period_txt}  ({c['x']:.0f},{c['y']:.0f})  {ident}")


def _fmt_variable(v: dict) -> str:
    gaia = v.get("gaia_source_id")
    ident = f"Gaia {gaia}" if gaia else "no Gaia match"
    return (f"P={v['ls_period_d']:.3f} d  power {v['ls_power']:.2f}  "
            f"FAP {v['ls_fap']:.0e}  amp {v['amp_pp']*100:.0f}%  "
            f"({v['x']:.0f},{v['y']:.0f})  {ident}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    dso = sys.argv[1]
    filter_name = sys.argv[2] if len(sys.argv) > 2 else "*"

    from transit_search import transit
    from utils.webchat_client import post_to_webchat

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / f"{dso.lower().replace(' ', '')}_{filter_name}_transit.png"

    t0 = time.time()

    def progress(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    try:
        entry = transit.run_transit_search(
            dso_name=dso, filter_name=filter_name,
            image_dir=FRAME_ROOT, output_plot_path=plot_path,
            progress_cb=progress,
        )
    except Exception as exc:
        post_to_webchat(f"🔭 Spark search failed for {dso} [{filter_name}]: {exc}")
        raise

    mins = (time.time() - t0) / 60
    lines = [
        f"🔭 Spark transit/variable search — {dso} [{filter_name}]",
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
        lines.append(f"Top variables (Lomb–Scargle, {len(variables)}):")
        for v in variables[:5]:
            lines.append("  " + _fmt_variable(v))
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


if __name__ == "__main__":
    sys.exit(main())
