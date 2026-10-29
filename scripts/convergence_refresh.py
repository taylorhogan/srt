"""convergence_refresh.py -- is convergence.json current for every target imaged last night?

    python scripts/convergence_refresh.py            # refresh what is stale
    python scripts/convergence_refresh.py --dry-run  # say what is stale, compute nothing

Run daily at 09:15 by the scheduled task IrisConvergenceRefresh
(convergence_refresh.cmd), after the 09:00 morning check and before the noon
plan reads the verdicts (docs/AUTOMATION_PLAN.md, A1).

WHY. The auto-stop (control/auto_stop) retires a target on
convergence.is_dso_done, which reads local/convergence.json. That file was
written only by the end-of-night SNR loop inside the imaging thread and by
the chat `snr` command; a night that ended badly (stop!, a crash, a close
that would not confirm) left last night's frames uncounted, and the noon
plan then judged the target on stale numbers. This run is the second chance:
for every DSO directory with LIGHT frames newer than its convergence entry it
recomputes the per-filter curves with the same code (compute_dso_convergence)
and merges them in.

What is stale (pure, stale_targets): a target imaged in the window whose
entry is missing, or whose newest `updated` date is before the day its
newest frame landed. Light-curve monitors are left alone: a convergence
curve says nothing about a Cepheid's brightness.

Each refreshed target is journaled as a CONVERGENCE_EVAL note through the
conductor (best effort; an unreachable conductor only loses the note).
"""
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

_logger = logging.getLogger(__name__)

WINDOW_HOURS = 20.0


def _key(dso: str) -> str:
    return str(dso).lower().replace(" ", "")


def newest_light(dso_dir) -> Optional[float]:
    """Epoch mtime of the newest LIGHT frame under *dso_dir*, or None."""
    newest = None
    try:
        for fp in Path(str(dso_dir)).rglob("*.fits"):
            if fp.parent.name.upper() != "LIGHT":
                continue
            try:
                m = fp.stat().st_mtime
            except OSError:
                continue
            if newest is None or m > newest:
                newest = m
    except Exception:  # noqa: BLE001
        return None
    return newest


def stale_targets(imaged: list, entries: dict, newest_frame: dict,
                  monitors=()) -> list:
    """[(dso, why)] of imaged targets whose convergence entry is older than
    their newest frame. Pure.

    entries      : convergence.json ({dso: {filter: {"updated": "YYYY-MM-DD", ...}}})
    newest_frame : {dso: epoch seconds of the newest LIGHT frame}
    monitors     : dso names that are light-curve monitors (skipped)
    """
    skip = {_key(m) for m in monitors or ()}
    out = []
    for dso in imaged or []:
        k = _key(dso)
        if k in skip:
            continue
        newest = newest_frame.get(dso, newest_frame.get(k))
        if newest is None:
            continue
        frame_day = date.fromtimestamp(float(newest))
        filters = entries.get(k) or {}
        if not filters:
            out.append((dso, "no convergence entry"))
            continue
        updated = []
        for info in filters.values():
            try:
                updated.append(date.fromisoformat(str((info or {}).get("updated"))))
            except (TypeError, ValueError):
                pass
        if not updated:
            out.append((dso, "entry carries no date"))
        elif max(updated) < frame_day:
            out.append((dso, "entry updated %s, newest frame %s"
                        % (max(updated).isoformat(), frame_day.isoformat())))
    return out


def _monitors(queue_path) -> list:
    try:
        with open(queue_path, encoding="utf-8") as fh:
            return [r.get("dso") for r in json.load(fh) if r.get("lightcurve")]
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report; compute nothing")
    ap.add_argument("--hours", type=float, default=WINDOW_HOURS,
                    help="how far back a frame counts as last night's")
    args = ap.parse_args()
    from utils import utils
    utils.set_logger()
    from configs import config
    from control import tonight_dsos
    from fits_processing import convergence as conv

    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    imaged = tonight_dsos.imaged(image_dir, max_age_hours=args.hours)
    newest = {d: newest_light(image_dir / d) for d in imaged}
    queue_path = os.path.join(ROOT, cfg["location"]["instructions"])
    stale = stale_targets(imaged, conv.load_convergence(), newest, _monitors(queue_path))
    if not stale:
        _logger.info("convergence refresh: %d target(s) imaged, all entries current", len(imaged))
        print("convergence refresh: nothing stale")
        return 0
    for dso, why in stale:
        _logger.info("convergence refresh: %s is stale (%s)", dso, why)
        print("%s: %s" % (dso, why))
    if args.dry_run:
        return 0
    rc = 0
    for dso, why in stale:
        t0 = datetime.now()
        try:
            results = conv.compute_dso_convergence(dso, image_dir)
        except Exception:  # noqa: BLE001
            _logger.exception("convergence refresh: %s failed", dso)
            rc = 1
            continue
        if not results:
            _logger.warning("convergence refresh: %s produced no curves", dso)
            continue
        conv.save_convergence(dso, results)
        took = (datetime.now() - t0).total_seconds()
        _logger.info("convergence refresh: %s saved %d filter(s) in %.0f s", dso, len(results), took)
        try:
            from iris import client as conductor_client
            conductor_client.post_event(
                "CONVERGENCE_EVAL", "convergence_refresh",
                {"dso": dso, "why": why, "filters": sorted(results),
                 "done": bool(conv.is_dso_done(dso)), "took_s": round(took)},
                kind="note")
        except Exception:  # noqa: BLE001
            _logger.warning("convergence refresh: note not journaled for %s", dso, exc_info=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
