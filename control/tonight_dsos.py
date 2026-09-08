"""tonight_dsos.py -- which targets did the night actually image, in order?

    from control import tonight_dsos
    tonight_dsos.imaged(image_dir, since=imaging_start)   # ["ngc7380", "m33"]

Before two-slot nights (2026-09-08) every end-of-night step and every
no-argument command took "the DSO" from the newest LIGHT frame on disk, and
the end-of-night convergence run took it from the grid's single published
pick. After NGC 7380 then M33 both of those name M33 and the Wizard gets no
summary, no curve and no chart. This is the one place that answers the
question from the frames themselves: the DSO directories that received
LIGHT frames since the run started, ordered by their first frame.

Directory-based on purpose, not header-based: every path that collects
lights already requires the parent to be named LIGHT and the DSO to be the
first path element under the image dir, so this agrees with the stacker,
the frame watcher and the summary about what belongs to whom.
"""
import os
import time
from pathlib import Path
from typing import Optional


def imaged(image_dir, since: Optional[float] = None, max_age_hours: float = 20.0) -> list:
    """DSO names (directory names) with LIGHT frames newer than *since*
    (epoch seconds; default: the last *max_age_hours*), ordered by first frame.
    Never raises; [] when nothing qualifies."""
    root = Path(str(image_dir))
    cutoff = float(since) if since else time.time() - max_age_hours * 3600.0
    first_seen: dict = {}
    try:
        for fp in root.rglob("*.fits"):
            if fp.parent.name.upper() != "LIGHT":
                continue
            try:
                m = fp.stat().st_mtime
            except OSError:
                continue
            if m < cutoff:
                continue
            try:
                dso = fp.relative_to(root).parts[0]
            except ValueError:
                continue
            if dso.lower() == "iris":          # finished products, never data
                continue
            if dso not in first_seen or m < first_seen[dso]:
                first_seen[dso] = m
    except Exception:  # noqa: BLE001
        return []
    return [d for d, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]


def imaging_start(project_root) -> Optional[float]:
    """Epoch seconds the current/last run started (imaging_start.txt), or None."""
    try:
        from datetime import datetime
        with open(os.path.join(str(project_root), "imaging_start.txt"), encoding="utf-8") as fh:
            return datetime.fromisoformat(fh.read().strip()).timestamp()
    except Exception:  # noqa: BLE001
        return None


def others_hint(image_dir, chosen: str, project_root, verb: str) -> str:
    """'Tonight also imaged X; run `verb x` for it.' or '' -- the nudge a
    no-argument command appends when the night had more than one target."""
    try:
        names = imaged(image_dir, since=imaging_start(project_root))
        others = [n for n in names if n.lower() != str(chosen).lower()]
        if not others:
            return ""
        return ("\nTonight also imaged %s; run %s for %s."
                % (", ".join(others), " / ".join("`%s %s`" % (verb, o) for o in others),
                   "it" if len(others) == 1 else "them"))
    except Exception:  # noqa: BLE001
        return ""
