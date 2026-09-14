"""Which targets got light frames tonight -- from the frames on disk, not
from the plan. Stdlib only, so the post-run convergence hook and its test
run anywhere.

N.I.N.A files a night's frames under ``<image_dir>/<dso>/<rig>/<night>/LIGHT``
where *night* is the date the night STARTED (frames after midnight keep the
evening's date), so a run that ends at dawn on the 14th lives under the 13th.
``night_of(now)`` gives that date: the calendar date twelve hours earlier.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

LIGHT_SUFFIXES = (".fits", ".fit", ".fts")


def night_of(now: datetime) -> date:
    """The night a moment belongs to: after noon it is today, before noon
    it is yesterday (the same rule as the journal and the shadow report)."""
    return (now - timedelta(hours=12)).date()


def dsos_imaged_on(image_dir: Path, night: date, min_frames: int = 1) -> list[str]:
    """Target names with at least *min_frames* light frames filed under
    *night*, sorted. Anything under the rendered-products tree (``Iris/``)
    is ignored, as are directories that hold no LIGHT folder for the night.
    """
    image_dir = Path(image_dir)
    stamp = night.isoformat()
    out = []
    if not image_dir.is_dir():
        return out
    for dso_dir in sorted(image_dir.iterdir()):
        if not dso_dir.is_dir() or dso_dir.name.lower() == "iris":
            continue
        n = 0
        for light in dso_dir.glob("*/%s/LIGHT" % stamp):
            if light.is_dir():
                n += sum(1 for f in light.iterdir()
                         if f.is_file() and f.suffix.lower() in LIGHT_SUFFIXES)
        if n >= min_frames:
            out.append(dso_dir.name)
    return out
