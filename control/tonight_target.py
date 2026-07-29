"""Tonight's selected DSO, published by the imaging grid and read by everyone else.

Only one component decides tonight's target: ``best_object_tonight`` in
``iris_astronomy.astro_dso_visibility``. Every other consumer reads that
decision from here rather than re-running the selection, for two reasons:

1. **Cost and safety.** The selector resolves each unqueued target through
   ``SkyCoord.from_name(..., cache=False)`` -- one SIMBAD round trip apiece,
   ~78 s for a 37-target queue -- and calls ``map_az_to_horizon()``, which
   draws to global pyplot state under a TkAgg backend. Neither belongs in a
   polled web request or a background thread.

2. **Correctness.** The selection is a function of *when it runs*: dark hours,
   weather and altitude all move through the night. A consumer that recomputes
   at 04:00 (e.g. the end-of-night SNR report) would rank against the *coming*
   night and name a different DSO than the one actually imaged. Reading the
   published pick returns what the observatory really targeted.

Deliberately free of astropy/matplotlib imports so the web server can read the
value without dragging in the astronomy stack.
"""
import json
import os
import time
from typing import Optional

if __package__ is None or __package__ == "":
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

# A night's pick is good for the night; anything older is a leftover.
MAX_AGE_SECONDS = 24 * 3600.0


def _path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(root, config.data()["location"]["tonight_target"])


def publish(dso: str, good_hours: int = 0) -> None:
    """Record the grid's pick. Best-effort — never raises."""
    try:
        path = _path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "dso": dso,
                "good_hours": good_hours,
                "computed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, f)
    except Exception as e:
        print(f"WARN: could not publish tonight's target: {e}")


def read() -> Optional[str]:
    """The DSO the grid last selected, or None if absent or stale.

    Cheap: one small JSON read. Never raises.
    """
    try:
        path = _path()
        if not os.path.exists(path):
            return None
        if (time.time() - os.path.getmtime(path)) > MAX_AGE_SECONDS:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("dso") or None
    except Exception:
        return None
