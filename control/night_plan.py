"""Tonight's plan as one decision: the target, its good hours, whether to image, the slots.

Phase 3 of docs/ARCHITECTURE_PLAN.md moves the noon and pre-sunset checks
from the scheduler into the conductor, "noon-check math verbatim". This is
that math in one place, so the scheduler (through
iris_astronomy.astro_dso_visibility.best_object_tonight) and the conductor's
shadow planner (iris/conductor/planner.py) run the same code rather than two
copies that drift:

  * the queue is ranked by rank_targets_tonight (visibility x forecast,
    priority first) -- unchanged;
  * the best target is the top row, and the night is on when it has at least
    MIN_GOOD_HOURS good hours (the scheduler's _MIN_GOOD_HOURS);
  * the slots come from control.slot_plan.plan_slots with the queue's
    max_hours caps (a short monitoring block such as the M31 Cepheid).

    python -m control.night_plan            # tonight's plan as JSON (the conductor runs this)

The CLI is what the conductor calls, in its own process: the planner hits
the network (forecasts, SIMBAD) and must never be able to stall or crash the
process that answers roof and mount requests.
"""
import json
import logging
import os
import sys
from typing import Optional

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

_logger = logging.getLogger(__name__)

MIN_GOOD_HOURS = 3          # == end_points/scheduler_server._MIN_GOOD_HOURS (tested)


def queue_caps(objects) -> dict:
    """{dso: max_hours} for waiting queue entries that carry one. Pure."""
    caps = {}
    for o in objects or []:
        if o.get("status") == "waiting" and o.get("max_hours"):
            caps[o["dso"]] = float(o["max_hours"])
    return caps


def read_queue_caps(instructions_path) -> dict:
    try:
        with open(instructions_path, "r") as fh:
            return queue_caps(json.load(fh))
    except Exception:
        _logger.exception("could not read max_hours from the queue; no caps")
        return {}


def slots_for(rows, dark_hours, weather_by_hour, caps: dict, nina_cfg: dict) -> list:
    """The night's slots from a ranking. Same call the scheduler always made."""
    from control import slot_plan
    return slot_plan.plan_slots(rows, dark_hours, weather_by_hour,
                                min_slot_hours=float(nina_cfg.get("min_slot_hours", 2.0)),
                                max_slots=int(nina_cfg.get("max_slots", 2)),
                                max_hours=caps)


def slots_record(slots) -> list:
    """The slots as plain JSON -- the shape scheduler_state.json carries."""
    return [{"dso": s.name, "start": s.start.isoformat(timespec="minutes"),
             "end": s.end.isoformat(timespec="minutes"), "hours": int(s.good_hours)}
            for s in slots or []]


def decide(rows, slots, min_good_hours: int = MIN_GOOD_HOURS) -> dict:
    """The noon / pre-sunset verdict from a ranking and its slots. Pure."""
    if not rows:
        return {"best": "", "good_hours": 0, "best_start": None, "will_image": False, "slots": []}
    best, good, _alt, _type, start, _sym, _pri = rows[0]
    will = int(good) >= min_good_hours
    return {"best": best, "good_hours": int(good),
            "best_start": start.isoformat(timespec="minutes") if start is not None else None,
            "will_image": will, "slots": slots_record(slots) if will else []}


def plan_tonight(instructions_path: Optional[str] = None) -> dict:
    """Rank the queue against tonight's sky and forecast and decide. Network."""
    from configs import config
    from iris_astronomy import astro_dso_visibility as v
    cfg = config.data()
    path = instructions_path or os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), cfg["location"]["instructions"])
    rows, dark_hours, weather_by_hour = v.rank_targets_tonight(path, verbose=False)
    slots = slots_for(rows, dark_hours, weather_by_hour, read_queue_caps(path),
                      cfg.get("nina", {})) if rows and dark_hours else []
    return decide(rows, slots)


def compare(shadow: dict, scheduler: dict) -> list:
    """Where the conductor's plan and the scheduler's decision differ. Pure.

    *scheduler* is scheduler_state.json: {"dso", "will image tonight", "slots"}.
    """
    diffs = []
    will = scheduler.get("will image tonight")
    will = will if isinstance(will, bool) else str(will).lower() in ("true", "yes")
    if bool(shadow.get("will_image")) != will:
        diffs.append("will image: conductor %s, scheduler %s" % (shadow.get("will_image"), will))
    if shadow.get("best") != scheduler.get("dso"):
        diffs.append("best target: conductor %s, scheduler %s" % (shadow.get("best"), scheduler.get("dso")))
    if will and shadow.get("will_image"):
        ours = [(s["dso"], s["start"], s["end"], s["hours"]) for s in shadow.get("slots") or []]
        theirs = [(s.get("dso"), s.get("start"), s.get("end"), s.get("hours")) for s in scheduler.get("slots") or []]
        if ours != theirs:
            diffs.append("slots: conductor %s, scheduler %s" % (ours, theirs))
    return diffs


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(json.dumps(plan_tonight(sys.argv[1] if len(sys.argv) > 1 else None)))
