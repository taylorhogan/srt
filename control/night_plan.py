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
    python -m control.night_plan --refresh  # the noon form: refresh the queue's hours first

The record also carries three things that are not the ranking (docs/
AUTOMATION_PLAN.md, 2026-10-29): "auto_stop", the waiting targets the
convergence and frame-goal verdicts say are finished (control/auto_stop;
shadow lists them, live retires them); "disk", the image drive's free
space band (utils/disk_space) -- a drive in the refuse band calls the night
off here, with the reason in "why", so the decision is in the plan the
conductor journals rather than in a refusal at 20:30; and, per slot, the
FILTERS (C1, 2026-10-30): "filters" is the ratio the sequence generator
should write for that target, "filter_source" says who decided it
(explicit: the queue's `filters` plan, which always wins; monitor: a
light-curve target's own fixed plan; advisor: fits_processing.filter_advisor
from the frames on disk, tonight's seeing and the moon; type: nothing
decided here, the generator's object-type split applies) and
"filter_reasons" is the advisor's why. "filter_mode" is cfg nina.filter_advisor
as the planner saw it: "shadow" records the ratio and the scheduler ignores
it, "default" makes the scheduler write it, "off" records nothing. The
mode rides the record so the journal says what the sequence was built from.
"expired_filter_plans" (C2) lists the explicit plans the noon refresh dropped
because their last night had passed (control/filter_plan).

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
    out = []
    for s in slots or []:
        rec = {"dso": s.name, "start": s.start.isoformat(timespec="minutes"),
               "end": s.end.isoformat(timespec="minutes"), "hours": int(s.good_hours)}
        if getattr(s, "filters", None):
            rec["filters"] = dict(s.filters)
        out.append(rec)
    return out


def slots_from_record(records) -> list:
    """control.slot_plan Slots back from slots_record's JSON. Pure.

    How the scheduler, carrying out the conductor's plan (ADR 0015), gets
    the slots it generates the sequence from without planning them itself.
    The record carries no priority; it only ordered the planning, which is
    done, so the default stands in.
    """
    from datetime import datetime
    from control.slot_plan import Slot
    out = []
    for r in records or []:
        filters = r.get("filters") if isinstance(r.get("filters"), dict) else None
        out.append(Slot(name=str(r["dso"]), start=datetime.fromisoformat(r["start"]),
                        end=datetime.fromisoformat(r["end"]), good_hours=int(r["hours"]),
                        priority=int(r.get("priority", 5)),
                        filters={k: int(v) for k, v in filters.items()} if filters else None))
    return out


def decide(rows, slots, min_good_hours: int = MIN_GOOD_HOURS) -> dict:
    """The noon / pre-sunset verdict from a ranking and its slots. Pure."""
    if not rows:
        return {"best": "", "good_hours": 0, "best_start": None, "will_image": False, "slots": []}
    best, good, _alt, _type, start, _sym, _pri = rows[0]
    will = int(good) >= min_good_hours
    return {"best": best, "good_hours": int(good),
            "best_start": start.isoformat(timespec="minutes") if start is not None else None,
            "will_image": will, "slots": slots_record(slots) if will else []}


def disk_band(usage, warn_gb: float, refuse_gb: float) -> dict:
    """{"free_gb", "total_gb", "band": ok|warn|refuse|unknown}. Pure.

    Unknown (the drive could not be read) never blocks a night: free space
    is a capacity question, the morning check reports it."""
    if not usage:
        return {"free_gb": None, "total_gb": None, "band": "unknown",
                "warn_gb": warn_gb, "refuse_gb": refuse_gb}
    from utils import disk_space
    return {"free_gb": float(usage["free_gb"]), "total_gb": usage.get("total_gb"),
            "band": disk_space.band(usage, warn_gb, refuse_gb),
            "warn_gb": warn_gb, "refuse_gb": refuse_gb}


def apply_disk(plan: dict, disk: dict) -> dict:
    """The disk verdict on a decision: a refuse band calls the night off. Pure."""
    out = dict(plan)
    out["disk"] = disk
    if disk.get("band") == "refuse" and out.get("will_image"):
        out["will_image"] = False
        out["slots"] = []
        out["why"] = ("disk: %.0f GB free on the image drive, a night needs %.0f GB"
                      % (disk["free_gb"], disk["refuse_gb"]))
    return out


def read_disk(cfg) -> dict:
    from utils import disk_space
    warn_gb, refuse_gb = disk_space.thresholds(cfg)
    return disk_band(disk_space.usage(disk_space.data_path(cfg)), warn_gb, refuse_gb)


def read_auto_stop(instructions_path, cfg) -> list:
    """control.auto_stop.evaluate on the queue, [] when off or unreadable."""
    from control import auto_stop
    if auto_stop.mode(cfg) == "off":
        return []
    try:
        with open(instructions_path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
    except Exception:
        _logger.exception("auto-stop: could not read the queue")
        return []
    return auto_stop.evaluate(rows)


# ------------------------------------------------------------------ filters (C1)

FILTER_MODES = ("off", "shadow", "default")
DEFAULT_FILTER_MODE = "shadow"


def filter_mode(cfg) -> str:
    """cfg nina.filter_advisor: off / shadow / default. Anything else reads as shadow."""
    m = str(((cfg or {}).get("nina") or {}).get("filter_advisor", DEFAULT_FILTER_MODE)).lower()
    return m if m in FILTER_MODES else DEFAULT_FILTER_MODE


def _norm(name) -> str:
    return str(name or "").lower().replace(" ", "")


def queue_row(rows, dso) -> Optional[dict]:
    """The first non-completed queue row for *dso* (instructions' own match rule). Pure."""
    key = _norm(dso)
    for r in rows or []:
        if _norm(r.get("dso")) == key and r.get("status") != "completed":
            return r
    return None


def _ratio(plan) -> Optional[dict]:
    if not isinstance(plan, dict):
        return None
    out = {}
    for f, n in plan.items():
        try:
            out[str(f)] = int(n)
        except (TypeError, ValueError):
            continue
    return out if any(v > 0 for v in out.values()) else None


def _explicit(row, night) -> Optional[dict]:
    """The row's `filters` plan if it applies to *night* (control/filter_plan: an
    expired plan is no plan); *night* None = no expiry check."""
    from control import filter_plan
    if night is None:
        return _ratio((row or {}).get("filter_plan"))
    return filter_plan.active_plan(row, night)


def slot_filters(dso, row, advice, night=None) -> dict:
    """Who decides tonight's filters for one slot, and what. Pure.

    *row* is the target's queue row (or None), *advice* the filter advisor's
    result for the slot (or None), *night* the night's date (for the plan's
    expiry, C2). Explicit > monitor > advisor > type: the same order the
    sequence generator keeps (nina_sequence_gen._plan_for), so the record
    and the file agree by construction.
    -> {"filters": {filter: ratio} or None, "filter_source": str, "filter_reasons": [str]}
    """
    row = row or {}
    explicit = _explicit(row, night)
    if explicit:
        from control import filter_plan
        src = "monitor" if row.get("lightcurve") else "explicit"
        until = row.get(filter_plan.UNTIL_KEY)
        why = ("monitor target (lightcurve): its own fixed plan" if src == "monitor"
               else "the queue's `filters` plan (explicit plans always win)")
        if until:
            why += " -- last night %s" % str(until)[:10]
        return {"filters": explicit, "filter_source": src, "filter_reasons": [why]}
    advice = advice or {}
    plan = _ratio(advice.get("plan"))
    if plan and advice.get("action") in ("plan", "done"):
        return {"filters": plan, "filter_source": "advisor",
                "filter_reasons": [str(r) for r in advice.get("reasons") or []]}
    reasons = [str(r) for r in advice.get("reasons") or []]
    reasons.append("no plan from the advisor: the generator's object-type split applies")
    return {"filters": None, "filter_source": "type", "filter_reasons": reasons}


def apply_filters(plan: dict, rows, advise, mode: str, night=None) -> dict:
    """Decide the filters for every slot in *plan* and record them. Pure given
    *advise* (dso, start, end) -> advice dict. Mode "off" records only the
    mode. *night* (a date) is what an explicit plan's expiry is judged
    against; None = the first slot's own night."""
    plan = dict(plan or {})
    plan["filter_mode"] = mode
    if mode == "off":
        return plan
    from datetime import datetime
    if night is None:
        try:
            night = datetime.fromisoformat((plan.get("slots") or [{}])[0]["start"]).date()
        except (KeyError, IndexError, TypeError, ValueError):
            night = None
    slots = []
    for rec in plan.get("slots") or []:
        rec = dict(rec)
        row = queue_row(rows, rec.get("dso"))
        advice = None
        if not _explicit(row, night):
            try:
                advice = advise(rec.get("dso"), datetime.fromisoformat(rec["start"]),
                                datetime.fromisoformat(rec["end"]))
            except Exception as exc:  # noqa: BLE001 -- a broken advisor is a reason, not a crash
                advice = {"plan": None, "action": "error",
                          "reasons": ["advisor failed: %s: %s" % (type(exc).__name__, exc)]}
        rec.update(slot_filters(rec.get("dso"), row, advice, night))
        if rec.get("filters") is None:
            rec.pop("filters", None)
        slots.append(rec)
    plan["slots"] = slots
    return plan


def read_filters(plan: dict, instructions_path, cfg) -> dict:
    """apply_filters on the live inputs: the queue file and the advisor. Network."""
    mode = filter_mode(cfg)
    if mode == "off" or not plan.get("will_image"):
        plan = dict(plan or {})
        plan["filter_mode"] = mode
        return plan
    try:
        with open(instructions_path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
    except Exception:
        _logger.exception("filters: could not read the queue; the advisor alone")
        rows = []
    from fits_processing import filter_advisor
    return apply_filters(plan, rows, filter_advisor.advise_for, mode)


def describe_filters(plan: dict, written: Optional[dict] = None) -> str:
    """The noon chat message for the plan's filters, '' when there are none.

    *written* is {dso: {filter: frames}} read back from the sequence file,
    so the owner sees the ratio beside what N.I.N.A will shoot."""
    mode = (plan or {}).get("filter_mode", "off")
    slots = [s for s in (plan or {}).get("slots") or [] if s.get("filter_source")]
    if mode == "off" or not slots:
        return ""
    head = ("Filters tonight:" if mode == "default"
            else "Filter advisor (shadow, nothing changed):")
    lines = [head]
    for s in slots:
        ratio = " ".join("%s=%s" % kv for kv in (s.get("filters") or {}).items()) or "-"
        seq = ""
        if written is not None:
            got = written.get(s["dso"]) or {}
            seq = "  |  sequence %s" % (" ".join("%s×%s" % kv for kv in got.items()) or "-")
        lines.append("%s %s-%s  %s %s (%s)%s" % (
            s["dso"], s["start"][11:16], s["end"][11:16],
            "filters" if mode == "default" else "advised", ratio, s["filter_source"], seq))
        lines.extend("    " + r for r in s.get("filter_reasons") or [])
    if mode != "default":
        lines.append("  (cfg nina.filter_advisor = default makes the plan's ratio the sequence's)")
    return "\n".join(lines)


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
    plan = decide(rows, slots)
    try:
        plan = apply_disk(plan, read_disk(cfg))
    except Exception:
        _logger.exception("disk check failed; planning without it")
    try:
        plan = read_filters(plan, path, cfg)
    except Exception:
        _logger.exception("filters failed; the generator decides on its own tonight")
        plan["filter_mode"] = "shadow"
    try:
        plan["auto_stop"] = read_auto_stop(path, cfg)
    except Exception:
        _logger.exception("auto-stop failed; planning without it")
        plan["auto_stop"] = []
    if _EXPIRED_PLANS:
        plan["expired_filter_plans"] = list(_EXPIRED_PLANS)
    return plan


_EXPIRED_PLANS: list = []     # what refresh_queue dropped this run, for the record


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


def refresh_queue() -> None:
    """The noon check's queue refresh (hours above horizon, air mass, best
    date), run before the noon decision as the scheduler always did.

    With planner.auto_stop = "live" the finished targets are retired here
    too (status "converged", control/auto_stop), BEFORE the ranking reads
    the queue, so tonight's plan never names a target that is done."""
    from control import instructions
    instructions.calc_and_store_hours_above_horizon()
    try:
        # Explicit `filters` plans that have run their nights (C2) go first,
        # so the plan below sees the advisor's ratio for those targets.
        gone = instructions.expire_filter_plans()
        _EXPIRED_PLANS[:] = gone
        for g in gone:
            _logger.info("filters: %s's plan %s expired (last night %s)", g["dso"], g["plan"], g["until"])
    except Exception:
        _logger.exception("filter plan expiry failed; the plans stand")
    try:
        from configs import config
        from control import auto_stop
        cfg = config.data()
        if auto_stop.mode(cfg) == "live":
            path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                                cfg["location"]["instructions"])
            for v in read_auto_stop(path, cfg):
                if instructions.set_status_db(v["dso"], auto_stop.CONVERGED,
                                              why="; ".join(v["why"]), who="auto_stop"):
                    _logger.info("auto-stop: %s -> converged (%s)", v["dso"], "; ".join(v["why"]))
    except Exception:
        _logger.exception("auto-stop (live) failed; the queue stands")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = [a for a in sys.argv[1:] if a != "--refresh"]
    if "--refresh" in sys.argv[1:]:
        refresh_queue()
    print(json.dumps(plan_tonight(args[0] if args else None)))
