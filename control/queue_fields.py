"""The queue's hand-set fields, as chat commands (E3 of docs/AUTOMATION_PLAN.md, 2026-10-30).

    hold <dso> [off]              status held (the planner never takes it) / back to waiting
    moon <dso> any|avoid|dark|clear   the moon rule for the target (iris_astronomy.moon_cost)
    maxhours <dso> N|clear        a short block a night (control/slot_plan caps)
    type <dso> galaxy|nebula|clear    the object type the filter plan reads

Until now these were hand edits of my_instructions.json. parse() is pure:
it turns the typed words into (dso, field, value) or an error in the
owner's words, and the command functions on the PC write through
instructions.set_status_db (hold) and instructions.set_field_db (the rest),
the one writer for a row's machine-read flags.
"""
from typing import Optional, Tuple

MOON_MODES = ("any", "avoid", "dark")       # iris_astronomy.moon_cost.MODES
TYPES = ("galaxy", "nebula")
CLEAR_WORDS = ("clear", "none", "off", "default")
MAX_HOURS_RANGE = (0.25, 12.0)
FIELDS = {"moon": "moon", "maxhours": "max_hours", "type": "obj_type"}
REPLAN_STATES = ("IDLE_DAY", "ARMED")       # where REPLAN_REQUESTED goes to PLANNING (iris/core/machine)
USAGE = {"hold": "hold <dso> [off]", "moon": "moon <dso> any|avoid|dark|clear",
         "maxhours": "maxhours <dso> N|clear", "type": "type <dso> galaxy|nebula|clear"}


def parse(cmd: str, args: list) -> dict:
    """-> {"dso", "field", "value", "error"}; value None clears the field.

    The DSO may be two words ("m 31"): the last token is the value when it
    is one the command knows, else the whole tail is the name and the value
    is absent (which `hold` reads as "on" and the others as an error).
    """
    cmd = str(cmd).lower()
    toks = [str(a) for a in (args or []) if str(a).strip()]
    out = {"dso": "", "field": FIELDS.get(cmd, "status" if cmd == "hold" else None), "value": None, "error": None}
    if cmd not in USAGE:
        out["error"] = "unknown command %s" % cmd
        return out
    if not toks:
        out["error"] = "Usage: " + USAGE[cmd]
        return out
    last = toks[-1].lower()
    name_toks, value_tok = toks, None
    if cmd == "hold":
        if last in ("off", "on") or last in CLEAR_WORDS:
            name_toks, value_tok = toks[:-1], last
    elif cmd == "moon":
        if last in MOON_MODES or last in CLEAR_WORDS:
            name_toks, value_tok = toks[:-1], last
    elif cmd == "maxhours":
        if last in CLEAR_WORDS or _is_number(last):
            name_toks, value_tok = toks[:-1], last
    elif cmd == "type":
        if last in TYPES or last in CLEAR_WORDS:
            name_toks, value_tok = toks[:-1], last
    dso = " ".join(name_toks).strip()
    out["dso"] = dso
    if not dso:
        out["error"] = "Usage: " + USAGE[cmd]
        return out
    if cmd == "hold":
        out["value"] = "waiting" if (value_tok in ("off",) or value_tok in CLEAR_WORDS) else "held"
        return out
    if value_tok is None:
        out["error"] = "%s: say what to set -- %s" % (dso, USAGE[cmd])
        return out
    if value_tok in CLEAR_WORDS:
        out["value"] = None
        return out
    if cmd == "moon":
        out["value"] = value_tok
    elif cmd == "type":
        out["value"] = value_tok
    elif cmd == "maxhours":
        h = float(value_tok)
        lo, hi = MAX_HOURS_RANGE
        if not (lo <= h <= hi):
            out["error"] = "maxhours %s: between %g and %g hours a night (`maxhours %s clear` lifts it)" % (
                value_tok, lo, hi, dso)
            return out
        out["value"] = h
    return out


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def describe(cmd: str, dso: str, value, row: Optional[dict] = None) -> str:
    """The reply after a write."""
    if cmd == "hold":
        return (f"{dso} held: the planner leaves it until `hold {dso} off`" if value == "held"
                else f"{dso} back to waiting: the planner may take it again")
    if value is None:
        return {"moon": f"{dso}: moon rule cleared (back to the default: avoid)",
                "maxhours": f"{dso}: no cap; it may take a main slot",
                "type": f"{dso}: type cleared; the catalogue decides"}[cmd]
    if cmd == "moon":
        why = {"any": "never held for the moon", "avoid": "hours held when the moon makes them worth too little",
               "dark": "held whenever the moon is up"}[value]
        return f"{dso}: moon {value} ({why})"
    if cmd == "maxhours":
        return f"{dso}: at most {value:g} h a night (a gap-fill block, never a main slot)"
    return f"{dso}: type {value} ({'L+RGB' if value == 'galaxy' else 'narrowband Ha/O-III/S-II'} from the next plan)"


# ------------------------------------------------------------ tonight's plan

def _squash(name) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _slot_for(rec: Optional[dict], dso: str) -> Optional[dict]:
    for s in ((rec or {}).get("plan") or {}).get("slots") or []:
        if _squash(s.get("dso")) == _squash(dso):
            return s
    return None


def replan_verdict(plan: Optional[dict], dso: str) -> Tuple[bool, str]:
    """Does a change to *dso*'s row re-run tonight's noon decision? Pure.

    *plan* is iris.client.get_plan(): {"noon": record, "sunset": record}, a
    record the NIGHT_PLAN note's data. -> (replan, text). 2026-11-20: the
    owner set IC 405's filters after noon; the sequence followed (the next
    generation reads the queue) but the noon "Filters tonight" line, the
    push and the site's board kept the advisor's O-III until the sunset
    record, and the sunset check, finding the file already right, said
    nothing. A target in tonight's plan is replanned so the plan record,
    the sequence and every reader of the record move together; after the
    sunset check has decided, the change is tomorrow's and the text says
    so; a target not in tonight's plan, no plan, or no conductor: silence,
    the next plan reads the row."""
    if not plan:
        return False, ""
    sunset = plan.get("sunset")
    if sunset:
        if _slot_for(sunset, dso):
            return False, ("%s is in tonight's plan, but the sunset check has already decided and the "
                           "sequence is written: this change applies from tomorrow's plan." % dso)
        return False, ""
    noon = plan.get("noon")
    if not noon or noon.get("outcome") != "good":
        return False, ""
    slot = _slot_for(noon, dso)
    if slot is None:
        return False, ""
    # REPLAN_REQUESTED is a transition from IDLE_DAY or ARMED only (iris/core/machine);
    # from SLOT_IMAGING it re-seats the slot, which no row change may do by itself.
    state = plan.get("state")
    if state and state not in REPLAN_STATES:
        return False, ("%s is in tonight's plan, but the conductor is in %s; the change applies to the "
                       "next plan (`replan` by hand if you mean now)." % (dso, state))
    return True, ("%s is in tonight's plan (%s-%s): replanning so the sequence, the filter line and the "
                  "site's board follow." % (dso, str(slot.get("start", ""))[11:16], str(slot.get("end", ""))[11:16]))
