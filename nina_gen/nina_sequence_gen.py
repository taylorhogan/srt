import json
import logging
import math
from pathlib import Path
from typing import Any, Optional


def _decompose_ra(ra_hours: float) -> tuple[int, int, float]:
    """Decompose decimal RA hours into (hours, minutes, seconds)."""
    h = int(ra_hours)
    remainder = (ra_hours - h) * 60
    m = int(remainder)
    s = (remainder - m) * 60
    return h, m, s


def _decompose_dec(dec_degrees: float) -> tuple[bool, int, int, float]:
    """Decompose decimal Dec degrees into (negative, degrees, minutes, seconds)."""
    negative = bool(dec_degrees < 0)
    abs_dec = abs(dec_degrees)
    d = int(abs_dec)
    remainder = (abs_dec - d) * 60
    m = int(remainder)
    s = (remainder - m) * 60
    return negative, d, m, s


_GALAXY_OTYPES = {"G", "SyG", "Sy1", "Sy2", "rG", "GrG", "GiC", "GiG", "BCG", "AGN", "QSO", "LIN", "Sy"}
_NEBULA_OTYPES = {"PN", "HII", "SNR", "RNe", "GNe", "MoC", "SFR", "HH", "Cld", "DN", "PoC", "ISM"}


def _queued_object_type(dso_name: str) -> Optional[str]:
    """An object type recorded on the queued instruction, if there is one.

    Positional targets are named by the requestor, not by a catalogue, so SIMBAD
    cannot classify them: "bubble" and "gravwav" both come back "unknown", and
    unknown means a broadband LRGB plan. On a planetary nebula that spends the
    night on the wrong filters. An explicit obj_type on the instruction is the
    escape hatch, and it is checked before the name lookup.
    """
    try:
        from control import instructions
        rec = instructions.get_instruction_by_dso(dso_name)
    except Exception:
        return None
    if not rec:
        return None
    t = str(rec.get("obj_type", "")).strip().lower()
    return t if t in ("galaxy", "nebula") else None


def _classify_object_type(dso_name: str) -> str:
    """Object type for the filter plan: 'galaxy', 'nebula', or 'unknown'.

    An explicit type on the queued instruction wins over the name lookup.
    """
    queued = _queued_object_type(dso_name)
    if queued:
        return queued
    try:
        from astroquery.simbad import Simbad
        custom = Simbad()
        custom.add_votable_fields("otype")
        result = custom.query_object(dso_name)
        if result is None or len(result) == 0:
            return "unknown"
        # astroquery renamed the column OTYPE -> otype (v0.4.8); accept either,
        # otherwise every target silently classifies "unknown" and nebulae get
        # a broadband LRGB plan instead of narrowband.
        otype_col = next(c for c in result.colnames if c.lower() == "otype")
        otype = str(result[otype_col][0])
        if otype in _GALAXY_OTYPES:
            return "galaxy"
        if otype in _NEBULA_OTYPES:
            return "nebula"
        return "unknown"
    except Exception:
        return "unknown"


def _collect_smart_exposures(obj: Any, result: Optional[list] = None) -> list:
    """Return all SmartExposure dicts in tree order."""
    if result is None:
        result = []
    if isinstance(obj, dict):
        if "NINA.Sequencer.SequenceItem.Imaging.SmartExposure" in obj.get("$type", ""):
            result.append(obj)
        for v in obj.values():
            _collect_smart_exposures(v, result)
    elif isinstance(obj, list):
        for item in obj:
            _collect_smart_exposures(item, result)
    return result


def _get_exposure_time(se: dict) -> float:
    """Read ExposureTime (seconds) from the TakeExposure inside a SmartExposure."""
    for item in se.get("Items", {}).get("$values", []):
        if "TakeExposure" in item.get("$type", ""):
            return float(item.get("ExposureTime", 300.0))
    return 300.0


def _filter_node(se: dict):
    """The inline FilterInfo dict for a SmartExposure, or None if it is a $ref.

    Block 0 of the template is a $ref, so its filter CANNOT be renamed -- an
    attempt silently leaves it on L. Only blocks with an inline node are usable
    for an explicit plan, which caps a plan at three filters on this template.
    """
    if isinstance(se, dict):
        if "SwitchFilter" in str(se.get("$type", "")):
            f = se.get("Filter")
            return f if isinstance(f, dict) and "_name" in f else None
        for v in se.values():
            got = _filter_node(v)
            if got is not None:
                return got
    elif isinstance(se, list):
        for v in se:
            got = _filter_node(v)
            if got is not None:
                return got
    return None


def _update_smart_exposure(se: dict, iterations: int, filter_name: Optional[str] = None,
                           position: Optional[int] = None) -> None:
    """Set LoopCondition iterations and optionally override the inline filter.

    Writes the wheel POSITION as well as the name when one is given. The name
    alone is not enough: the template's blocks carry positions 1/2/3, which are
    R/G/B, so renaming a block to Ha while leaving its position would point the
    wheel at red.
    """
    for cond in se.get("Conditions", {}).get("$values", []):
        if "LoopCondition" in cond.get("$type", ""):
            cond["Iterations"] = max(0, iterations)
            cond["CompletedIterations"] = 0

    if filter_name is not None:
        for item in se.get("Items", {}).get("$values", []):
            if "SwitchFilter" in item.get("$type", ""):
                f = item.get("Filter", {})
                if "_name" in f:          # inline FilterInfo only, skip $ref nodes
                    f["_name"] = filter_name
                    if position is not None:
                        f["_position"] = int(position)


def describe_sequence(path) -> list:
    """[(filter, iterations, exposure_seconds), ...] as actually written.

    Read back from the generated file rather than from the plan dict, so what
    gets reported is what N.I.N.A will run, not what we intended.
    """
    with open(path, encoding="utf-8-sig") as fh:
        seq = json.load(fh)
    rows = []
    for se in _collect_smart_exposures(seq):
        iterations = 0
        for cond in se.get("Conditions", {}).get("$values", []):
            if "LoopCondition" in cond.get("$type", ""):
                iterations = int(cond.get("Iterations", 0) or 0)
        if iterations <= 0:
            continue
        node = _filter_node(se)
        rows.append((node["_name"] if node else "?", iterations,
                     float(_get_exposure_time(se) or 0.0)))
    return rows


def _wheel() -> dict:
    from configs import config
    return config.data().get("nina", {}).get("filter_wheel", {}) or {}


def max_explicit_filters(sequence: Any) -> int:
    """How many filters an explicit plan can name for this template."""
    return sum(1 for se in _collect_smart_exposures(sequence)
               if _filter_node(se) is not None)


def _apply_explicit_plan(smart_exposures: list, plan: dict) -> dict[str, int]:
    """Honour a user-set {filter: exposures} plan.

    Counts are taken literally -- the user asked for that many frames, so this
    does not scale them to the hours available the way the automatic split does.
    Blocks that cannot be renamed, and any left over, are zeroed so nothing from
    the template runs by accident.
    """
    wheel = _wheel()
    usable = [se for se in smart_exposures if _filter_node(se) is not None]
    for se in smart_exposures:
        if se not in usable:
            _update_smart_exposure(se, iterations=0)

    applied: dict[str, int] = {}
    for se, (name, count) in zip(usable, plan.items()):
        _update_smart_exposure(se, iterations=int(count), filter_name=name,
                               position=wheel.get(name))
        applied[name] = int(count)
    for se in usable[len(applied):]:
        _update_smart_exposure(se, iterations=0)
    return applied


def _apply_filter_plan(sequence: Any, above_horizon_seconds: float, obj_type: str,
                       explicit: Optional[dict] = None) -> dict[str, int]:
    """
    Compute per-filter iteration counts, patch all SmartExposure blocks in the
    sequence in-place, and return the plan as {filter_name: iterations}.

    An explicit plan from the `filters` command wins outright; without one the
    automatic split by object type applies as before.
    """
    actual_seconds = above_horizon_seconds * 0.8
    smart_exposures = _collect_smart_exposures(sequence)

    # Template order is L, R, G, B (4 blocks)
    if len(smart_exposures) < 4:
        return {}

    if explicit:
        return _apply_explicit_plan(smart_exposures, explicit)

    se_l, se_r, se_g, se_b = smart_exposures[:4]
    exp_l = _get_exposure_time(se_l)
    exp_r = _get_exposure_time(se_r)
    exp_g = _get_exposure_time(se_g)
    exp_b = _get_exposure_time(se_b)

    if obj_type == "nebula":
        # Divide equally across Ha, O-III, S-II; no L frames.
        #
        # The names are the filter wheel's own -- "O-III" and "S-II", not "O"
        # and "S" as this used before. The wheel has no filter called "O" or
        # "S", and sh2-92 bears that out: 330 auto-sequenced frames, 137 Ha and
        # 193 O-III, and not one S-II. The position is written alongside the
        # name for the same reason (see _update_smart_exposure).
        wheel = _wheel()
        each = actual_seconds / 3.0
        plan = {
            "Ha":    math.floor(each / exp_r),
            "O-III": math.floor(each / exp_g),
            "S-II":  math.floor(each / exp_b),
        }
        _update_smart_exposure(se_l, iterations=0)
        _update_smart_exposure(se_r, iterations=plan["Ha"], filter_name="Ha",
                               position=wheel.get("Ha"))
        _update_smart_exposure(se_g, iterations=plan["O-III"], filter_name="O-III",
                               position=wheel.get("O-III"))
        _update_smart_exposure(se_b, iterations=plan["S-II"], filter_name="S-II",
                               position=wheel.get("S-II"))
        return plan
    else:
        # Galaxy / unknown: L = 50%, R+G+B = 50%/3 each
        l_secs = actual_seconds * 0.5
        rgb_secs = actual_seconds * 0.5 / 3.0
        plan = {
            "L": math.floor(l_secs  / exp_l),
            "R": math.floor(rgb_secs / exp_r),
            "G": math.floor(rgb_secs / exp_g),
            "B": math.floor(rgb_secs / exp_b),
        }
        _update_smart_exposure(se_l, iterations=plan["L"])
        _update_smart_exposure(se_r, iterations=plan["R"])
        _update_smart_exposure(se_g, iterations=plan["G"])
        _update_smart_exposure(se_b, iterations=plan["B"])
        return plan


def _need_plan(sequence: Any, dso_name: str, obj_type: str,
               above_horizon_seconds: float) -> Optional[dict]:
    """{filter: count} from measured convergence need (§2c), or None.

    Sits between the explicit `filters` plan (which wins outright) and the
    object-type default split (which applies when the DSO has no history).
    Weights come from fits_processing.filter_need; they are converted to
    exposure counts here — against the same 0.8 usable-night factor and the
    template block's exposure time the automatic split uses — and then handed
    to the EXPLICIT plan path, so need-weighting adds no new patching code.
    """
    try:
        from fits_processing import filter_need as _fn
        weights = _fn.filter_need_for_dso(dso_name, obj_type)
    except Exception:
        return None
    if not weights:
        # None = no history (fall back to type default). {} = every filter
        # converged — also fall back, but say so: the planner should not have
        # scheduled this target, and a silent zero-frame night would look
        # exactly like the bug this module replaced.
        if weights == {}:
            logging.getLogger(__name__).warning(
                "_need_plan: every %s filter for '%s' is converged; "
                "falling back to the default split", obj_type, dso_name)
        return None
    smart_exposures = [se for se in _collect_smart_exposures(sequence)
                       if _filter_node(se) is not None]
    if len(smart_exposures) < len(weights):
        return None                    # 2-block template: let the guard handle it
    exposure_s = _get_exposure_time(smart_exposures[0])
    counts = _fn.counts_from_weights(weights, above_horizon_seconds * 0.8,
                                     exposure_s)
    if not counts or not any(counts.values()):
        return None
    logging.getLogger(__name__).info(
        "need-weighted filter plan for %s (%s): %s", dso_name, obj_type, counts)
    return counts


def _walk_and_replace(obj: Any, dso_name: str, coords: dict) -> None:
    """Recursively walk the JSON structure and replace target name and coordinates in-place."""
    if isinstance(obj, dict):
        obj_type = obj.get("$type", "")

        if "NINA.Astrometry.InputCoordinates" in obj_type:
            obj["RAHours"] = coords["RAHours"]
            obj["RAMinutes"] = coords["RAMinutes"]
            obj["RASeconds"] = coords["RASeconds"]
            obj["NegativeDec"] = coords["NegativeDec"]
            obj["DecDegrees"] = coords["DecDegrees"]
            obj["DecMinutes"] = coords["DecMinutes"]
            obj["DecSeconds"] = coords["DecSeconds"]

        if "TargetName" in obj:
            obj["TargetName"] = dso_name

        if "NINA.Sequencer.Container.DeepSkyObjectContainer" in obj_type:
            obj["Name"] = dso_name

        # Update the horizon-wait Pushover message that references the target name
        if "SendToPushover" in obj_type and "Message" in obj:
            if "above horizon" in obj["Message"]:
                obj["Message"] = f"Waiting for {dso_name} to be above horizon and dark"

        for v in obj.values():
            _walk_and_replace(v, dso_name, coords)

    elif isinstance(obj, list):
        for item in obj:
            _walk_and_replace(item, dso_name, coords)


def _plan_for(container: Any, dso_name: str, seconds: Optional[float]) -> dict[str, int]:
    """Size the SmartExposure blocks under *container* for *dso_name*.

    §2c resolution order: explicit `filters` plan > need-weighted split
    (measured per-filter convergence) > object-type default split. An explicit
    plan applies even when the hours are unknown: its counts are absolute.
    *container* may be the whole sequence or one target's container.
    """
    explicit = None
    try:
        from control import instructions as _instr
        explicit = _instr.get_filter_plan(dso_name)
    except Exception:
        explicit = None
    if explicit:
        return _apply_filter_plan(container, seconds or 0.0, "explicit", explicit=explicit)
    if seconds is not None and seconds > 0:
        obj_type = _classify_object_type(dso_name)
        need = _need_plan(container, dso_name, obj_type, seconds)
        if need:
            return _apply_filter_plan(container, seconds, obj_type, explicit=need)
        return _apply_filter_plan(container, seconds, obj_type)
    return {}


# --------------------------------------------------------------------------- #
# Multi-slot nights: one target container per slot
# --------------------------------------------------------------------------- #

def _short_type(node: dict) -> str:
    return str(node.get("$type", "")).split(",")[0].split(".")[-1]


def _find_target_area(sequence: Any) -> Optional[dict]:
    if isinstance(sequence, dict):
        if _short_type(sequence) == "TargetAreaContainer":
            return sequence
        for v in sequence.values():
            got = _find_target_area(v)
            if got is not None:
                return got
    elif isinstance(sequence, list):
        for v in sequence:
            got = _find_target_area(v)
            if got is not None:
                return got
    return None


def _items_of(container: dict) -> list:
    items = container.get("Items")
    return items.get("$values", []) if isinstance(items, dict) else (items or [])


def _max_id(node: Any, best: int = 0) -> int:
    if isinstance(node, dict):
        v = node.get("$id")
        if isinstance(v, str) and v.isdigit():
            best = max(best, int(v))
        for c in node.values():
            best = _max_id(c, best)
    elif isinstance(node, list):
        for c in node:
            best = _max_id(c, best)
    return best


def _clone_with_fresh_ids(node: Any, next_id: list) -> Any:
    """Deep-copy *node*, giving every $id inside it a fresh number and
    remapping every $ref that pointed at one of those ids. $refs to objects
    outside the copy (shared filter definitions, the parent area) are kept:
    N.I.N.A's JSON uses Newtonsoft reference tracking, and a duplicated $id
    would make the loader silently alias two objects into one.
    """
    import copy
    clone = copy.deepcopy(node)
    mapping: dict[str, str] = {}

    def assign(n):
        if isinstance(n, dict):
            if "$id" in n:
                new = str(next_id[0]); next_id[0] += 1
                mapping[n["$id"]] = new
                n["$id"] = new
            for c in n.values():
                assign(c)
        elif isinstance(n, list):
            for c in n:
                assign(c)

    def remap(n):
        if isinstance(n, dict):
            if "$ref" in n and n["$ref"] in mapping:
                n["$ref"] = mapping[n["$ref"]]
            for c in n.values():
                remap(c)
        elif isinstance(n, list):
            for c in n:
                remap(c)

    assign(clone)
    remap(clone)
    return clone


def _set_times(container: Any, start, end) -> None:
    """Point every WaitForTime under *container* at *start* and every
    TimeCondition at *end* (local wall-clock datetimes). The template's
    values are static (21:13 / 04:03); per-slot times are what make a slot
    hand over on time whatever its counts say."""
    if isinstance(container, dict):
        t = _short_type(container)
        if t == "WaitForTime" and start is not None:
            container["Hours"], container["Minutes"], container["Seconds"] = start.hour, start.minute, 0
            container["MinutesOffset"] = 0
        elif t == "TimeCondition" and end is not None:
            container["Hours"], container["Minutes"], container["Seconds"] = end.hour, end.minute, 0
            container["MinutesOffset"] = 0
        for v in container.values():
            _set_times(v, start, end)
    elif isinstance(container, list):
        for v in container:
            _set_times(v, start, end)


def _find_first(node: Any, short_type: str) -> Optional[dict]:
    if isinstance(node, dict):
        if _short_type(node) == short_type:
            return node
        for v in node.values():
            got = _find_first(v, short_type)
            if got is not None:
                return got
    elif isinstance(node, list):
        for v in node:
            got = _find_first(v, short_type)
            if got is not None:
                return got
    return None


def _script_item(prototype: dict, script: str, parent_id: str, next_id: list) -> dict:
    item = _clone_with_fresh_ids(prototype, next_id)
    item["Script"] = script
    item["Parent"] = {"$ref": parent_id}
    return item


def generate_slots_sequence(template_path: Path, slots: list, output_path: Path,
                            state_script: Optional[str] = None) -> list:
    """Write a sequence with one target container per slot.

    *slots* is a list of dicts: ``name, ra_hours, dec_degrees, seconds, start,
    end`` (start/end local datetimes, may be None). The template's single
    DeepSkyObjectContainer is cloned once per slot with fresh object ids, each
    clone patched for its target, timed for its window and sized for its
    hours -- the same planning as a one-slot night, applied per slot. One roof
    open, one prelude, one set of flats: only the main section changes.

    Every slot after the first gets two ExternalScript steps around its setup
    (*state_script*, i.e. scripts/set_imaging_state.bat): DONE_MAIN as the
    previous slot ends and IN_MAIN as this one begins imaging. That is the
    hand-over signal the conductor's shadow turns into NINA_SLOT_DONE /
    SLOT_STARTED, and it costs the run nothing.

    Returns the per-slot filter plans, in slot order.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        sequence = json.load(f)
    area = _find_target_area(sequence)
    if area is None:
        raise ValueError("template has no TargetAreaContainer")
    items = _items_of(area)
    idx = next((i for i, it in enumerate(items)
                if _short_type(it) == "DeepSkyObjectContainer"), None)
    if idx is None:
        raise ValueError("template has no DeepSkyObjectContainer to clone")
    proto = items[idx]
    script_proto = _find_first(sequence, "ExternalScript")
    next_id = [_max_id(sequence) + 1]

    clones, plans = [], []
    for k, slot in enumerate(slots):
        c = proto if k == 0 else _clone_with_fresh_ids(proto, next_id)
        ra_h, ra_m, ra_s = _decompose_ra(slot["ra_hours"])
        dec_neg, dec_d, dec_m, dec_s = _decompose_dec(slot["dec_degrees"])
        coords = {"RAHours": ra_h, "RAMinutes": ra_m, "RASeconds": round(ra_s, 5),
                  "NegativeDec": dec_neg, "DecDegrees": dec_d, "DecMinutes": dec_m,
                  "DecSeconds": round(dec_s, 5)}
        _walk_and_replace(c, slot["name"], coords)
        _set_times(c, slot.get("start"), slot.get("end"))
        plans.append(_plan_for(c, slot["name"], slot.get("seconds")))
        if k > 0 and state_script and script_proto is not None:
            setup = next((it for it in _items_of(c)
                          if _short_type(it) == "SequentialContainer"), None)
            if setup is not None:
                sitems = _items_of(setup)
                pid = setup.get("$id")
                sitems.insert(0, _script_item(script_proto,
                                              '"%s" DONE_MAIN' % state_script, pid, next_id))
                sitems.append(_script_item(script_proto,
                                           '"%s" IN_MAIN' % state_script, pid, next_id))
        clones.append(c)
    items[idx:idx + 1] = clones

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sequence, f, indent=2)
    return plans


def generate_sequence(
    template_path: Path,
    dso_name: str,
    ra_hours: float,
    dec_degrees: float,
    output_path: Path,
    above_horizon_seconds: Optional[float] = None,
) -> dict[str, int]:
    """
    Generate a NINA imaging sequence from a template with a new target.

    Args:
        template_path:          Path to the NINA JSON sequence template.
        dso_name:               Display name of the target (e.g. "M 51").
        ra_hours:               Right ascension in decimal hours (e.g. 13.4978).
        dec_degrees:            Declination in decimal degrees; negative for south (e.g. 47.195).
        output_path:            Destination path for the generated sequence file.
        above_horizon_seconds:  Seconds the DSO is above the horizon tonight. When provided,
                                filter iteration counts are derived automatically based on object type.

    Returns:
        Dict mapping filter name to iteration count (empty if above_horizon_seconds not given).
    """
    with open(template_path, "r", encoding="utf-8") as f:
        sequence = json.load(f)

    ra_h, ra_m, ra_s = _decompose_ra(ra_hours)
    dec_neg, dec_d, dec_m, dec_s = _decompose_dec(dec_degrees)

    coords = {
        "RAHours": ra_h,
        "RAMinutes": ra_m,
        "RASeconds": round(ra_s, 5),
        "NegativeDec": dec_neg,
        "DecDegrees": dec_d,
        "DecMinutes": dec_m,
        "DecSeconds": round(dec_s, 5),
    }

    _walk_and_replace(sequence, dso_name, coords)

    filter_plan = _plan_for(sequence, dso_name, above_horizon_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sequence, f, indent=2)

    return filter_plan


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        print("Usage: nina_sequence_gen.py <template> <dso_name> <ra_hours> <dec_degrees> <output>")
        sys.exit(1)

    generate_sequence(
        template_path=Path(sys.argv[1]),
        dso_name=sys.argv[2],
        ra_hours=float(sys.argv[3]),
        dec_degrees=float(sys.argv[4]),
        output_path=Path(sys.argv[5]),
    )
