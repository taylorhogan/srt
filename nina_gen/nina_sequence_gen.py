import json
import logging
import math
import os
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

    An explicit obj_type on the queued instruction wins. Then the imaging
    grid's own classifier (astro_dso_visibility.get_dso_type: its override
    table, its type cache, then SIMBAD), so this and the `tonight` table can
    never disagree. On 2026-10-15 the table called ic405, ngc2237 and m20
    nebulae while this function, with its own smaller SIMBAD map and no
    overrides, called them unknown and would have planned LRGB. A star
    cluster plans broadband ("unknown"), as before. Only when that classifier
    cannot be imported (no astropy) or answers "?" does the direct SIMBAD
    lookup below run.
    """
    queued = _queued_object_type(dso_name)
    if queued:
        return queued
    try:
        import importlib
        grid = importlib.import_module("iris_astronomy.astro_dso_visibility").get_dso_type(dso_name)
    except Exception:
        grid = "?"
    if grid == "G":
        return "galaxy"
    if grid == "N":
        return "nebula"
    if grid == "C":
        return "unknown"
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


def _set_exposure_time(se: dict, seconds: float) -> None:
    """Write *seconds* into the block's TakeExposure, keeping the dither cadence.

    The template dithers every AfterExposures frames at its own exposure; a
    shorter frame with the same count would dither 30x as often at 10 s.
    AfterExposures is rescaled so dithers stay the same number of seconds
    apart (at least every frame).
    """
    old = _get_exposure_time(se) or 300.0
    for item in se.get("Items", {}).get("$values", []):
        if "TakeExposure" in item.get("$type", ""):
            item["ExposureTime"] = float(seconds)
    for trig in se.get("Triggers", {}).get("$values", []):
        if "DitherAfterExposures" in str(trig.get("$type", "")):
            every = max(1, int(trig.get("AfterExposures", 1) or 1))
            trig["AfterExposures"] = max(1, int(round(every * old / float(seconds))))


def _exposures_for(dso_name: str) -> Optional[dict]:
    """The queue entry's {filter: seconds} override, or None."""
    try:
        from control import instructions as _instr
        return _instr.get_exposure_plan(dso_name)
    except Exception:
        return None


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


# The QHY wheel as cfg["nina"]["filter_wheel"] has it; used only when the
# config does not say, so a renamed block can never keep the template's slot.
DEFAULT_WHEEL = {"L": 0, "R": 1, "G": 2, "B": 3, "S-II": 4, "O-III": 5, "Ha": 6}


def _wheel() -> dict:
    try:
        from configs import config
        wheel = config.data().get("nina", {}).get("filter_wheel", {}) or {}
    except Exception:
        wheel = {}
    return wheel or dict(DEFAULT_WHEEL)


# --------------------------------------------------------------------------- #
# Per-filter camera gain
# --------------------------------------------------------------------------- #
#
# N.I.N.A has no gain per filter: gain and offset are fields of each exposure
# instruction, -1 meaning the camera profile's value. So "narrowband at gain
# 26" is written into the TakeExposure of every block that shoots Ha or O-III,
# and -1 into every other block, every time a sequence is generated. It has to
# be every block, every time: the blocks are renamed per plan, so a 26 that
# stayed on a block after it was renamed to L would shoot L at the wrong gain.
# cfg["nina"]["gain"] is {filter: [gain, offset]}; a filter not in it is -1/-1.
# The flats run from their own sequence (scripts/nina_flats.bat ->
# flat_sequence.json), where the same two instructions carry the same gain,
# and the stacker matches calibration to the lights' gain (stacker.select_gain).


def _gain_map() -> dict:
    """{filter: (gain, offset)} from cfg["nina"]["gain"]; empty when unset."""
    try:
        from configs import config
        raw = config.data().get("nina", {}).get("gain", {}) or {}
    except Exception:
        raw = {}
    out = {}
    for name, val in raw.items():
        try:
            gain, offset = (int(val[0]), int(val[1])) if isinstance(val, (list, tuple)) else (int(val), -1)
        except (TypeError, ValueError, IndexError):
            logging.getLogger(__name__).warning("nina.gain[%r] = %r is not [gain, offset]; ignored", name, val)
            continue
        out[str(name)] = (gain, offset)
    return out


def _block_filter_name(se: dict, ids: Optional[dict] = None) -> Optional[str]:
    """The filter a block shoots: its inline FilterInfo, or the one its $ref points at."""
    node = _filter_node(se)
    if node is not None:
        return node.get("_name")
    for item in se.get("Items", {}).get("$values", []):
        if "SwitchFilter" in str(item.get("$type", "")):
            f = item.get("Filter")
            if isinstance(f, dict) and "$ref" in f and ids:
                target = ids.get(str(f["$ref"]))
                if isinstance(target, dict):
                    return target.get("_name")
    return None


def _block_gain(se: dict) -> tuple[int, int]:
    """(Gain, Offset) as written in the block's TakeExposure; -1 where absent."""
    for item in se.get("Items", {}).get("$values", []):
        if "TakeExposure" in str(item.get("$type", "")):
            return int(item.get("Gain", -1)), int(item.get("Offset", -1))
    return -1, -1


def apply_filter_gain(container: Any, ids: Optional[dict] = None,
                      gain_map: Optional[dict] = None) -> dict:
    """Write each block's gain and offset from its filter. Returns {filter: (gain, offset)} written.

    Every block gets written, mapped filters at their gain and the rest at
    -1/-1 (the profile), so nothing a template or an earlier plan left in a
    block survives a rename. A block whose filter cannot be resolved is left
    at -1/-1 and logged.
    """
    gmap = _gain_map() if gain_map is None else gain_map
    written: dict = {}
    for se in _collect_smart_exposures(container):
        name = _block_filter_name(se, ids)
        gain, offset = gmap.get(name, (-1, -1)) if name else (-1, -1)
        if name is None and gmap:
            logging.getLogger(__name__).warning(
                "a SmartExposure block's filter could not be resolved; its gain stays at the profile's")
        for item in se.get("Items", {}).get("$values", []):
            if "TakeExposure" in str(item.get("$type", "")):
                item["Gain"] = int(gain)
                item["Offset"] = int(offset)
        if name:
            written[name] = (int(gain), int(offset))
    return written


DEFAULT_EFFICIENCY = 0.75


def _efficiency() -> float:
    """Fraction of a target's good hours that goes into exposures.

    The rest is centering, autofocus (per filter and per temperature step),
    filter changes, dither settles and the odd plate-solve. cfg["nina"]
    ["efficiency"], 0.75 by default; the split below sizes its counts to
    hours * efficiency so one pass fits the window. Before 2026-09-10 this
    was a literal 0.8 in two places, and the explicit `filters` plan ignored
    the hours entirely.
    """
    try:
        from configs import config
        e = float(config.data().get("nina", {}).get("efficiency", DEFAULT_EFFICIENCY))
    except Exception:
        e = DEFAULT_EFFICIENCY
    return e if 0.0 < e <= 1.0 else DEFAULT_EFFICIENCY


def max_explicit_filters(sequence: Any) -> int:
    """How many filters an explicit plan can name for this template."""
    return sum(1 for se in _collect_smart_exposures(sequence)
               if _filter_node(se) is not None)


DEFAULT_FRAME_OVERHEAD_S = 12.4


def _frame_overhead() -> float:
    """Seconds each frame costs beyond its exposure: download, save, dither.

    Measured 2026-10-20 from DATE-OBS gaps between consecutive same-filter
    frames on three targets: median 12.1-12.5 s, p90 13.3 s. Ignored until
    then, which cost 300 s frames ~4% but doubled the true time of 10 s ones:
    ngc7790's hour was sized as 180 L + 45 G, ~90 minutes of work. Every count
    below divides by exposure + this. cfg["nina"]["frame_overhead_s"]."""
    try:
        from configs import config      # this module imports config locally everywhere
        v = float(config.data().get("nina", {}).get("frame_overhead_s", DEFAULT_FRAME_OVERHEAD_S))
    except Exception:
        v = DEFAULT_FRAME_OVERHEAD_S
    return v if v >= 0 else DEFAULT_FRAME_OVERHEAD_S


def _max_seconds_for(dso_name: str) -> Optional[float]:
    """The queue entry's max_hours in seconds, or None. A capped target's
    frames are sized to the cap, not to its whole slot: slots are planned in
    whole hours, so max_hours 0.5 got a full hour's frames (2026-10-20). The
    block then finishes early and the next slot starts sooner."""
    try:
        from control import instructions as _instr
        rec = _instr.get_instruction_by_dso(dso_name) or {}
        mh = rec.get("max_hours")
        return float(mh) * 3600.0 if mh else None
    except Exception:
        return None


def _apply_explicit_plan(smart_exposures: list, plan: dict,
                         usable_seconds: Optional[float] = None,
                         exposures: Optional[dict] = None) -> dict[str, int]:
    """Honour a user-set {filter: exposures} plan.

    With *usable_seconds* (the window's good hours times the efficiency) the
    plan is a RATIO: the seconds are split between the filters in proportion
    to the counts and each block gets the whole frames that fit, at least one
    for any filter with a share. Without it the counts are literal -- a
    hand-made sequence for a target that is not in tonight's ranking.

    The ratio reading is what the operator means. ngc7380's plan of 36/24/12
    at 300 s is six hours; the blocks run in sequence, so in a three-hour
    window the old literal reading spent all of it on Ha and never reached
    O-III or S-II (2026-09-10). Blocks that cannot be renamed, and any left
    over, are zeroed so nothing from the template runs by accident.

    *exposures* ({filter: seconds}, the queue entry's exposure_s) replaces a
    block's exposure time BEFORE its count is sized, so the hours buy the
    right number of short frames.
    """
    wheel = _wheel()
    usable = [se for se in smart_exposures if _filter_node(se) is not None]
    for se in smart_exposures:
        if se not in usable:
            _update_smart_exposure(se, iterations=0)

    total = float(sum(max(0, int(v)) for v in plan.values()))
    scale = usable_seconds is not None and usable_seconds > 0 and total > 0
    applied: dict[str, int] = {}
    for se, (name, count) in zip(usable, plan.items()):
        count = max(0, int(count))
        if exposures and name in exposures:
            _set_exposure_time(se, exposures[name])
        if scale and count > 0:
            exp_s = _get_exposure_time(se) or 1.0
            count = max(1, int(usable_seconds * count / total // (exp_s + _frame_overhead())))
        _update_smart_exposure(se, iterations=count, filter_name=name,
                               position=wheel.get(name))
        applied[name] = count
    for se in usable[len(applied):]:
        _update_smart_exposure(se, iterations=0)
    return applied


def _apply_filter_plan(sequence: Any, above_horizon_seconds: float, obj_type: str,
                       explicit: Optional[dict] = None,
                       exposures: Optional[dict] = None) -> dict[str, int]:
    """
    Compute per-filter iteration counts, patch all SmartExposure blocks in the
    sequence in-place, and return the plan as {filter_name: iterations}.

    An explicit plan from the `filters` command wins outright, read as a
    ratio over the hours; without one the automatic split by object type
    applies. Either way the counts consume hours * cfg["nina"]["efficiency"]
    (_efficiency) so one pass fits the target's window.
    """
    actual_seconds = above_horizon_seconds * _efficiency()
    smart_exposures = _collect_smart_exposures(sequence)

    # Template order is L, R, G, B (4 blocks)
    if len(smart_exposures) < 4:
        return {}

    if explicit:
        return _apply_explicit_plan(
            smart_exposures, explicit,
            usable_seconds=actual_seconds if above_horizon_seconds > 0 else None,
            exposures=exposures)

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
        oh = _frame_overhead()
        plan = {
            "Ha":    math.floor(each / (exp_r + oh)),
            "O-III": math.floor(each / (exp_g + oh)),
            "S-II":  math.floor(each / (exp_b + oh)),
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
        oh = _frame_overhead()
        plan = {
            "L": math.floor(l_secs  / (exp_l + oh)),
            "R": math.floor(rgb_secs / (exp_r + oh)),
            "G": math.floor(rgb_secs / (exp_g + oh)),
            "B": math.floor(rgb_secs / (exp_b + oh)),
        }
        # Every block named, with its wheel position. Until 2026-10-14 this
        # branch set only the counts and trusted the template's names, and
        # cdk_full_sequence_4block.json (2026-08-31) carries Ha / O-III /
        # O-III / O-III: ngc6946 and ngc1961, the first targets in weeks
        # without a `filters` plan, were announced as LRGB and shot as Ha and
        # O-III. _verify_written now refuses any such mismatch.
        wheel = _wheel()
        for se, f in ((se_l, "L"), (se_r, "R"), (se_g, "G"), (se_b, "B")):
            _update_smart_exposure(se, iterations=plan[f], filter_name=f, position=wheel.get(f))
        return plan


def _need_plan(sequence: Any, dso_name: str, obj_type: str,
               above_horizon_seconds: float) -> Optional[dict]:
    """{filter: count} from measured convergence need (§2c), or None.

    Sits between the explicit `filters` plan (which wins outright) and the
    object-type default split (which applies when the DSO has no history).
    Weights come from fits_processing.filter_need; they are converted to
    exposure counts here — against the same efficiency factor and the
    template block's exposure time the automatic split uses — and then handed
    to the EXPLICIT plan path (which, given the hours, scales them again to
    the same total: a no-op on already-fitted counts), so need-weighting adds
    no new patching code.
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
    counts = _fn.counts_from_weights(weights, above_horizon_seconds * _efficiency(),
                                     exposure_s + _frame_overhead())
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
    (measured per-filter convergence) > object-type default split. Every
    path sizes its counts to hours * efficiency (_efficiency). An explicit
    plan applies even when the hours are unknown: then its counts are
    literal, otherwise they are the ratio. *container* may be the whole
    sequence or one target's container.
    """
    explicit = None
    try:
        from control import instructions as _instr
        explicit = _instr.get_filter_plan(dso_name)
    except Exception:
        explicit = None
    exposures = _exposures_for(dso_name)
    if exposures and not explicit:
        # The automatic splits size counts against the template's names,
        # which are not the plan's; an override there would land on
        # whichever block happened to carry the name. Explicit only.
        logging.getLogger(__name__).warning(
            "%s: exposure_s %s ignored -- it needs an explicit filter_plan", dso_name, exposures)
        exposures = None
    cap = _max_seconds_for(dso_name)
    if cap and seconds and seconds > cap:
        logging.getLogger(__name__).info(
            "%s: frames sized to max_hours (%.0f min), not the %.0f-min slot",
            dso_name, cap / 60.0, seconds / 60.0)
        seconds = cap
    if explicit:
        plan = _apply_filter_plan(container, seconds or 0.0, "explicit", explicit=explicit,
                                  exposures=exposures)
    elif seconds is not None and seconds > 0:
        obj_type = _classify_object_type(dso_name)
        need = _need_plan(container, dso_name, obj_type, seconds)
        if need:
            plan = _apply_filter_plan(container, seconds, obj_type, explicit=need)
        else:
            plan = _apply_filter_plan(container, seconds, obj_type)
    else:
        return {}
    _verify_written(container, plan, dso_name, exposures)
    return plan


def _expected_exposures(dso_name: str) -> Optional[dict]:
    """The exposure override the read-back must find, or None (explicit plans only,
    as _plan_for applies it)."""
    exposures = _exposures_for(dso_name)
    if not exposures:
        return None
    try:
        from control import instructions as _instr
        return exposures if _instr.get_filter_plan(dso_name) else None
    except Exception:
        return None


def _iterations(se: dict) -> int:
    for cond in se.get("Conditions", {}).get("$values", []):
        if "LoopCondition" in cond.get("$type", ""):
            return int(cond.get("Iterations", 0) or 0)
    return 0


def written_plan(container: Any) -> dict:
    """{filter: frames} as written under *container*: what N.I.N.A will shoot."""
    out: dict = {}
    for se in _collect_smart_exposures(container):
        n = _iterations(se)
        if n <= 0:
            continue
        node = _filter_node(se)
        name = node["_name"] if node else "?"
        out[name] = out.get(name, 0) + n
    return out


def _verify_written(container: Any, plan: dict, dso_name: str,
                    exposures: Optional[dict] = None) -> None:
    """Refuse a target whose blocks do not say what its plan says.

    The plan is what gets announced (the noon report, the logs); the blocks
    are what N.I.N.A shoots. 2026-10-14 they differed -- LRGB announced, Ha
    and O-III shot -- and nothing compared them. Also checks each block's
    wheel position against cfg["nina"]["filter_wheel"], since a name with
    the wrong position points the wheel at another filter."""
    if not plan:
        return          # no plan applied: the template's own blocks, nothing announced to compare
    from nina_gen import sequence_lint
    want = {f: int(n) for f, n in plan.items() if int(n) > 0}
    got = written_plan(container)
    problems = []
    if got != want:
        problems.append("%s: the plan is %s but the exposure blocks are written as %s"
                        % (dso_name, want, got))
    wheel = _wheel()
    gmap = _gain_map()
    for se in _collect_smart_exposures(container):
        node = _filter_node(se)
        if (_iterations(se) > 0 and node and node["_name"] in wheel
                and node.get("_position") is not None
                and int(node["_position"]) != int(wheel[node["_name"]])):
            problems.append("%s: block %s is at wheel position %s, but %s is position %s"
                            % (dso_name, node["_name"], node["_position"],
                               node["_name"], wheel[node["_name"]]))
        if (exposures and _iterations(se) > 0 and node and node["_name"] in exposures
                and abs(_get_exposure_time(se) - float(exposures[node["_name"]])) > 1e-6):
            problems.append("%s: block %s exposes %g s, but the plan says %g s"
                            % (dso_name, node["_name"], _get_exposure_time(se),
                               float(exposures[node["_name"]])))
        if _iterations(se) > 0 and node:
            want_gain = gmap.get(node["_name"], (-1, -1))
            if _block_gain(se) != want_gain:
                problems.append("%s: block %s is written at gain/offset %s, but nina.gain says %s"
                                % (dso_name, node["_name"], _block_gain(se), want_gain))
    if problems:
        logging.getLogger(__name__).error("REFUSING the plan for %s:\n  %s",
                                          dso_name, "\n  ".join(problems))
        raise sequence_lint.SequenceLintError(problems)


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


# Object types N.I.N.A hands back as ONE instance however many times they
# appear: the date-time providers behind WaitForTime / TimeCondition
# (TimeProvider, NauticalDuskProvider, DawnProvider, ...). Newtonsoft's
# reference resolver refuses to register the same instance under a second
# $id -- "A different Id has already been assigned for value 'NINA.Sequencer.
# Utility.DateTimeProvider.NauticalDuskProvider'" was the load failure on
# 2026-09-08 -- so after all edits, _dedupe_singletons keeps the FIRST
# occurrence of each provider type as the definition and turns every later
# one into a $ref to it.
_SINGLETON_TYPE_MARKERS = ("DateTimeProvider",)
# The fixed-clock provider: with it, a WaitForTime / TimeCondition's Hours,
# Minutes and Seconds ARE the time. With any other provider (nautical dusk,
# dawn, ...) N.I.N.A computes the time and those fields are display only --
# which is why the first two-slot file loaded with both targets ending "at
# astronomical dawn" whatever the JSON said.
TIME_PROVIDER_TYPE = "NINA.Sequencer.Utility.DateTimeProvider.TimeProvider, NINA.Sequencer"


def _dedupe_singletons(sequence: Any) -> None:
    """First occurrence of each singleton type defines it; later ones $ref it."""
    first: dict[str, str] = {}

    def walk(n):
        if isinstance(n, dict):
            t = str(n.get("$type", ""))
            if "$id" in n and any(m in t for m in _SINGLETON_TYPE_MARKERS):
                if t in first:
                    ref = {"$ref": first[t]}
                    n.clear()
                    n.update(ref)
                    return
                first[t] = n["$id"]
            for c in list(n.values()):
                walk(c)
        elif isinstance(n, list):
            for c in n:
                walk(c)

    walk(sequence)


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


def _set_hard_end(container: Any, end, next_id: list) -> None:
    """Make every TimeCondition under *container* end at *end* (local
    wall-clock) by switching its provider to the fixed Time provider.

    Only the END is pinned, on EVERY slot. The WaitForTime stays the
    template's nautical dusk: for the first slot that is the dark start
    anyway, and for a later slot it has already passed, so the slot begins
    the moment the one before it ends -- early if that one ran out of counts
    early -- and "wait until above horizon" covers the rise.

    The end of a slot is the planner's window end (control/slot_plan): the
    hour after its last GOOD hour, where good = above the treeline AND the
    forecast accepts that hour AND it is inside the dark hours. So it is the
    earliest of horizon, dawn and weather. Until 2026-09-10 the last slot kept
    the template's condition (astronomical dawn) and the imaging container
    looped past its planned hours: on 09-10 the plan said ngc7380 22:00-01:00
    with 100% cloud forecast from 01:00, and the sequence would have kept the
    roof open under that overcast until the target set at 04:00. The counts
    inside the container size ONE pass; only this condition stops the next.
    """
    if isinstance(container, dict):
        if _short_type(container) == "TimeCondition" and end is not None:
            container["Hours"], container["Minutes"], container["Seconds"] = end.hour, end.minute, 0
            container["MinutesOffset"] = 0
            container["SelectedProvider"] = {"$id": str(next_id[0]), "$type": TIME_PROVIDER_TYPE}
            next_id[0] += 1
        for v in container.values():
            _set_hard_end(v, end, next_id)
    elif isinstance(container, list):
        for v in container:
            _set_hard_end(v, end, next_id)


TIME_SPAN_TYPE = "NINA.Sequencer.Conditions.TimeSpanCondition, NINA.Sequencer"


def _add_cap_span(container: Any, seconds: float, next_id: list) -> bool:
    """Beside the slot's TimeCondition, add a TimeSpanCondition of *seconds*.

    A capped target (max_hours) used to end only at a clock time, start + cap.
    The first slot does not wait for its planned start -- it begins when the
    sequence does -- so on 2026-10-25 ngc7790, planned 21:00 with a 30-min
    cap, imaged 19:43-21:28: five passes, nine autofocus runs. N.I.N.A's
    TimeSpanCondition ("loop for a time span", NINA.Sequencer.dll) counts from
    when the container starts, and a loop runs only while every condition
    holds, so the slot now ends at whichever comes first: cap after its real
    start, or the clock end. Same Hours/Minutes/Seconds fields as
    TimeCondition. Returns False when there was no TimeCondition to sit by.
    """
    found = []

    def walk(o):
        if isinstance(o, dict):
            conds = o.get("Conditions")
            if isinstance(conds, dict) and any(_short_type(x) == "TimeCondition"
                                               for x in conds.get("$values", [])):
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(container)
    if not found:
        return False
    total = int(round(seconds))
    for loop in found:
        vals = loop["Conditions"]["$values"]
        vals[:] = [x for x in vals if _short_type(x) != "TimeSpanCondition"]
        vals.append({"$id": str(next_id[0]), "$type": TIME_SPAN_TYPE,
                     "Hours": total // 3600, "Minutes": (total % 3600) // 60, "Seconds": total % 60,
                     "Parent": {"$ref": loop.get("$id")}})
        next_id[0] += 1
    return True


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


CENTER_AND_ROTATE_TYPE = "NINA.Sequencer.SequenceItem.Platesolving.CenterAndRotate, NINA.Sequencer"


def _apply_rotation(container: Any, degrees) -> bool:
    """Give *container*'s target a position angle and rotate to it.

    Sets InputTarget.PositionAngle and turns the setup's Center into Center
    And Rotate (same node, retyped, PositionAngle added), which N.I.N.A
    inherits from the target. The start sequence still parks the rotator at
    mechanical 0, so a target without a rotation images at 0 as before. True
    if a Center was found to convert.
    """
    if degrees is None:
        return False
    deg = float(degrees) % 360.0
    done = False

    def walk(n):
        nonlocal done
        if isinstance(n, dict):
            t = _short_type(n)
            if t == "InputTarget":
                n["PositionAngle"] = deg
            elif t == "Center":
                n["$type"] = CENTER_AND_ROTATE_TYPE
                n["PositionAngle"] = deg
                done = True
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(container)
    return done


def _rotation_for(dso_name: str):
    try:
        from control import instructions as _instr
        return _instr.get_rotation(dso_name)
    except Exception:
        return None


def _simple_item(type_name: str, parent_id: str, next_id: list) -> dict:
    """A bare sequence item (ParkScope / UnparkScope): no settings of its own."""
    node = {"$id": str(next_id[0]), "$type": type_name,
            "Parent": {"$ref": parent_id}, "ErrorBehavior": 0, "Attempts": 1}
    next_id[0] += 1
    return node


def horizon_rise(name: str, ra_hours: float, dec_degrees: float, after):
    """When *name* next clears the observatory horizon at/after *after*, or None.

    Uses the planner's own horizon (configs/my.hrz, the same altitude-vs-
    azimuth test astro_dso_visibility ranks the night with), so "above the
    trees" means here exactly what it means when the night is ranked. None on
    any failure -- an unanswerable sky must not decide whether to park.

    Until 2026-10-09 this went through find_alt_az_horizon_times, which also
    requires the sample to fall in "tonight's" dark window -- and for any
    *after* past about 23:00 that function takes the NEXT evening's window,
    so every post-midnight gap came back as "never rises" and no slot after
    midnight was ever parked across (ngc2146 would have tracked ngc7320 into
    the western trees for three hours). Darkness is not this function's
    question: a later slot already follows a WaitForTime for dusk.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
        import numpy as np
        from configs import config
        from iris_astronomy import astro_dso_visibility as av

        loc = config.data()["location"]
        site = EarthLocation.from_geodetic(loc["longitude"] * u.deg, loc["latitude"] * u.deg,
                                           loc["elevation"] * u.m)
        target = SkyCoord(ra=ra_hours * 15.0 * u.deg, dec=dec_degrees * u.deg)
        # Sample from `after` forward: the first sample above the horizon is
        # the rise we care about, and if it is already up that is sample one.
        grid = Time(after) + np.linspace(0, 10, 121) * u.hour
        aa = target.transform_to(AltAz(obstime=grid, location=site))
        az_h, al_h = av.map_az_to_horizon()
        for t, alt, az in zip(grid, aa.alt.deg, aa.az.deg):
            if alt >= av.get_horizon_from_azimuth(float(az) % 360.0, az_h, al_h):
                return t.to_datetime(timezone=after.tzinfo) if getattr(after, "tzinfo", None) else t.to_datetime()
        return None
    except Exception:  # noqa: BLE001 -- never let the sky break generation
        logging.getLogger(__name__).warning("horizon rise for %s unavailable", name, exc_info=True)
        return None


def park_across_gap(setup: dict, next_id: list) -> bool:
    """Park before this slot's wait and unpark after it. True when inserted.

    The park goes FIRST, so it happens the moment the previous slot's hard end
    fires, and the unpark immediately before the first thing that needs the
    mount -- whichever of Tracking / Center / CenterAndRotate / SwitchFilter
    comes first. Everything between them (the pushover, the WaitForTime, the
    WaitUntilAboveHorizon) is waiting, which a parked mount does better.
    """
    items = _items_of(setup)
    if not items:
        return False
    pid = setup.get("$id")
    needs_mount = ("Tracking", "Center", "CenterAndRotate", "SwitchFilter",
                   "SlewScopeToRaDec", "RunAutofocus")
    at = next((i for i, it in enumerate(items)
               if _short_type(it) in needs_mount), None)
    if at is None:
        return False
    items.insert(at, _simple_item(UNPARK_TYPE, pid, next_id))
    items.insert(0, _simple_item(PARK_TYPE, pid, next_id))
    return True


def _script_item(prototype: dict, script: str, parent_id: str, next_id: list) -> dict:
    item = _clone_with_fresh_ids(prototype, next_id)
    item["Script"] = script
    item["Parent"] = {"$ref": parent_id}
    return item


# --------------------------------------------------------------------------- #
# Focus seeds: a MoveFocuserByTemperature before every SmartExposure block
# --------------------------------------------------------------------------- #

FOCUS_SEED_TYPE = "NINA.Sequencer.SequenceItem.Focuser.MoveFocuserByTemperature, NINA.Sequencer"
PARK_TYPE = "NINA.Sequencer.SequenceItem.Telescope.ParkScope, NINA.Sequencer"
UNPARK_TYPE = "NINA.Sequencer.SequenceItem.Telescope.UnparkScope, NINA.Sequencer"

# A wait between slots longer than this gets a park across it. Below it the
# park is not worth two slews: the measured hand-over on a back-to-back night
# is 14 minutes (journal, 2026-10-05..08) and every one of those minutes is
# slewing, plate-solving or focusing -- the mount is working, not idling.
#
# Above it the mount would otherwise sit TRACKING THE PREVIOUS TARGET for the
# whole wait, because the next slot only issues Tracking and Center when it
# actually starts. It follows a dead field west with the roof open, and the
# primary -- an open truss -- stares at the sky radiating to an effective
# temperature far below ambient, which is how a mirror reaches dew point on a
# humid night (2026-09-29: 84% RH, under 3 C of dew-point spread). Parked at
# the horizon it looks at the tree line instead, which is AT ambient.
#
# THE WAIT IS NOT THE PLANNER'S GAP. A later slot's WaitForTime is the
# template's nautical dusk, long past by then, so the slot starts the moment
# the one before it ends and WaitUntilAboveHorizon is the only thing that
# holds it (see _set_hard_end). On 2026-10-07 the plan read ngc7320 20:00-01:00
# then ngc2146 03:00-06:00, a two-hour gap on paper -- and the journal shows
# the hand-over took 14 minutes, because ngc2146 was already above the trees.
# Parking on the planned gap would have cost two slews for nothing. So the
# wait is measured from when the next target actually clears the observatory
# horizon, with the same function the planner ranks by.
GAP_PARK_MIN_S = 20 * 60


def _focus_model():
    """The per-filter temperature model, refreshed from N.I.N.A's autofocus
    reports, or None when seeding is off / no model exists."""
    try:
        from configs import config
        nina = config.data().get("nina", {}) or {}
    except Exception:
        nina = {}
    if not nina.get("focus_seed", True):
        return None
    try:
        from fits_processing import focus_model as fm
        reports = os.path.expandvars(str(nina.get("autofocus_reports") or fm.DEFAULT_REPORTS_DIR))
        return fm.refresh(reports, str(nina.get("focus_model") or fm.DEFAULT_MODEL_PATH))
    except Exception as e:  # never let the seed break sequence generation
        logging.getLogger(__name__).warning("focus model unavailable: %s", e)
        return None


def _iterations(se: dict) -> Optional[int]:
    for cond in (se.get("Conditions") or {}).get("$values", []) if isinstance(se.get("Conditions"), dict) else []:
        if "LoopCondition" in str(cond.get("$type", "")):
            try:
                return int(cond.get("Iterations", 0))
            except (TypeError, ValueError):
                return None
    return None


def _resolve_filter_name(se: dict, ids: dict) -> Optional[str]:
    """The filter a SmartExposure will switch to, following a $ref if the
    block shares the template's filter definition."""
    for item in _items_of(se):
        if "SwitchFilter" in str(item.get("$type", "")):
            f = item.get("Filter")
            if isinstance(f, dict) and "$ref" in f:
                f = ids.get(f["$ref"])
            if isinstance(f, dict):
                return f.get("_name")
    return None


def _index_ids(node: Any, ids: Optional[dict] = None) -> dict:
    if ids is None:
        ids = {}
    if isinstance(node, dict):
        if "$id" in node:
            ids[node["$id"]] = node
        for v in node.values():
            _index_ids(v, ids)
    elif isinstance(node, list):
        for v in node:
            _index_ids(v, ids)
    return ids


# There is NO focuser move inside a SmartExposure. N.I.N.A's SmartExposure is
# a fixed-layout container (SwitchFilter, TakeExposure, its own loop and
# dither) and SmartExposure.Validate() indexes those by position: an extra
# item between them throws NullReferenceException on every validation pass
# and the block is SKIPPED -- no filter switch, no subs, the container done
# in five seconds. 2026-09-14: "temperature tracking between subs" did exactly
# that to every block; the night was relaunched by hand on a sequence with
# the in-block moves stripped out. Seeds (below) sit BEFORE the block, which
# N.I.N.A accepts; anything per-sub needs a different container shape.


def seed_focus(container: Any, model: Optional[dict], next_id: list,
               ids: Optional[dict] = None) -> dict[str, tuple[float, float]]:
    """Insert a MoveFocuserByTemperature before every SmartExposure under
    *container* whose filter the model knows. Returns {filter: (slope,
    intercept)} for the blocks that got one.

    Absolute mode: N.I.N.A moves to slope * T + intercept with T read from
    the focuser at run time, so the seed follows the night's temperature
    rather than a forecast made at noon. It runs BEFORE the block's own
    SwitchFilter, so the autofocus-after-filter-change trigger that fires
    inside the block starts its sweep centred on the seed -- and if that
    run fails, the position N.I.N.A restores is the seed, not the previous
    filter's focus (the 2026-09-11 first-O-III-light failure). Blocks whose
    filter has no model are left untouched. Call AFTER the filter plan has
    renamed the blocks. The inside of a SmartExposure is never touched (see
    the note above).
    """
    if not model:
        return {}
    from fits_processing import focus_model as fm
    if ids is None:
        ids = _index_ids(container)
    seeded: dict[str, tuple[float, float]] = {}

    def walk(n):
        if isinstance(n, dict):
            vals = _items_of(n)
            if vals and any(isinstance(it, dict) and _short_type(it) == "SmartExposure"
                            for it in vals):
                out = []
                for it in vals:
                    if isinstance(it, dict) and _short_type(it) == "SmartExposure":
                        name = _resolve_filter_name(it, ids)
                        si = fm.seed_for(model, name) if name else None
                        if _iterations(it) == 0:
                            si = None    # a zeroed spare block: no move for nothing
                        if si is not None:
                            out.append({"$id": str(next_id[0]), "$type": FOCUS_SEED_TYPE,
                                        "Slope": round(si[0], 3), "Intercept": round(si[1], 1),
                                        "Absolute": True, "Parent": {"$ref": n.get("$id")},
                                        "ErrorBehavior": 0, "Attempts": 1})
                            next_id[0] += 1
                            seeded[name] = si
                    out.append(it)
                vals[:] = out
            for k, v in n.items():
                if k != "Parent":
                    walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(container)
    if seeded:
        logging.getLogger(__name__).info(
            "Focus seeds: " + "  ".join("%s %+.0f*T%+.0f" % (f, s, i)
                                        for f, (s, i) in seeded.items()))
    return seeded


def _refuse_if_broken(sequence: Any, template: Any, output_path: Path,
                      lint_scripts: bool) -> None:
    """Run sequence_lint over the finished sequence and raise rather than
    write a file N.I.N.A would skip parts of. The night of 2026-09-14 is why:
    every exposure block silently skipped, nothing said until a target
    finished with no subs. *lint_scripts* checks that every ExternalScript's
    program exists -- true on the observatory, off for unit tests whose
    templates name made-up paths."""
    from nina_gen import sequence_lint
    problems = sequence_lint.lint(sequence, template=template, check_scripts=lint_scripts)
    if problems:
        logging.getLogger(__name__).error(
            "REFUSING to write %s: %d lint problem(s):\n  %s",
            output_path, len(problems), "\n  ".join(problems))
        raise sequence_lint.SequenceLintError(problems)


SWITCH_FILTER_TYPE = "NINA.Sequencer.SequenceItem.FilterWheel.SwitchFilter, NINA.Sequencer"
FILTER_INFO_TYPE = "NINA.Core.Model.Equipment.FilterInfo, NINA.Core"
# The filter every slot's plate solve runs through. Broadband, and position 0
# on this wheel.
CENTER_FILTER = "L"


def _filter_info(sequence: Any, name: str) -> Optional[dict]:
    """The template's own FilterInfo for *name*: the first SwitchFilter that
    names it, following a $ref. None if the template never switches to it."""
    ids = _index_ids(sequence)

    def walk(n):
        if isinstance(n, dict):
            if _short_type(n) == "SwitchFilter":
                f = n.get("Filter")
                if isinstance(f, dict) and "$ref" in f:
                    f = ids.get(f["$ref"])
                if isinstance(f, dict) and f.get("_name") == name:
                    return f
            for v in n.values():
                got = walk(v)
                if got is not None:
                    return got
        elif isinstance(n, list):
            for v in n:
                got = walk(v)
                if got is not None:
                    return got
        return None

    return walk(sequence)


def ensure_center_filter(container: Any, sequence: Any, next_id: list,
                         name: str = CENTER_FILTER) -> int:
    """Put a SwitchFilter *name* immediately before every Center /
    CenterAndRotate under *container* that does not already follow one.
    Returns how many were inserted.

    2026-09-15 03:00: the second slot's centering ran through the H-alpha
    filter the previous block left in the wheel, at the H-alpha focus, and
    ASTAP failed eight solves in fifteen minutes; the run was rescued by
    moving the wheel to L over N.I.N.A's API. The first slot never saw it
    because the prelude leaves L in place. Now every slot's plate solve gets
    broadband light regardless of what the block before it was doing. The
    FilterInfo is the template's own for *name* (fresh ids, so Newtonsoft's
    reference tracking sees a new object, not an alias), or a minimal one
    when the template never names it.
    """
    proto = _filter_info(sequence, name)
    inserted = 0

    def walk(n):
        nonlocal inserted
        if isinstance(n, dict):
            vals = _items_of(n)
            if vals:
                out = []
                for it in vals:
                    if (isinstance(it, dict)
                            and _short_type(it) in ("Center", "CenterAndRotate")
                            and not (out and isinstance(out[-1], dict)
                                     and _short_type(out[-1]) == "SwitchFilter")):
                        if proto is not None:
                            finfo = _clone_with_fresh_ids(proto, next_id)
                        else:
                            finfo = {"$id": str(next_id[0]), "$type": FILTER_INFO_TYPE,
                                     "_name": name, "_position": 0}
                            next_id[0] += 1
                        out.append({"$id": str(next_id[0]), "$type": SWITCH_FILTER_TYPE,
                                    "Filter": finfo, "Parent": {"$ref": n.get("$id")},
                                    "ErrorBehavior": 0, "Attempts": 1})
                        next_id[0] += 1
                        inserted += 1
                    out.append(it)
                vals[:] = out
            for k, v in n.items():
                if k != "Parent":
                    walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(container)
    return inserted


def _normal_name(name) -> str:
    return "".join(str(name or "").lower().split())


def read_back(path) -> list:
    """What the sequence FILE says each target will shoot, read from disk.

    [(target name, {filter: frames}, {filter: exposure_s}), ...] in running
    order, one per DeepSkyObjectContainer. The operator's rule after
    2026-10-14 ("after generating the plan, read it back to make sure it is
    what you think it should be"): every report of tonight's filters comes
    from this, never from the plan dicts the generator computed.
    """
    with open(path, encoding="utf-8-sig") as fh:
        seq = json.load(fh)
    area = _find_target_area(seq)
    out = []
    for t in (_items_of(area) if area else []):
        if _short_type(t) != "DeepSkyObjectContainer":
            continue
        name = (t.get("Target") or {}).get("TargetName") or t.get("Name")
        exposures = {}
        for se in _collect_smart_exposures(t):
            node = _filter_node(se)
            if node and _iterations(se) > 0:
                exposures.setdefault(node["_name"], float(_get_exposure_time(se) or 0.0))
        out.append((name, written_plan(t), exposures))
    return out


def _write_verified(sequence: Any, output_path: Path, expected: list) -> None:
    """Write *sequence*, read the file back, and keep it only if it says what
    *expected* says: [(target name, {filter: frames}[, {filter: seconds}]), ...]
    in order; the optional third item is an exposure override to confirm.

    Written beside the final name first and moved into place only after the
    read-back agrees, so a file that would shoot something other than the
    announced plan never exists under the name N.I.N.A loads. A target whose
    expected plan is empty (no hours known, nothing announced) is not
    compared. Mismatch: SequenceLintError, nothing new on disk."""
    from nina_gen import sequence_lint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".unverified" + output_path.suffix)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sequence, f, indent=2)
    problems = []
    try:
        back = read_back(tmp)
    except Exception as exc:
        back = []
        problems.append("the written sequence could not be read back: %s" % exc)
    if not problems and len(back) != len(expected):
        problems.append("the file holds %d target(s), the plan %d: %s vs %s"
                        % (len(back), len(expected), [b[0] for b in back], [e[0] for e in expected]))
    if not problems:
        for exp_row, (got_name, got_plan, got_exp) in zip(expected, back):
            name, plan = exp_row[0], exp_row[1]
            want_exp = exp_row[2] if len(exp_row) > 2 else None
            want = {f: int(n) for f, n in (plan or {}).items() if int(n) > 0}
            if _normal_name(name) != _normal_name(got_name):
                problems.append("target order differs: planned %s, file has %s" % (name, got_name))
            elif want and got_plan != want:
                problems.append("%s: the plan is %s but the file will shoot %s" % (name, want, got_plan))
            elif want and want_exp:
                for f in want:
                    if f in want_exp and abs(float(got_exp.get(f, 0.0)) - float(want_exp[f])) > 1e-6:
                        problems.append("%s: %s should expose %g s, the file says %g s"
                                        % (name, f, float(want_exp[f]), float(got_exp.get(f, 0.0))))
    if problems:
        try:
            tmp.unlink()
        except OSError:
            pass
        logging.getLogger(__name__).error(
            "REFUSING %s: the written sequence does not say what the plan says:\n  %s",
            output_path, "\n  ".join(problems))
        raise sequence_lint.SequenceLintError(problems)
    os.replace(tmp, output_path)
    logging.getLogger(__name__).info(
        "Sequence read back from %s: %s", output_path.name,
        "; ".join("%s %s" % (n, p) for n, p, _e in back))


def generate_slots_sequence(template_path: Path, slots: list, output_path: Path,
                            state_script: Optional[str] = None,
                            lint_scripts: bool = True) -> list:
    """Write a sequence with one target container per slot.

    *slots* is a list of dicts: ``name, ra_hours, dec_degrees, seconds, start,
    end`` (start/end local datetimes, may be None). The template's single
    DeepSkyObjectContainer is cloned once per slot with fresh object ids, each
    clone patched for its target and sized for its hours -- the same planning
    as a one-slot night, applied per slot -- and every slot given a hard end
    at its window end (_set_hard_end; None leaves the template's dawn). One
    roof open, one prelude, one set of flats: only the main section changes.

    Every slot after the first gets two ExternalScript steps around its setup
    (*state_script*, i.e. scripts/set_imaging_state.bat): DONE_MAIN as the
    previous slot ends and IN_MAIN as this one begins imaging. That is the
    hand-over signal the conductor's shadow turns into NINA_SLOT_DONE /
    SLOT_STARTED, and it costs the run nothing.

    Returns the per-slot filter plans, in slot order.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        sequence = json.load(f)
    import copy
    template = copy.deepcopy(sequence)
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
    focus_model = _focus_model()
    # Clone from a PRISTINE copy: the first slot edits the original in place
    # (name, coordinates, hard end), and a clone taken after that would carry
    # the first slot's end time into the second.
    import copy
    pristine = copy.deepcopy(proto)

    clones, plans, ends = [], [], []
    for k, slot in enumerate(slots):
        c = proto if k == 0 else _clone_with_fresh_ids(pristine, next_id)
        ra_h, ra_m, ra_s = _decompose_ra(slot["ra_hours"])
        dec_neg, dec_d, dec_m, dec_s = _decompose_dec(slot["dec_degrees"])
        coords = {"RAHours": ra_h, "RAMinutes": ra_m, "RASeconds": round(ra_s, 5),
                  "NegativeDec": dec_neg, "DecDegrees": dec_d, "DecMinutes": dec_m,
                  "DecSeconds": round(dec_s, 5)}
        _walk_and_replace(c, slot["name"], coords)
        _apply_rotation(c, slot.get("rotation", _rotation_for(slot["name"])))
        ensure_center_filter(c, sequence, next_id)
        # A capped target (max_hours) ENDS at its start + cap. The counts only
        # size one pass; the container repeats passes until this hard end, so
        # sizing the counts to the cap (5eb465f) did nothing on its own: on
        # 2026-10-20 ngc7790's 40 L + 13 G pass ran 3.8 times to its 00:00 end,
        # 191 frames instead of ~53. The next slot then starts at the earlier end.
        from datetime import timedelta
        end = slot.get("end")
        cap = _max_seconds_for(slot["name"])
        start = slot.get("start")
        if cap and start is not None and end is not None and start + timedelta(seconds=cap) < end:
            end = start + timedelta(seconds=cap)
            logging.getLogger(__name__).info(
                "Slot %d (%s): ends at %s, its max_hours, not the slot's %s",
                k + 1, slot["name"], end.strftime("%H:%M"), slot["end"].strftime("%H:%M"))
        ends.append(end)
        _set_hard_end(c, end, next_id)
        if cap:
            # ...and also the cap counted from the slot's REAL start (_add_cap_span).
            _add_cap_span(c, cap, next_id)
        plans.append(_plan_for(c, slot["name"], slot.get("seconds")))
        seed_focus(c, focus_model, next_id, ids=_index_ids(sequence))
        apply_filter_gain(c, ids=_index_ids(sequence))
        if k > 0:
            # No explicit autofocus after the first slot. Focus follows
            # temperature and filter, and both already have triggers inside
            # every block; the slot's first exposure changes filter anyway
            # (S-II -> L on the 09-08 plan), so the explicit run would only
            # add a second pass -- through the filter this rig focuses worst
            # on -- before the trigger's own.
            setup0 = next((it for it in _items_of(c)
                           if _short_type(it) == "SequentialContainer"), None)
            if setup0 is not None:
                vals = _items_of(setup0)
                vals[:] = [it for it in vals if _short_type(it) != "RunAutofocus"]
        if k > 0:
            # Park across a long wait (GAP_PARK_MIN_S). The gap is the planned
            # one: previous slot's hard end to this slot's start. The slot's
            # own Center re-acquires, so the park costs two slews and nothing
            # else. Inserted BEFORE the hand-over scripts below, which then
            # bracket the whole thing.
            prev_end = ends[k - 1]          # the previous slot's EFFECTIVE end (cap applied)
            rise = (horizon_rise(slot["name"], slot["ra_hours"], slot["dec_degrees"], prev_end)
                    if prev_end is not None else None)
            gap_s = max(0.0, (rise - prev_end).total_seconds()) if rise is not None else 0.0
            if gap_s >= GAP_PARK_MIN_S:
                setup = next((it for it in _items_of(c)
                              if _short_type(it) == "SequentialContainer"), None)
                if setup is not None and park_across_gap(setup, next_id):
                    logging.getLogger(__name__).info(
                        "Slot %d (%s): parking across a %.0f-minute gap",
                        k + 1, slot["name"], gap_s / 60.0)
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
    _dedupe_singletons(sequence)
    _refuse_if_broken(sequence, template, output_path, lint_scripts)
    _write_verified(sequence, output_path,
                    [(s["name"], p, _expected_exposures(s["name"])) for s, p in zip(slots, plans)])
    return plans


def generate_sequence(
    template_path: Path,
    dso_name: str,
    ra_hours: float,
    dec_degrees: float,
    output_path: Path,
    above_horizon_seconds: Optional[float] = None,
    end=None,
    lint_scripts: bool = True,
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
        end:                    Local wall-clock datetime at which the target's imaging must
                                stop (the planner's window end: earliest of horizon, dawn and
                                weather -- see _set_hard_end). None keeps the template's
                                condition, astronomical dawn.

    Returns:
        Dict mapping filter name to iteration count (empty if above_horizon_seconds not given).
    """
    with open(template_path, "r", encoding="utf-8") as f:
        sequence = json.load(f)
    import copy
    template = copy.deepcopy(sequence)

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
    _apply_rotation(sequence, _rotation_for(dso_name))
    ensure_center_filter(sequence, sequence, [_max_id(sequence) + 1])
    if end is not None:
        # Only the target's own container: the template's start and end
        # sections keep whatever clocks they carry.
        area = _find_target_area(sequence)
        targets = [it for it in _items_of(area)
                   if _short_type(it) == "DeepSkyObjectContainer"] if area else []
        next_id = [_max_id(sequence) + 1]
        for t in targets:
            _set_hard_end(t, end, next_id)
        _dedupe_singletons(sequence)

    filter_plan = _plan_for(sequence, dso_name, above_horizon_seconds)
    seed_focus(sequence, _focus_model(), [_max_id(sequence) + 1])
    apply_filter_gain(sequence, ids=_index_ids(sequence))
    _refuse_if_broken(sequence, template, output_path, lint_scripts)
    _write_verified(sequence, output_path,
                    [(dso_name, filter_plan, _expected_exposures(dso_name))])
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
