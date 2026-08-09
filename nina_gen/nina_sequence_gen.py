import json
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

    # An explicit plan set by the `filters` command applies even when the hours
    # above horizon are unknown: the counts are absolute, not a share of the
    # night, so there is nothing to scale them against.
    explicit = None
    try:
        from control import instructions as _instr
        explicit = _instr.get_filter_plan(dso_name)
    except Exception:
        explicit = None

    filter_plan: dict[str, int] = {}
    if explicit:
        filter_plan = _apply_filter_plan(sequence, above_horizon_seconds or 0.0,
                                         "explicit", explicit=explicit)
    elif above_horizon_seconds is not None and above_horizon_seconds > 0:
        obj_type = _classify_object_type(dso_name)
        filter_plan = _apply_filter_plan(sequence, above_horizon_seconds, obj_type)

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
