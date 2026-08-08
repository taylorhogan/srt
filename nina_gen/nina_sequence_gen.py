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


def _update_smart_exposure(se: dict, iterations: int, filter_name: Optional[str] = None) -> None:
    """Set LoopCondition iterations and optionally override the inline filter name."""
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


def _apply_filter_plan(sequence: Any, above_horizon_seconds: float, obj_type: str) -> dict[str, int]:
    """
    Compute per-filter iteration counts, patch all SmartExposure blocks in the
    sequence in-place, and return the plan as {filter_name: iterations}.
    """
    actual_seconds = above_horizon_seconds * 0.8
    smart_exposures = _collect_smart_exposures(sequence)

    # Template order is L, R, G, B (4 blocks)
    if len(smart_exposures) < 4:
        return {}

    se_l, se_r, se_g, se_b = smart_exposures[:4]
    exp_l = _get_exposure_time(se_l)
    exp_r = _get_exposure_time(se_r)
    exp_g = _get_exposure_time(se_g)
    exp_b = _get_exposure_time(se_b)

    if obj_type == "nebula":
        # Divide equally across Ha, O (OIII), S (SII); no L frames
        each = actual_seconds / 3.0
        plan = {
            "Ha": math.floor(each / exp_r),
            "O":  math.floor(each / exp_g),
            "S":  math.floor(each / exp_b),
        }
        _update_smart_exposure(se_l, iterations=0)
        _update_smart_exposure(se_r, iterations=plan["Ha"], filter_name="Ha")
        _update_smart_exposure(se_g, iterations=plan["O"],  filter_name="O")
        _update_smart_exposure(se_b, iterations=plan["S"],  filter_name="S")
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

    filter_plan: dict[str, int] = {}
    if above_horizon_seconds is not None and above_horizon_seconds > 0:
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
