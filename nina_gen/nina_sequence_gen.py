import json
from pathlib import Path
from typing import Any


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
) -> None:
    """
    Generate a NINA imaging sequence from a template with a new target.

    Args:
        template_path: Path to the NINA JSON sequence template.
        dso_name:      Display name of the target (e.g. "M 51").
        ra_hours:      Right ascension in decimal hours (e.g. 13.4978).
        dec_degrees:   Declination in decimal degrees; negative for south (e.g. 47.195).
        output_path:   Destination path for the generated sequence file.
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sequence, f, indent=2)


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
