"""The filter advisor: tonight's filter ratio for a target, and why.

    advise(frames, obj_type, entry, tonight) -> {"dso", "plan", "reasons", ...}   pure
    advise_for(dso, start, end) -> same, from the live inputs; never raises

docs/FILTER_ADVISOR_PLAN.md. Built 2026-10-18 after three hand edits before
noon that the convergence-based split (fits_processing.filter_need) could not
make: a soft L that read as "converged" (ngc1961, 51 L at a median 2.55"),
Hα that a galaxy's fixed L/R/G/B set never includes (ngc6946), and a monitor
target that needs a fixed recipe (hubblev1).

The plan is a RATIO in the same form as the `filters` command, which the
sequence generator reads as a share of the slot's hours. Need per filter is
goal minus have, where "have" counts only frames that would survive stacking
(FWHM within 1.25x that filter's median, stacker.quality_gate) and, for the
detail channel, only SHARP ones. Tonight's seeing and moon then tilt it.

Step 1 is shadow only: the scheduler posts this beside the real plan at noon
and changes nothing.
"""
import re
from pathlib import Path
from typing import Optional

REF_EXPOSURE_S = 300.0
GATE_MULTIPLIER = 1.25          # stacker.quality_gate's max_fwhm_multiplier default
SHARP_ARCSEC = 2.4              # detail-channel frames must be at least this sharp
MAX_FILTERS = 4                 # template blocks whose filter can be set
RATIO_SCALE = 4                 # the largest share is written as this

# Goals in REF_EXPOSURE_S frames. For the detail channel the goal counts SHARP
# frames only; for the others, usable ones.
GOALS = {
    "galaxy": {"L": 60, "R": 24, "G": 24, "B": 24, "Ha": 36},
    "nebula": {"Ha": 40, "O-III": 40, "S-II": 40},
}
DETAIL = {"galaxy": "L", "nebula": "Ha"}
BASE_SET = {"galaxy": ("L", "R", "G", "B"), "nebula": ("Ha", "O-III", "S-II")}
DEEPEN = {"galaxy": {"L": 2, "R": 1, "G": 1, "B": 1}, "nebula": {"Ha": 1, "O-III": 1, "S-II": 1}}

SEEING_FACTOR = {"good": 1.5, "fair": 1.0, "poor": 0.5, "bad": 0.5}
MOON_LIT = 0.5                  # illuminated fraction above which the moon matters
MOON_FACTOR = {"L": 0.5, "B": 0.5, "Ha": 1.5, "O-III": 0.7}

_CANON = {"l": "L", "lum": "L", "luminance": "L", "r": "R", "red": "R", "g": "G", "green": "G",
          "b": "B", "blue": "B", "ha": "Ha", "halpha": "Ha", "h-alpha": "Ha",
          "oiii": "O-III", "o-iii": "O-III", "o3": "O-III", "sii": "S-II", "s-ii": "S-II", "s2": "S-II"}


def canonical(name) -> Optional[str]:
    return _CANON.get(str(name or "").strip().lower().replace(" ", "").replace("_", ""))


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return None if not n else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def tally(frames: list, exposure_s: Optional[dict] = None) -> dict:
    """{filter: {"frames", "usable", "sharp", "median_fwhm"}} from frame records.

    *frames* are {"filter", "fwhm_arcsec", "exptime"} (frame_stats.json rows
    with the exposure added). Only frames at the filter's exposure count --
    the entry's exposure_s if it has one, else REF_EXPOSURE_S -- so a night
    of 10 s frames does not pass for depth.
    """
    by = {}
    for fr in frames:
        f = canonical(fr.get("filter"))
        if f is None:
            continue
        want = float((exposure_s or {}).get(f, REF_EXPOSURE_S))
        exp = fr.get("exptime")
        if exp is not None and abs(float(exp) - want) > 0.5:
            continue
        by.setdefault(f, []).append(fr.get("fwhm_arcsec"))
    out = {}
    for f, fw in by.items():
        measured = [float(x) for x in fw if x]
        med = _median(measured)
        usable = sum(1 for x in measured if med is None or x <= GATE_MULTIPLIER * med)
        out[f] = {"frames": len(fw), "usable": usable,
                  "sharp": sum(1 for x in measured if x <= SHARP_ARCSEC),
                  "median_fwhm": med}
    return out


def _ratio(weights: dict) -> dict:
    """Weights -> small whole numbers, largest = RATIO_SCALE, any share >= 1."""
    top = max(weights.values()) if weights else 0
    if top <= 0:
        return {}
    return {f: max(1, int(round(RATIO_SCALE * w / top))) for f, w in weights.items() if w > 0}


def advise(frames: list, obj_type: str, entry: Optional[dict] = None,
           tonight: Optional[dict] = None, dso: str = "") -> dict:
    """Tonight's filter ratio for one target, with the reasons. Pure.

    obj_type: "galaxy" / "nebula" / anything else (treated as galaxy: broadband).
    entry: the queue entry (lightcurve, recipe, exposure_s, filter_plan).
    tonight: {"seeing": "good"|"fair"|"poor"|"bad"|"unknown",
              "moon_lit": fraction or None, "moon_up": bool}.
    -> {"dso", "plan": {filter: ratio} or None, "reasons": [str], "need": {...},
        "have": tally, "action": "plan"|"monitor"|"done"}
    """
    entry = entry or {}
    tonight = tonight or {}
    kind = "nebula" if obj_type == "nebula" else "galaxy"
    have = tally(frames, entry.get("exposure_s"))
    reasons = []
    res = {"dso": dso or entry.get("dso", ""), "kind": kind, "have": have, "need": {},
           "plan": None, "reasons": reasons, "tonight": tonight}

    if entry.get("lightcurve"):
        reasons.append("monitor target (lightcurve): its own fixed plan stands")
        res["action"] = "monitor"
        res["plan"] = entry.get("filter_plan")
        return res

    fset = list(BASE_SET[kind])
    if kind == "galaxy" and ("Ha" in have or str(entry.get("recipe", "")).upper() == "HALRGB"):
        fset.append("Ha")
        reasons.append("Hα joins: %s" % ("%d Hα frames already" % have["Ha"]["frames"]
                                          if "Ha" in have else "the entry asks for HALRGB"))
    detail = DETAIL[kind]
    goals = GOALS[kind]
    need = {}
    for f in fset:
        h = have.get(f, {"frames": 0, "usable": 0, "sharp": 0, "median_fwhm": None})
        got = h["sharp"] if f == detail else h["usable"]
        need[f] = max(0.0, float(goals.get(f, 0)) - got)
        if f == detail and h["frames"] and h["sharp"] < h["usable"]:
            reasons.append("%s: %d frames, %d usable but only %d sharp (<= %.1f\"; median %.2f\")"
                           % (f, h["frames"], h["usable"], h["sharp"], SHARP_ARCSEC,
                              h["median_fwhm"] or 0))

    if not any(need.values()):
        reasons.append("every goal met (%s): a candidate to retire; deepening split"
                       % ", ".join("%s %d" % (f, goals.get(f, 0)) for f in fset))
        res["need"] = need
        res["plan"] = {f: w for f, w in DEEPEN[kind].items() if f in fset}
        res["action"] = "done"
        return res

    w = dict(need)
    seeing = tonight.get("seeing") or "unknown"
    fac = SEEING_FACTOR.get(seeing)
    if fac and fac != 1.0 and w.get(detail):
        w[detail] *= fac
        reasons.append("seeing forecast %s: %s x%.1f" % (seeing, detail, fac))
    lit = tonight.get("moon_lit")
    if tonight.get("moon_up") and lit is not None and lit > MOON_LIT:
        moved = []
        for f, k in MOON_FACTOR.items():
            if w.get(f):
                w[f] *= k
                moved.append("%s x%.1f" % (f, k))
        if moved:
            reasons.append("moon %d%% lit and up: %s" % (round(100 * lit), ", ".join(moved)))

    active = {f: v for f, v in w.items() if v > 0}
    if len(active) > MAX_FILTERS:
        out = min(active, key=active.get)
        reasons.append("%s sits out tonight (smallest need, %d filters for %d blocks); "
                       "it rotates in as the others fill" % (out, len(active), MAX_FILTERS))
        active.pop(out)
    res["need"] = need
    res["plan"] = _ratio(active)
    res["action"] = "plan"
    reasons.insert(0, "need (goal - have): " + ", ".join(
        "%s %d" % (f, need[f]) for f in fset))
    return res


def describe(adv: dict) -> str:
    plan = adv.get("plan") or {}
    p = " ".join("%s=%s" % (f, n) for f, n in plan.items()) or "none"
    return "%s: %s  (%s)" % (adv.get("dso"), p, "; ".join(adv.get("reasons") or []))


# --------------------------------------------------------------------------- live
_EXP_RE = re.compile(r"_(\d+(?:\.\d+)?)s_")


def _frames_for(dso: str) -> list:
    import json
    from configs import config
    root = Path(config.data()["nina"]["image_dir"])
    key = dso.lower().replace(" ", "")
    d = next((p for p in root.iterdir() if p.is_dir() and p.name.lower().replace(" ", "") == key), None)
    if d is None or not (d / "frame_stats.json").exists():
        return []
    rows = json.loads((d / "frame_stats.json").read_text(encoding="utf-8"))
    out = []
    for r in rows if isinstance(rows, list) else []:
        m = _EXP_RE.search(str(r.get("path", "")))
        out.append({"filter": r.get("filter"), "fwhm_arcsec": r.get("fwhm_arcsec"),
                    "exptime": float(m.group(1)) if m else None})
    return out


def tonight_for(start, end) -> dict:
    """Seeing label (worst 850 hPa wind over [start, end)) and the moon at mid-slot."""
    from configs import config
    loc = config.data()["location"]
    lat, lon = float(loc["latitude"]), float(loc["longitude"])
    out = {"seeing": "unknown", "moon_lit": None, "moon_up": False}
    try:
        from iris_astronomy import weather
        hours, wind = weather.get_seeing_wind_by_hour(lat, lon, 48)
        want = set()
        t = start
        from datetime import timedelta
        while t < end:
            want.add(t.hour)
            t += timedelta(hours=1)
        # The first occurrence of each hour is tonight's (the list runs forward from now).
        seen, vals = set(), []
        for h, w in zip(hours, wind):
            if h in want and h not in seen and w is not None:
                seen.add(h)
                vals.append(float(w))
        if vals:
            out["seeing"] = weather.seeing_from_wind(max(vals))
            out["seeing_wind_kmh"] = max(vals)
    except Exception:
        pass
    try:
        import astropy.units as u
        from astroplan import moon_illumination
        from astropy.coordinates import AltAz, EarthLocation, get_body
        from astropy.time import Time
        mid = Time(start + (end - start) / 2)
        site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
        alt = get_body("moon", mid, site).transform_to(AltAz(obstime=mid, location=site)).alt.deg
        out.update({"moon_lit": float(moon_illumination(mid)), "moon_up": bool(alt > 0),
                    "moon_alt_deg": float(alt)})
    except Exception:
        pass
    return out


def advise_for(dso: str, start=None, end=None) -> dict:
    """advise() on the live inputs. Never raises: a failure is returned as a reason."""
    try:
        entry = None
        try:
            from control import instructions
            entry = instructions.get_instruction_by_dso(dso)
        except Exception:
            entry = None
        try:
            from nina_gen.nina_sequence_gen import _classify_object_type
            obj_type = _classify_object_type(dso)
        except Exception:
            obj_type = "unknown"
        tonight = tonight_for(start, end) if start is not None and end is not None else {}
        return advise(_frames_for(dso), obj_type, entry, tonight, dso=dso)
    except Exception as exc:  # noqa: BLE001
        return {"dso": dso, "plan": None, "action": "error",
                "reasons": ["advisor failed: %s: %s" % (type(exc).__name__, exc)]}
