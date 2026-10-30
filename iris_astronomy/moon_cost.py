"""What the moon costs a target, hour by hour: the planner's moon rule (2026-10-25).

A broadband target under a bright moon gets a fraction of a dark hour's worth;
narrowband barely notices. Measured on this site's own frames by
scripts/moon_sky_cost.py: each filter's sky rises as 1 + c x the moon's sky
brightness (Krisciunas & Schaefer 1991), and a 300 s frame is worth the
noise-variance ratio against a dark one (read noise 5.82 ADU, gain 1.328
e-/ADU at gain 0). At a 95% moon 60 deg away: L 0.45, G 0.37, Ha 0.88,
O-III 0.81.

The rule: an hour above the tree line is held for the moon ("m" in the
planner's symbols, so slot planning skips it) when the target's predicted
worth, averaged over its own filter plan, is below cfg["planner"]["moon_min_worth"]
(0.6). Per target, the queue entry's "moon" overrides it:

    "any"    never held (narrowband projects, or a deliberate moonlit run)
    "avoid"  the worth rule (the default)
    "dark"   held whenever the moon is above the horizon at all

Light-curve monitors default to "any": a point every night matters more than
its depth (and the Cepheids in NGC 7790 are 10th magnitude).

Pure apart from load_fit(), which prefers the latest measured fit in
local/analysis/moon_sky_fit.json and falls back to the 2026-10-25 values.
"""
import json
import math
from pathlib import Path
from typing import Optional

K_V = 0.172
READ_NOISE_ADU, GAIN = 5.82, 1.328
# 2026-10-25 fit (scripts/moon_sky_cost.py): c per nL, and the dark sky in ADU/s.
DEFAULT_FIT = {
    "c": {"L": 1.82e-3, "R": 1.19e-3, "G": 3.00e-3, "B": 2.00e-3, "Ha": 1.19e-2, "O-III": 2.11e-2},
    "dark_sky_adu_s": {"L": 1.2845, "R": 0.2668, "G": 0.5468, "B": 0.4061, "Ha": 0.0024, "O-III": 0.0023},
}
# S-II has no moonlit frames yet; it is the narrowest line and behaves like Ha.
_ALIAS = {"S-II": "Ha", "SII": "Ha", "OIII": "O-III", "H": "Ha"}
BROADBAND_DEFAULT = {"L": 2, "R": 1, "G": 1, "B": 1}
NARROWBAND_DEFAULT = {"Ha": 1, "O-III": 1}
MODES = ("any", "avoid", "dark")

FIT_FILE = Path(__file__).resolve().parent.parent / "local" / "analysis" / "moon_sky_fit.json"


def load_fit(path: Path = FIT_FILE) -> dict:
    try:
        d = json.loads(path.read_text())
        if d.get("c") and d.get("dark_sky_adu_s"):
            return {"c": {**DEFAULT_FIT["c"], **d["c"]},
                    "dark_sky_adu_s": {**DEFAULT_FIT["dark_sky_adu_s"], **d["dark_sky_adu_s"]}}
    except (OSError, ValueError):
        pass
    return DEFAULT_FIT


def _airmass(alt_deg: float) -> float:
    z = math.radians(90 - alt_deg)
    return (1 - 0.96 * math.sin(z) ** 2) ** -0.5


def moon_brightness(illum: float, sep_deg: float, moon_alt: float, target_alt: float) -> float:
    """The moon's sky brightness at the target in nanoLamberts (Krisciunas & Schaefer 1991)."""
    if moon_alt <= 0:
        return 0.0
    elong = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * illum))))
    a = 180.0 - elong
    istar = 10 ** (-0.4 * (3.84 + 0.026 * abs(a) + 4e-9 * a ** 4))
    rho = max(sep_deg, 1.0)
    f = 10 ** 5.36 * (1.06 + math.cos(math.radians(rho)) ** 2) + 10 ** (6.15 - rho / 40)
    return (f * istar * 10 ** (-0.4 * K_V * _airmass(moon_alt))
            * (1 - 10 ** (-0.4 * K_V * _airmass(max(target_alt, 5)))))


def frame_worth(dark_sky_adu_s: float, c: float, b_moon: float, exptime: float = 300.0) -> float:
    """What a frame under moonlight b_moon is worth against a dark one (noise-variance ratio)."""
    s0 = dark_sky_adu_s * exptime / GAIN
    rn2 = READ_NOISE_ADU ** 2
    return (s0 + rn2) / (s0 * (1 + c * b_moon) + rn2)


def target_plan(entry: dict, dso_type: str) -> dict:
    """The filter mix the moon is judged against: the entry's own plan, else by type."""
    from datetime import datetime
    from control import filter_plan
    plan = filter_plan.active_plan(entry, filter_plan.night_for(datetime.now()))
    if plan:
        return plan
    kind = str(entry.get("obj_type") or dso_type or "").lower()
    return NARROWBAND_DEFAULT if kind in ("n", "nebula", "pn", "snr") else BROADBAND_DEFAULT


def plan_worth(plan: dict, b_moon: float, fit: Optional[dict] = None) -> float:
    """Worth of an hour on *plan* under b_moon, weighted by the plan's own ratio."""
    fit = fit or DEFAULT_FIT
    tot = wsum = 0.0
    for f, n in plan.items():
        key = _ALIAS.get(f, f)
        c, dark = fit["c"].get(key), fit["dark_sky_adu_s"].get(key)
        if c is None or dark is None or not n:
            continue
        tot += float(n)
        wsum += float(n) * frame_worth(dark, c, b_moon)
    return wsum / tot if tot else 1.0


def moon_mode(entry: dict) -> str:
    m = str(entry.get("moon") or "").lower()
    if m in MODES:
        return m
    return "any" if entry.get("lightcurve") else "avoid"


def hour_held(mode: str, worth: float, moon_alt: float, min_worth: float) -> bool:
    """True when the moon should hold this hour for this target."""
    if mode == "any":
        return False
    if mode == "dark":
        return moon_alt > 0
    return worth < min_worth
