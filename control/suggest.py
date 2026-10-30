"""What is worth adding to the queue (E1 of docs/AUTOMATION_PLAN.md, 2026-10-30).

    score(cand, nights, moon_worth, fov, excluded) -> {"score", "reasons", ...}   pure
    rank(cands, ...) -> the candidates in order, best first                      pure

The owner chose "suggest, owner approves": nothing here queues anything. A
candidate from configs/candidates.json (hand-curated showpieces, so this
works in a SIMBAD outage) scores on four things, each 0..1, multiplied:

  visibility   the share of the next NIGHTS nights with at least MIN_HOURS
               hours above the tree line in the dark (the planner's own
               horizon and dark times, computed by scripts/suggest_targets);
               below MIN_NIGHTS such nights the candidate is out
  size         how the object's long axis sits in the field: between
               SIZE_BAND of the field's SHORT side is 1, tapering to 0 at
               a twentieth of the field (a dot) and at 1.5 fields (a mosaic)
  moon         the mean worth of an hour over the next fortnight for the
               candidate's palette (iris_astronomy.moon_cost.plan_worth):
               narrowband barely notices the moon, broadband does
  season       1 while the best of those nights is still rising or near
               transit, less when the object is already setting at dusk
               (a candidate caught late is a candidate for next year)

A name that is already in the queue (any status), already has frames on
disk, or is already on the site (by its own name or a known alias) is
excluded with the reason said, never scored.
"""
import json
from typing import Optional

NIGHTS = 30
MIN_HOURS = 3.0
MIN_NIGHTS = 10
SIZE_BAND = (0.15, 0.9)
SENSOR_PX = (9576, 6388)        # QHY600M (iris_astronomy.show_dso)
KINDS = ("nebula", "snr", "pn", "reflection", "galaxy", "cluster")
DEFAULT_PLANS = {"SHO": {"Ha": 1, "O-III": 1, "S-II": 1}, "HOO": {"Ha": 1, "O-III": 1},
                 "LRGB": {"L": 2, "R": 1, "G": 1, "B": 1}, "HALRGB": {"L": 2, "R": 1, "G": 1, "B": 1, "Ha": 1}}


def squash(name) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def load_catalogue(path) -> list:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return list(d.get("candidates") or [])


def fov_arcmin(arcsec_per_px: float, sensor_px=SENSOR_PX) -> tuple:
    """(long, short) side of the field in arcminutes."""
    w, h = sensor_px
    return (w * arcsec_per_px / 60.0, h * arcsec_per_px / 60.0)


def size_fit(size_arcmin: float, fov: tuple) -> float:
    """0..1: how the object's long axis sits against the field's short side."""
    short = min(fov)
    if not size_arcmin or short <= 0:
        return 0.5
    r = float(size_arcmin) / short
    lo, hi = SIZE_BAND
    if lo <= r <= hi:
        return 1.0
    if r < lo:
        floor = 0.05
        return max(0.0, (r - floor) / (lo - floor))
    ceil = 1.5
    return max(0.0, (ceil - r) / (ceil - hi))


def visibility(hours_by_night: list, min_hours: float = MIN_HOURS) -> tuple:
    """(share of nights with >= min_hours, count of such nights)."""
    good = [h for h in hours_by_night if (h or 0) >= min_hours]
    return (len(good) / len(hours_by_night) if hours_by_night else 0.0), len(good)


def season(hours_by_night: list) -> float:
    """1 when the nights ahead are as good as or better than the first;
    down to 0.5 when the object is clearly on its way out."""
    if len(hours_by_night) < 4:
        return 1.0
    q = max(1, len(hours_by_night) // 4)
    first, last = sum(hours_by_night[:q]) / q, sum(hours_by_night[-q:]) / q
    if first <= 0:
        return 1.0 if last > 0 else 0.5
    return max(0.5, min(1.0, 0.5 + 0.5 * (last / first)))


def excluded_by(name: str, aliases: list, queue: set, imaged: set, site: set) -> Optional[str]:
    """Why a candidate is out, or None. The sets hold squashed names."""
    keys = {squash(name)} | {squash(a) for a in aliases or []}
    keys = {k for k in keys if len(k) >= 2}
    if keys & queue:
        return "already in the queue"
    if keys & imaged:
        return "already has frames on disk"
    if keys & site:
        return "already on the site"
    return None


def score(cand: dict, hours_by_night: list, moon_worth: float, fov: tuple,
          excluded: Optional[str] = None) -> dict:
    """One candidate's verdict. Pure."""
    out = {"name": cand.get("name"), "common": cand.get("common"), "kind": cand.get("kind"),
           "palette": cand.get("palette"), "score": 0.0, "reasons": [], "excluded": excluded}
    if excluded:
        out["reasons"].append(excluded)
        return out
    vis, good = visibility(hours_by_night)
    if good < MIN_NIGHTS:
        out["reasons"].append("only %d of the next %d nights give %.0f h above the trees" % (
            good, len(hours_by_night), MIN_HOURS))
        return out
    sz = size_fit(cand.get("size_arcmin") or 0, fov)
    se = season(hours_by_night)
    mw = max(0.0, min(1.0, float(moon_worth if moon_worth is not None else 1.0)))
    out.update({"visibility": round(vis, 3), "good_nights": good, "size_fit": round(sz, 3),
                "moon_worth": round(mw, 3), "season": round(se, 3),
                "score": round(vis * sz * mw * se, 4)})
    out["reasons"].append("%d of %d nights with %.0f h+" % (good, len(hours_by_night), MIN_HOURS))
    short = min(fov)
    out["reasons"].append("%s' across a %.0f' field (%s)" % (
        cand.get("size_arcmin"), short,
        "fits" if sz >= 0.99 else ("small" if (cand.get("size_arcmin") or 0) < SIZE_BAND[0] * short else "larger than the field")))
    out["reasons"].append("moon: an hour is worth %.2f for %s" % (mw, cand.get("palette")))
    if se < 0.99:
        out["reasons"].append("setting: later nights give %.0f%% of the first" % (100 * (2 * se - 1)))
    return out


def rank(cands: list, hours: dict, moon: dict, fov: tuple, queue: set, imaged: set, site: set,
         aliases: Optional[dict] = None) -> list:
    """Every candidate scored, best first; the excluded and the unscorable last.

    hours: {name: [hours per night]}, moon: {name: worth}, aliases: {name: [alias]}.
    """
    out = []
    for c in cands:
        name = c.get("name")
        ex = excluded_by(name, (aliases or {}).get(name) or [], queue, imaged, site)
        out.append(score(c, hours.get(name) or [], moon.get(name), fov, ex))
    return sorted(out, key=lambda v: (-v["score"], v["name"] or ""))


def catalogue_kind(name: str, cands: list) -> Optional[str]:
    """The queue's obj_type for a catalogue name: galaxy / nebula (an emission
    object of any kind) / None (clusters and reflection nebulae plan broadband,
    an unknown name says nothing). Pure. E2: `image <name>` falls back to this
    when the catalogue classifier cannot say."""
    key = squash(name)
    for c in cands or []:
        if squash(c.get("name")) == key:
            k = c.get("kind")
            return "galaxy" if k == "galaxy" else ("nebula" if k in ("nebula", "snr", "pn") else None)
    return None


def describe(ranked: list, top: int = 3) -> str:
    """The chat text: the top candidates and why, plus the line that queues one."""
    picks = [v for v in ranked if v["score"] > 0][:top]
    if not picks:
        return "Nothing worth adding this month: every candidate is queued, imaged, on the site or out of season."
    lines = ["Worth adding this month (suggest; `image <name>` queues one):"]
    for v in picks:
        lines.append("  %s -- %s (%s, %s): %s" % (v["name"], v["common"], v["kind"], v["palette"],
                                                "; ".join(v["reasons"])))
    return "\n".join(lines)
