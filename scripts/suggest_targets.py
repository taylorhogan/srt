"""suggest_targets.py -- what is worth adding to the queue this month (E1/E2).

    python scripts/suggest_targets.py              # the top 3, with reasons
    python scripts/suggest_targets.py --top 5 --all
    python scripts/suggest_targets.py --post       # the card to the web chat (E2)

docs/AUTOMATION_PLAN.md E. The scoring is control/suggest (pure); this
script gathers the live inputs: for every candidate in
configs/candidates.json, the hours above the tree line inside the dark
hours on each of the next cfg["suggest"]["nights"] nights (the planner's own
horizon, configs/my.hrz, and astronomical twilight, astroplan), the mean
worth of an hour under the moon over the next fortnight for the
candidate's palette (iris_astronomy.moon_cost), the field from
nina.arc_sec_per_pixel, and the exclusions: every queue row, every target
folder under the image dir, every gallery entry on the site (with SIMBAD
aliases when the network answers; the catalogue's own names otherwise).

Nothing is queued: `image <name>` is the owner's.
"""
from __future__ import annotations

import argparse
import datetime as dt
import warnings
import json
import os
import sys
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from control import suggest  # noqa: E402

DEFAULTS = {"top": 3, "weekday": 0, "catalogue": "configs/candidates.json", "nights": 30}


def settings() -> dict:
    try:
        from configs import config
        return {**DEFAULTS, **(config.data().get("suggest") or {})}
    except Exception:  # noqa: BLE001
        return dict(DEFAULTS)


def hours_above(cands: list, nights: int, lat: float, lon: float, elev: float, tz: str) -> dict:
    """{name: [hours above the horizon inside the dark, per night]} for the next *nights*."""
    import astropy.units as u
    import numpy as np
    from astroplan import FixedTarget, Observer
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    from iris_astronomy.astro_dso_visibility import get_horizon_from_azimuth, map_az_to_horizon
    site = Observer(location=EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, elev * u.m), timezone=tz)
    az_h, al_h = map_az_to_horizon()
    today = dt.date.today()
    # one grid of hourly sample times per night, between astronomical dusk and dawn
    grids = []
    for n in range(nights):
        d = today + dt.timedelta(days=n)
        noon = Time(dt.datetime(d.year, d.month, d.day, 17, 0, 0))        # 17:00 UT ~ noon local
        dusk = site.twilight_evening_astronomical(noon, which="next")
        dawn = site.twilight_morning_astronomical(dusk, which="next")
        hrs = max(1, int((dawn - dusk).to_value("hr")))
        grids.append(dusk + np.arange(hrs) * u.hour)
    flat = Time(np.concatenate([g.jd for g in grids]), format="jd")
    counts = [len(g) for g in grids]
    out = {}
    for c in cands:
        tgt = FixedTarget(SkyCoord(ra=c["ra_deg"] * u.deg, dec=c["dec_deg"] * u.deg), name=c["name"])
        aa = site.altaz(flat, tgt)
        alt, az = aa.alt.deg, aa.az.deg
        up = np.array([a >= get_horizon_from_azimuth(z, az_h, al_h) for a, z in zip(alt, az)])
        hours, i = [], 0
        for k in counts:
            hours.append(float(up[i:i + k].sum()))
            i += k
        out[c["name"]] = hours
    return out


def moon_worths(cands: list, lat: float, lon: float, elev: float, nights: int = 14) -> dict:
    """{name: mean worth of a midnight hour over the next *nights* for its palette}."""
    import astropy.units as u
    from astroplan import moon_illumination
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
    from astropy.time import Time

    from iris_astronomy import moon_cost
    # the moon's separation from a fixed target: astropy warns that a GCRS
    # (moon) to ICRS (target) separation depends on the direction; at the
    # degree level the moon rule works at, it does not
    warnings.filterwarnings("ignore", message=".*NonRotationTransformationWarning.*")
    warnings.filterwarnings("ignore", message=".*transforming other coordinates.*")
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, elev * u.m)
    today = dt.date.today()
    moons = []
    for n in range(nights):
        d = today + dt.timedelta(days=n)
        t = Time(dt.datetime(d.year, d.month, d.day, 5, 0, 0))             # 05:00 UT ~ 01:00 local
        frame = AltAz(obstime=t, location=loc)
        m = get_body("moon", t, loc)
        moons.append((t, frame, m, float(moon_illumination(t))))
    fit = moon_cost.load_fit()
    out = {}
    for c in cands:
        plan = suggest.DEFAULT_PLANS.get(c.get("palette"), suggest.DEFAULT_PLANS["LRGB"])
        tgt = SkyCoord(ra=c["ra_deg"] * u.deg, dec=c["dec_deg"] * u.deg)
        ws = []
        for t, frame, m, illum in moons:
            ma = m.transform_to(frame).alt.deg
            ta = tgt.transform_to(frame).alt.deg
            b = moon_cost.moon_brightness(illum, float(tgt.separation(m).deg), float(ma), float(ta))
            ws.append(moon_cost.plan_worth(plan, b, fit))
        out[c["name"]] = sum(ws) / len(ws) if ws else 1.0
    return out


def exclusions(cfg: dict) -> tuple:
    """(queue, imaged, site) as sets of squashed names."""
    queue, imaged, site = set(), set(), set()
    try:
        # On the Spark the queue is the PC's 11:40 mirror, not the repo copy.
        mirror = Path.home() / "Desktop" / "iris_state" / "live" / "local" / "my_instructions.json"
        path = mirror if mirror.exists() else Path(_root) / cfg["location"]["instructions"]
        rows = json.loads(path.read_text(encoding="utf-8"))
        queue = {suggest.squash(r.get("dso")) for r in rows if r.get("dso")}
    except Exception:  # noqa: BLE001
        pass
    try:
        image_dir = Path(cfg["nina"]["image_dir"])
        imaged = {suggest.squash(p.name) for p in image_dir.iterdir() if p.is_dir()}
    except Exception:  # noqa: BLE001
        pass
    try:
        from scripts import astro_output as ao
        html = (ao.SITE / "index.html").read_text(encoding="utf-8")
        for a in ao.gallery_articles(html):
            site.add(suggest.squash(a["id"]))
            site.add(suggest.squash(a["title"].split("-")[0]))
    except Exception:  # noqa: BLE001
        pass
    return queue, imaged, site


def catalogue_aliases(cands: list) -> dict:
    """The common names as aliases, so 'Pelican' on the site excludes ic5070."""
    return {c["name"]: [c.get("common") or ""] for c in cands}


def compute(cfg: dict, s: dict) -> list:
    cands = suggest.load_catalogue(Path(_root) / s["catalogue"])
    loc = cfg["location"]
    lat, lon, elev = float(loc["latitude"]), float(loc["longitude"]), float(loc.get("elevation") or 0)
    hours = hours_above(cands, int(s["nights"]), lat, lon, elev, loc.get("timezone", "US/Eastern"))
    moon = moon_worths(cands, lat, lon, elev)
    fov = suggest.fov_arcmin(float(cfg["nina"]["arc_sec_per_pixel"]))
    queue, imaged, site = exclusions(cfg)
    return suggest.rank(cands, hours, moon, fov, queue, imaged, site, catalogue_aliases(cands))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="every candidate with its score or its exclusion")
    ap.add_argument("--post", action="store_true", help="post the card to the web chat")
    a = ap.parse_args(argv)
    from configs import config
    cfg = config.data()
    s = settings()
    ranked = compute(cfg, s)
    text = suggest.describe(ranked, a.top or int(s["top"]))
    print(text)
    if a.all:
        print()
        for v in ranked:
            print("%-16s %6.3f  %s" % (v["name"], v["score"], "; ".join(v["reasons"])))
    if a.post:
        from utils.webchat_client import post_to_webchat
        ok = post_to_webchat(text)
        print("posted" if ok else "NOT posted")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
