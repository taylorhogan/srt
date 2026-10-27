"""Store coordinates in every queue entry that lacks them, cross-checked (2026-10-26).

    python scripts/fill_queue_coords.py            # dry run: show what would be written
    python scripts/fill_queue_coords.py --write    # write the verified ones
    python scripts/fill_queue_coords.py --wait     # wait for SIMBAD (every 10 min, up to 12 h), then --write

Why: on 2026-10-26 SIMBAD (CDS) was down and each lookup hung ~20 s before
failing; 30 of 34 waiting targets had no stored RA/Dec, so `tonight` took 5.5
minutes. With ra_deg/dec_deg in the entry, resolve_target never asks SIMBAD.

A position is written only when SIMBAD agrees with a second, independent
source: NED (IPAC) within NED_TOL, or the pointing in our own light frames'
headers within FRAME_TOL (a framing offset is allowed for). Anything else is
listed for the owner and left alone. Each written entry records where its
position came from in "coord_source".
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

NED_TOL_ARCMIN = 2.0
FRAME_TOL_ARCMIN = 10.0


def verdict(simbad, ned, frames) -> tuple:
    """(write?, note) from separations in arcmin (None = that source had no answer). Pure."""
    if simbad is None:
        return False, "SIMBAD has no position"
    checks = []
    if ned is not None:
        checks.append(("NED", ned, NED_TOL_ARCMIN))
    if frames is not None:
        checks.append(("frames", frames, FRAME_TOL_ARCMIN))
    if not checks:
        return False, "no second source to check against"
    ok = [(n, d) for n, d, tol in checks if d <= tol]
    bad = [(n, d) for n, d, tol in checks if d > tol]
    note = ", ".join("%s %.1f'" % (n, d) for n, d, _t in checks)
    if bad and not ok:
        return False, "disagrees: " + note
    return True, note


def simbad_up(timeout=15.0) -> bool:
    try:
        socket.create_connection(("simbad.cds.unistra.fr", 443), timeout=timeout).close()
        from astroquery.simbad import Simbad
        return Simbad.query_object("M 1") is not None
    except Exception:  # noqa: BLE001
        return False


def frame_pointing(dso: str):
    """The pointing recorded in our first light frame of *dso*, or None."""
    from configs import config
    from scripts.moon_sky_cost import target_coord
    d = Path(config.data()["nina"]["image_dir"]) / dso.replace(" ", "")
    return target_coord(d) if d.is_dir() else None


def simbad_coord(name: str):
    """SIMBAD's position via its TAP service (astroquery.Simbad), not the Sesame
    name resolver: on 2026-10-27 Sesame hung while SIMBAD's TAP answered."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.simbad import Simbad
    try:
        t = Simbad.query_object(name)
        if t is None or not len(t):
            return None
        cols = {c.lower(): c for c in t.colnames}
        if "ra" in cols and "dec" in cols:
            return SkyCoord(float(t[cols["ra"]][0]) * u.deg, float(t[cols["dec"]][0]) * u.deg)
    except Exception:  # noqa: BLE001
        return None
    return None


def ned_coord(name: str):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.ipac.ned import Ned
    try:
        t = Ned.query_object(name)
        if len(t):
            return SkyCoord(float(t["RA"][0]) * u.deg, float(t["DEC"][0]) * u.deg)
    except Exception:  # noqa: BLE001 -- NED knows galaxies best; a nebula may be missing
        return None
    return None


def run(write: bool, log=print, statuses=("waiting",)) -> dict:
    from configs import config
    from iris_astronomy import astro_dso_visibility as v
    path = config.data()["location"]["instructions"]
    q = json.load(open(path))
    todo = [x for x in q if x.get("status") in statuses and x.get("ra_deg") is None]
    log("%d %s entries without coordinates" % (len(todo), "/".join(statuses)))
    results = {"written": [], "skipped": []}
    for x in todo:
        name = x["dso"]
        sc = simbad_coord(name)
        ned = ned_coord(name) if sc is not None else None
        fp = frame_pointing(name) if sc is not None else None
        d_ned = sc.separation(ned).arcmin if (sc is not None and ned is not None) else None
        d_fr = sc.separation(fp).arcmin if (sc is not None and fp is not None) else None
        ok, note = verdict(None if sc is None else 0.0, d_ned, d_fr)
        if ok:
            x["ra_deg"], x["dec_deg"] = round(float(sc.ra.deg), 5), round(float(sc.dec.deg), 5)
            x["coord_source"] = "SIMBAD %s; checked: %s" % (time.strftime("%Y-%m-%d"), note)
            results["written"].append((name, note))
            log("  %-10s RA %9.4f Dec %+8.4f  ok (%s)" % (name, x["ra_deg"], x["dec_deg"], note))
        else:
            results["skipped"].append((name, note))
            log("  %-10s NOT written: %s" % (name, note))
    if write and results["written"]:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(q, fh, indent=4)
        os.replace(tmp, path)
        log("wrote %d entries to %s" % (len(results["written"]), path))
    elif results["written"]:
        log("dry run: nothing written (use --write)")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--wait", action="store_true", help="wait for SIMBAD, then write")
    ap.add_argument("--all", action="store_true", help="completed entries too (the queue refresh resolves them)")
    a = ap.parse_args(argv)
    if a.wait:
        for i in range(72):                      # 12 h at 10 min
            if simbad_up():
                break
            print("SIMBAD still down (%s); next try in 10 min" % time.strftime("%H:%M"), flush=True)
            time.sleep(600)
        else:
            print("SIMBAD did not come back in 12 h; nothing written")
            return 1
        a.write = True
    elif not simbad_up():
        print("SIMBAD is down; try --wait")
        return 1
    r = run(a.write, statuses=("waiting", "completed") if a.all else ("waiting",))
    print("written %d, left for review %d" % (len(r["written"]), len(r["skipped"])))
    for n, why in r["skipped"]:
        print("  review: %s -- %s" % (n, why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
