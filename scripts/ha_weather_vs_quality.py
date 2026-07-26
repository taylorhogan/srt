"""Compare per-night Ha image quality vs weather for a DSO.

Quality comes from the frame_stats.json cache (FWHM/ecc/stars/sky); weather comes
from the per-frame FITS headers NINA writes (cloud, humidity, dewpoint, ambient
temp, wind, pressure), plus airmass/focuser temp. Buckets by observing night
(the NINA session-date folder) and prints a night-by-night table so we can see
what conditions went with the best Ha frames.
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from astropy.io import fits

DSO = sys.argv[1] if len(sys.argv) > 1 else "sh2-92"
FILT = sys.argv[2] if len(sys.argv) > 2 else "Ha"

base = Path(f"C:/Users/iriso/Documents/N.I.N.A/Targets/{DSO}")
stats = json.load(open(base / "frame_stats.json"))
qual = {r["path"]: r for r in stats}


def night_of(path: str) -> str:
    # .../cdk17/2026-07-11/LIGHT/xxx.fits -> 2026-07-11 (NINA session date)
    for part in Path(path).parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return "?"


WX = ["CLOUDCVR", "HUMIDITY", "DEWPOINT", "AMBTEMP", "WINDSPD", "PRESSURE", "AIRMASS", "FOCTEMP"]
by_night = defaultdict(lambda: defaultdict(list))

ha = [r for r in stats if r.get("filter") == FILT]
print(f"{DSO} {FILT}: {len(ha)} frames across nights\n", flush=True)
for r in ha:
    p = r["path"]
    n = night_of(p)
    by_night[n]["fwhm"].append(r["fwhm_arcsec"])
    by_night[n]["ecc"].append(r["eccentricity"])
    by_night[n]["stars"].append(r["star_count"])
    if r.get("sky_adu_per_s") is not None:
        by_night[n]["sky"].append(r["sky_adu_per_s"])
    try:
        h = fits.getheader(p)
    except Exception:
        continue
    for k in WX:
        v = h.get(k)
        if v is not None:
            by_night[n][k].append(float(v))
    # dew-point spread: ambient minus dewpoint (large = dry/no dew, good transparency)
    if h.get("AMBTEMP") is not None and h.get("DEWPOINT") is not None:
        by_night[n]["DEWSPREAD"].append(float(h["AMBTEMP"]) - float(h["DEWPOINT"]))


def med(xs):
    return st.median(xs) if xs else float("nan")


nights = sorted(by_night)
# Rank by median FWHM (lower = sharper = better)
ranked = sorted(nights, key=lambda n: med(by_night[n]["fwhm"]))

hdr = (f"{'night':<12}{'n':>4}{'FWHM':>7}{'ecc':>6}{'stars':>7}{'sky':>6}"
       f"{'cloud':>7}{'humid':>7}{'dewsprd':>8}{'ambT':>6}{'wind':>6}{'press':>7}{'airm':>6}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for n in nights:
    b = by_night[n]
    star = " *BEST" if n == ranked[0] else ("  <last" if n == nights[-1] else "")
    print(
        f"{n:<12}{len(b['fwhm']):>4}"
        f"{med(b['fwhm']):>7.2f}{med(b['ecc']):>6.2f}{med(b['stars']):>7.0f}{med(b['sky']):>6.2f}"
        f"{med(b['CLOUDCVR']):>7.0f}{med(b['HUMIDITY']):>7.0f}{med(b['DEWSPREAD']):>8.1f}"
        f"{med(b['AMBTEMP']):>6.1f}{med(b['WINDSPD']):>6.1f}{med(b['PRESSURE']):>7.0f}{med(b['AIRMASS']):>6.2f}"
        f"{star}", flush=True)

print(f"\nFWHM rank (sharpest first): {' < '.join(ranked)}", flush=True)
print("Columns: FWHM/ecc/stars from frame_stats; weather (cloud%/humid%/dew-spread degC/"
      "ambT/wind/pressure) from FITS headers. dewsprd = ambT - dewpoint (higher=drier).", flush=True)
