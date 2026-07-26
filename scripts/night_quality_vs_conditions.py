"""Per-night image quality vs atmospheric conditions for a DSO.

Extends scripts/ha_weather_vs_quality.py in two ways:
  1. Adds the conditions that are *not* in the FITS headers -- US AQI, PM2.5,
     aerosol optical depth and 250 hPa jet-stream wind -- pulled from Open-Meteo's
     historical hourly APIs and joined to each frame by its observation hour.
  2. Writes a nightly and a per-frame CSV, then runs Spearman correlations of
     quality (FWHM, star count, eccentricity, sky) against every condition.

Filter matters: star count and FWHM are not comparable between e.g. Ha and O-III,
and this rig shoots one filter per night, so filter is confounded with night.
Correlations are therefore run on values z-scored *within* each filter, and the
per-filter breakdown is printed so the confound stays visible.

Usage:  python scripts/night_quality_vs_conditions.py [DSO] [--out DIR]
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytz
import requests
from astropy.io import fits
from scipy import stats

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from fits_processing import sky_pedestal as sp

LOCAL_TZ = pytz.timezone("America/New_York")

# Conditions read straight from the FITS headers N.I.N.A writes (on-site sensors).
HDR_WX = {
    "cloud_pct":    "CLOUDCVR",
    "humidity_pct": "HUMIDITY",
    "dewpoint_c":   "DEWPOINT",
    "ambtemp_c":    "AMBTEMP",
    "windspd_ms":   "WINDSPD",
    "pressure_hpa": "PRESSURE",
    "airmass":      "AIRMASS",
    "foctemp_c":    "FOCTEMP",
}

# Conditions fetched from Open-Meteo (no on-site sensor for these).
# NOTE: us_aqi/pm2_5/aod come from the CAMS *model*, which validates badly for
# smoke here -- see fetch_airnow_pm25. pm25_epa is the measured ground truth and
# is the column to trust for aerosol.
API_WX = ["us_aqi", "pm2_5", "aod", "jet250_kmh", "pm25_epa"]

QUALITY = ["fwhm_arcsec", "star_count", "eccentricity", "sky_adu_per_s"]

# Lower is better for these; used only to phrase the correlation write-up.
LOWER_IS_BETTER = {"fwhm_arcsec", "eccentricity", "sky_adu_per_s"}


def night_of(path: str) -> str:
    """NINA session-date folder, e.g. .../cdk17/2026-07-11/LIGHT/x.fits -> 2026-07-11."""
    for part in Path(path).parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return "?"


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _fetch(url: str, params: dict, keys: list[str]) -> dict[str, dict]:
    """GET an Open-Meteo hourly endpoint -> {local_hour_iso: {key: value}}."""
    out: dict[str, dict] = {}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        times = data["hourly"]["time"]
        for i, t in enumerate(times):
            out[t] = {k: data["hourly"].get(k, [None] * len(times))[i] for k in keys}
    except requests.RequestException as e:
        print(f"  ! fetch failed ({url.rsplit('/', 1)[-1]}): {e}")
    return out


def fetch_conditions(lat: float, lon: float, start: str, end: str) -> dict[str, dict]:
    """Hourly AQI / PM2.5 / AOD / 250 hPa wind, keyed by local-time ISO hour."""
    print(f"Fetching Open-Meteo hourly conditions {start} -> {end} ...")
    aq = _fetch(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        {"latitude": lat, "longitude": lon,
         "hourly": ["us_aqi", "pm2_5", "aerosol_optical_depth"],
         "start_date": start, "end_date": end, "timezone": "auto"},
        ["us_aqi", "pm2_5", "aerosol_optical_depth"],
    )
    jet = _fetch(
        "https://api.open-meteo.com/v1/forecast",
        {"latitude": lat, "longitude": lon,
         "hourly": ["wind_speed_250hPa"],
         "start_date": start, "end_date": end, "timezone": "auto"},
        ["wind_speed_250hPa"],
    )

    merged: dict[str, dict] = {}
    for t in set(aq) | set(jet):
        a = aq.get(t, {})
        merged[t] = {
            "us_aqi":     a.get("us_aqi"),
            "pm2_5":      a.get("pm2_5"),
            "aod":        a.get("aerosol_optical_depth"),
            "jet250_kmh": jet.get(t, {}).get("wind_speed_250hPa"),
        }
    print(f"  got {len(merged)} hourly rows "
          f"({sum(1 for v in merged.values() if v['us_aqi'] is not None)} with AQI, "
          f"{sum(1 for v in merged.values() if v['jet250_kmh'] is not None)} with jet wind)")
    return merged


def fetch_airnow_pm25(utc_hours: set[str], state: str = "Connecticut") -> dict[str, float]:
    """Measured PM2.5 (ug/m3) from EPA AirNow ground stations, by UTC hour.

    Open-Meteo's pm2_5/us_aqi come from the CAMS global model at ~40 km, and
    validated against these stations for July 2026 it both under-called the real
    smoke event (peak 75 modelled vs 121 measured, lagged ~1 day) and ran ~2x
    high on clean nights -- enough to invert the ranking between nights. These
    files are the actual regulatory monitors and need no API key.

    Takes UTC hour strings ("2026-07-12T02"), returns {hour: max PM2.5 across
    that state's stations}. Uses the max because a single plume can sit over one
    monitor, and we want the worst air anywhere nearby.
    """
    hours = sorted(utc_hours)
    print(f"Fetching AirNow measured PM2.5 for {len(hours)} hours ...")

    def grab(h: str):
        day, hh = h[:10].replace("-", ""), h[11:13]
        url = f"https://files.airnowtech.org/airnow/{h[:4]}/{day}/HourlyData_{day}{hh}.dat"
        try:
            r = requests.get(url, timeout=25)
            if r.status_code != 200:
                return h, None
            vals = []
            for line in r.text.splitlines():
                f = line.split("|")
                if len(f) > 7 and f[5] == "PM2.5" and state in f[-1]:
                    try:
                        vals.append(float(f[7]))
                    except ValueError:
                        pass
            return h, (max(vals) if vals else None)
        except requests.RequestException:
            return h, None

    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for h, v in pool.map(grab, hours):
            if v is not None:
                out[h] = v
    print(f"  got {len(out)}/{len(hours)} hours with measured PM2.5")
    return out


def build_frames(dso: str) -> list[dict]:
    """One row per light frame: quality + on-site weather + API conditions."""
    base = Path(f"C:/Users/iriso/Documents/N.I.N.A/Targets/{dso}")
    stats_path = base / "frame_stats.json"
    raw = json.load(open(stats_path))
    print(f"{dso}: {len(raw)} frames in {stats_path.name}")

    rows: list[dict] = []
    for r in raw:
        # DATE-OBS (and frame_stats "time") is UTC; conditions are keyed local.
        obs_utc = datetime.fromisoformat(r["time"]).replace(tzinfo=pytz.UTC)
        obs_local = obs_utc.astimezone(LOCAL_TZ)
        row = {
            "path":      r["path"],
            "night":     night_of(r["path"]),
            "utc":       obs_utc.strftime("%Y-%m-%dT%H:%M:%S"),
            "local":     obs_local.strftime("%Y-%m-%dT%H:%M:%S"),
            "hour_key":  obs_local.strftime("%Y-%m-%dT%H:00"),
            "utc_hour":  obs_utc.strftime("%Y-%m-%dT%H"),
            "filter":    r.get("filter"),
            "fwhm_arcsec":   r.get("fwhm_arcsec"),
            "eccentricity":  r.get("eccentricity"),
            "star_count":    r.get("star_count"),
            "sky_adu_per_s": r.get("sky_adu_per_s"),
        }
        try:
            h = fits.getheader(r["path"])
            for name, key in HDR_WX.items():
                v = h.get(key)
                row[name] = float(v) if v is not None else None
        except Exception:
            for name in HDR_WX:
                row.setdefault(name, None)

        if row.get("ambtemp_c") is not None and row.get("dewpoint_c") is not None:
            # Dew-point spread: high = dry air = better transparency, no dewing.
            row["dewspread_c"] = round(row["ambtemp_c"] - row["dewpoint_c"], 2)
        else:
            row["dewspread_c"] = None
        rows.append(row)

    # frame_stats.json caches sky_adu_per_s from before the pedestal fix, where it
    # was ~96% bias pedestal. Recompute it here from the corners (cheap -- no star
    # detection) so the table reflects the corrected pipeline without waiting for
    # the whole cache to be rebuilt.
    print("Recomputing sky brightness through the pedestal-corrected pipeline ...")

    def _sky(row: dict):
        lvl = sp.corner_level(row["path"])
        if lvl is None:
            return row, None, None
        try:
            h = fits.getheader(row["path"])
            exp = float(h.get("EXPTIME", h.get("EXPOSURE", 1.0)))
            ped = sp.lookup(h.get("GAIN"), h.get("OFFSET"), h.get("CCD-TEMP"), exp)
        except Exception:
            return row, None, None
        if ped is None:
            return row, None, None
        return row, max(lvl - ped["pedestal_adu"], 0.0) / max(exp, 1e-6), ped["extrapolated"]

    n_extrap = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for row, val, extrap in pool.map(_sky, rows):
            row["sky_adu_per_s"] = round(val, 6) if val is not None else None
            row["sky_pedestal_extrapolated"] = extrap
            if extrap:
                n_extrap += 1
    n_ok = sum(1 for r in rows if r["sky_adu_per_s"] is not None)
    print(f"  {n_ok}/{len(rows)} frames measured "
          f"({n_extrap} with an extrapolated pedestal)")

    rows.sort(key=lambda d: d["utc"])

    nights = sorted({r["night"] for r in rows})
    cfg = config.data()["location"]
    # Pad by a day either side so post-midnight frames always find their hour.
    start = (datetime.fromisoformat(nights[0]) - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.fromisoformat(nights[-1]) + timedelta(days=1)).strftime("%Y-%m-%d")
    cond = fetch_conditions(cfg["latitude"], cfg["longitude"], start, end)

    epa = fetch_airnow_pm25({r["utc_hour"] for r in rows})

    missed = 0
    for row in rows:
        c = cond.get(row["hour_key"])
        if c is None:
            missed += 1
            c = {}
        for k in API_WX:
            row[k] = c.get(k)
        row["pm25_epa"] = epa.get(row["utc_hour"])
    if missed:
        print(f"  ! {missed} frames had no matching condition hour")
    return rows


def nightly_table(rows: list[dict]) -> list[dict]:
    """Collapse frames to one median row per night."""
    by_night: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_night[r["night"]].append(r)

    cols = QUALITY + list(HDR_WX) + ["dewspread_c"] + API_WX
    out = []
    for night in sorted(by_night):
        frames = by_night[night]
        filters = sorted({f["filter"] for f in frames if f["filter"]})
        rec = {
            "night":   night,
            "frames":  len(frames),
            "filter":  "+".join(filters),
            "start_local": min(f["local"] for f in frames)[11:16],
            "end_local":   max(f["local"] for f in frames)[11:16],
        }
        for c in cols:
            v = med([f.get(c) for f in frames])
            rec[c] = round(v, 3) if isinstance(v, float) else v
        out.append(rec)
    return out


def _zscore_within_filter(rows: list[dict], key: str) -> dict[int, float]:
    """Index -> value z-scored against other frames shot through the same filter.

    Removes the constant per-filter offset in star count / FWHM so frames from
    Ha and O-III nights can be pooled in one correlation.
    """
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is not None:
            groups[r["filter"]].append((i, float(v)))
    z: dict[int, float] = {}
    for _, items in groups.items():
        vals = np.array([v for _, v in items])
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        if sd == 0:
            continue
        mu = vals.mean()
        for (i, v) in items:
            z[i] = (v - mu) / sd
    return z


def _within_night(rows: list[dict], key: str) -> dict[int, float]:
    """Index -> value minus its own night's median.

    Isolates drift *inside* a night from differences *between* nights. A pooled
    correlation mixes the two, so a condition that only varies night-to-night can
    look significant on 249 frames when it really has 7 independent data points.
    """
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is not None:
            groups[r["night"]].append((i, float(v)))
    out: dict[int, float] = {}
    for _, items in groups.items():
        m = st.median([v for _, v in items])
        for (i, v) in items:
            out[i] = v - m
    return out


def correlate(rows: list[dict], nights: list[dict]) -> list[dict]:
    """Spearman rho of each quality metric vs each condition, at three levels."""
    conds = list(HDR_WX) + ["dewspread_c"] + API_WX
    results = []

    for q in QUALITY:
        zq = _zscore_within_filter(rows, q)
        wq = _within_night(rows, q)
        for c in conds:
            # --- per-frame (within-filter z-scored, n ~ 249) ---
            xs, ys = [], []
            for i, r in enumerate(rows):
                if i in zq and r.get(c) is not None:
                    xs.append(float(r[c]))
                    ys.append(zq[i])
            if len(xs) >= 10 and len(set(xs)) > 2:
                rho, p = stats.spearmanr(xs, ys)
                n_f = len(xs)
            else:
                rho, p, n_f = float("nan"), float("nan"), len(xs)

            # --- within-night (both sides de-medianed by night) ---
            wc = _within_night(rows, c)
            wxs = [wc[i] for i in range(len(rows)) if i in wq and i in wc]
            wys = [wq[i] for i in range(len(rows)) if i in wq and i in wc]
            if len(wxs) >= 10 and len(set(wxs)) > 2:
                wrho, wp = stats.spearmanr(wxs, wys)
            else:
                wrho, wp = float("nan"), float("nan")

            # --- per-night medians (n = number of nights) ---
            nx = [n[c] for n in nights if n.get(c) is not None and n.get(q) is not None]
            ny = [n[q] for n in nights if n.get(c) is not None and n.get(q) is not None]
            if len(nx) >= 4 and len(set(nx)) > 2:
                nrho, npv = stats.spearmanr(nx, ny)
            else:
                nrho, npv = float("nan"), float("nan")

            results.append({
                "quality": q, "condition": c,
                "frame_rho": rho, "frame_p": p, "frame_n": n_f,
                "within_night_rho": wrho, "within_night_p": wp,
                "night_rho": nrho, "night_p": npv, "night_n": len(nx),
            })
    return results


def write_report(path: Path, dso: str, rows: list[dict], nights: list[dict],
                 res: list[dict]) -> None:
    """Markdown summary: the nightly table plus the correlations that survive."""
    L: list[str] = []
    L.append(f"# {dso}: image quality vs atmospheric conditions\n")
    L.append(f"{len(rows)} frames (300 s subs) over {len(nights)} nights, "
             f"{nights[0]['night']} to {nights[-1]['night']}.\n")
    L.append("Quality from `frame_stats.json`; on-site weather from the N.I.N.A FITS "
             "headers; AQI / PM2.5 / AOD / 250 hPa jet wind from Open-Meteo, joined "
             "to each frame by observation hour.\n")

    L.append("\n## Nightly medians\n")
    cols = [("night", "night", "s"), ("filter", "filt", "s"), ("frames", "n", "d"),
            ("fwhm_arcsec", "FWHM\"", ".2f"), ("eccentricity", "ecc", ".2f"),
            ("star_count", "stars", ".0f"), ("sky_adu_per_s", "sky ADU/s", ".5f"),
            ("pm25_epa", "PM2.5 measured", ".1f"),
            ("us_aqi", "AQI~", ".0f"), ("pm2_5", "PM2.5~", ".1f"), ("aod", "AOD~", ".2f"),
            ("jet250_kmh", "jet250", ".0f"), ("humidity_pct", "humid%", ".0f"),
            ("dewspread_c", "dewspr", ".1f"), ("ambtemp_c", "ambT", ".1f"),
            ("cloud_pct", "cloud%", ".0f"), ("windspd_ms", "wind", ".1f"),
            ("pressure_hpa", "press", ".0f"), ("airmass", "airm", ".2f")]
    L.append("| " + " | ".join(h for _, h, _ in cols) + " |")
    L.append("|" + "|".join("---" for _ in cols) + "|")
    for n in nights:
        cells = []
        for key, _, spec in cols:
            v = n.get(key)
            cells.append("--" if v is None else (str(v) if spec == "s" else format(v, spec)))
        L.append("| " + " | ".join(cells) + " |")

    L.append("\nUnits: FWHM arcsec, PM2.5 ug/m3, jet250 km/h at 250 hPa, dewspr = ambient "
             "- dewpoint degC (higher = drier), wind m/s, press hPa.\n")
    L.append("`PM2.5 measured` is EPA/CT-DEEP ground stations. Columns marked `~` are "
             "CAMS *model* output from Open-Meteo and are kept only for comparison -- "
             "see Caveats, they do not reproduce the measurements.\n")

    L.append("\n## Correlations (Spearman rho)\n")
    L.append("Three levels, because they answer different questions:\n")
    L.append("- **pooled** - all frames, z-scored within filter. Mixes between- and "
             "within-night variation and treats non-independent frames as independent, "
             "so its p-value is optimistic. A hint, not evidence.")
    L.append("- **within** - both sides de-medianed by night: does the condition track "
             "quality as it drifts during a night?")
    L.append(f"- **between** - night medians, n={len(nights)}. Honest but low power; "
             "needs |rho| > 0.79 for p < 0.05.\n")

    for q in QUALITY:
        block = [r for r in res if r["quality"] == q]
        block.sort(key=lambda r: -abs(r["frame_rho"]) if r["frame_rho"] == r["frame_rho"] else 0)
        L.append(f"\n### {q}\n")
        L.append("| condition | pooled | p | within | p | between | p |")
        L.append("|---|---|---|---|---|---|---|")
        for r in block:
            def g(v, spec=".3f"):
                return "--" if v != v else format(v, spec)
            L.append(f"| {r['condition']} | {g(r['frame_rho'])} | {g(r['frame_p'], '.2g')} "
                     f"| {g(r['within_night_rho'])} | {g(r['within_night_p'], '.2g')} "
                     f"| {g(r['night_rho'])} | {g(r['night_p'], '.2g')} |")

    L.append("\n## Why the modelled air-quality columns are not trusted\n")
    L.append("Validated against EPA/CT-DEEP ground stations, daily max PM2.5, "
             "2026-07-11 to 07-25 (n=15): Spearman rho 0.58, but the model/measured "
             "ratio swings from 0.24 to 1.70.\n")
    L.append("| period | modelled PM2.5 | measured PM2.5 | model/measured |")
    L.append("|---|---|---|---|")
    L.append("| 07-15 smoke | 24.9 | 102.1 | 0.24 |")
    L.append("| 07-16 smoke | 35.5 | 131.2 | 0.27 |")
    L.append("| 07-17 smoke | 51.2 | 128.6 | 0.40 |")
    L.append("| 07-23 clean | 14.4 | 9.3 | 1.55 |")
    L.append("| 07-24 clean | 21.3 | 12.5 | 1.70 |")
    L.append("\nCAMS under-called the real smoke event by 2.5-4x and runs high on clean "
             "nights, compressing everything toward 10-40 ug/m3 and destroying exactly "
             "the dynamic range that matters. Use `pm25_epa`.\n")
    L.append("Separately, `us_aqi` is the max over all pollutant sub-indices, and on "
             "2026-07-11, 07-13, 07-14, 07-25 and 07-26 it was driven by **ozone**, not "
             "PM2.5. Ozone does not scatter starlight, so `us_aqi` is the wrong variable "
             "for a smoke/transparency gate -- `us_aqi_pm2_5` or measured PM2.5 is right.\n")

    L.append("\n## Caveats\n")
    sky_vals = sorted({r["sky_adu_per_s"] for r in rows if r["sky_adu_per_s"] is not None})
    wind_vals = sorted({r["windspd_ms"] for r in rows if r["windspd_ms"] is not None})
    n_extrap = sum(1 for r in rows if r.get("sky_pedestal_extrapolated"))
    L.append(f"- `sky_adu_per_s` here is **recomputed live** through the "
             "pedestal-corrected pipeline, not read from `frame_stats.json` - that "
             "cache still holds pre-fix values where the number was ~96% bias "
             "pedestal (0.480-0.500 ADU/s across every night, a 4% spread). Corrected, "
             f"these nights span {min(sky_vals)}-{max(sky_vals)} ADU/s.")
    if n_extrap:
        L.append(f"- {n_extrap}/{len(rows)} frames used an **extrapolated** pedestal: "
                 "the only BIAS/DARK set on disk was shot at -10 degC but these lights "
                 "ran at -13 and -20 degC, so the dark term was scaled by the standard "
                 "2x-per-6.3-degC rule. Bias dominates the pedestal and is "
                 "temperature-independent, so the error is small in absolute terms "
                 "(~1 ADU) but the real sky signal is only a few ADU - shoot BIAS/DARK "
                 "at every temperature you image at before leaning on these values.")
    L.append(f"- `windspd_ms` has only {len(wind_vals)} distinct values across all "
             "frames, so wind correlations are weak by construction.")
    L.append("- One filter per night on this rig, so filter is confounded with night. "
             "Pooled correlations z-score within filter to compensate; the between-night "
             "numbers cannot separate the two at all.")
    L.append("- Humidity and dew spread are the same axis (rho = -0.99), and ambient "
             "temperature is strongly tied to both. They cannot be told apart with "
             f"{len(nights)} nights.")
    epa_vals = [r["pm25_epa"] for r in rows if r.get("pm25_epa") is not None]
    if epa_vals:
        L.append(f"- Measured PM2.5 spans only {min(epa_vals):.1f}-{max(epa_vals):.1f} "
                 "ug/m3 across these nights -- all clean air. The real smoke event "
                 "(07-15 to 07-19, measured peak 131) falls entirely in the gap in the "
                 "imaging record, so **whether smoke hurts image quality is untested "
                 "here**, not disproven.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main() -> None:
    dso = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "sh2-92"
    out_dir = Path(os.path.expanduser("~/Desktop"))
    if "--out" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out") + 1])

    rows = build_frames(dso)
    nights = nightly_table(rows)

    # ---------------- nightly table ----------------
    hdr = (f"{'night':<12}{'filt':>6}{'n':>4}{'FWHM':>7}{'ecc':>6}{'stars':>7}{'sky':>9}"
           f"{'PM2.5*':>8}{'AQI~':>6}{'PM2.5~':>8}{'AOD~':>6}{'jet250':>8}"
           f"{'humid':>7}{'dewsp':>7}{'ambT':>6}{'cloud':>7}{'wind':>6}{'press':>7}{'airm':>6}")
    print(f"\n{dso}: {len(rows)} frames over {len(nights)} nights\n")
    print(hdr)
    print("-" * len(hdr))

    def f(v, spec):
        return format(v, spec) if v is not None else "  --"

    for n in nights:
        print(f"{n['night']:<12}{n['filter']:>6}{n['frames']:>4}"
              f"{f(n['fwhm_arcsec'], '>7.2f')}{f(n['eccentricity'], '>6.2f')}"
              f"{f(n['star_count'], '>7.0f')}{f(n['sky_adu_per_s'], '>9.5f')}"
              f"{f(n['pm25_epa'], '>8.1f')}"
              f"{f(n['us_aqi'], '>6.0f')}{f(n['pm2_5'], '>8.1f')}{f(n['aod'], '>6.2f')}"
              f"{f(n['jet250_kmh'], '>8.1f')}"
              f"{f(n['humidity_pct'], '>7.0f')}{f(n['dewspread_c'], '>7.1f')}"
              f"{f(n['ambtemp_c'], '>6.1f')}{f(n['cloud_pct'], '>7.0f')}"
              f"{f(n['windspd_ms'], '>6.1f')}{f(n['pressure_hpa'], '>7.0f')}"
              f"{f(n['airmass'], '>6.2f')}")

    night_fields = (["night", "filter", "frames", "start_local", "end_local"]
                    + QUALITY + API_WX + list(HDR_WX) + ["dewspread_c"])
    frame_fields = (["night", "local", "utc", "filter"] + QUALITY + API_WX
                    + list(HDR_WX) + ["dewspread_c"])

    print()
    write_csv(out_dir / f"{dso}_nightly_conditions.csv", nights, night_fields)
    write_csv(out_dir / f"{dso}_frame_conditions.csv", rows, frame_fields)

    # ---------------- correlations ----------------
    res = correlate(rows, nights)
    write_csv(out_dir / f"{dso}_correlations.csv", res,
              ["quality", "condition", "frame_rho", "frame_p", "frame_n",
               "within_night_rho", "within_night_p", "night_rho", "night_p", "night_n"])

    print("\nSpearman rho -- quality vs condition, at three levels")
    print("  pooled  : all frames, z-scored within filter (n~frames). Mixes between-night")
    print("            and within-night variation, and frames in a night are not")
    print("            independent, so its p-value is optimistic -- read it as a hint.")
    print("  within  : both sides de-medianed by night. Does the condition track quality")
    print("            as it drifts *during* a night?")
    print(f"  between : per-night medians, n={len(nights)} nights. Honest, but low power --")
    print("            needs |rho|>0.79 to reach p<0.05.\n")
    ch = (f"{'quality':<15}{'condition':<14}{'pooled':>8}{'p':>9}"
          f"{'within':>9}{'p':>9}{'between':>9}{'p':>8}")
    print(ch)
    print("-" * len(ch))
    for q in QUALITY:
        block = [r for r in res if r["quality"] == q]
        block.sort(key=lambda r: -abs(r["frame_rho"]) if r["frame_rho"] == r["frame_rho"] else 0)
        for r in block:
            print(f"{r['quality']:<15}{r['condition']:<14}"
                  f"{r['frame_rho']:>8.3f}{r['frame_p']:>9.2g}"
                  f"{r['within_night_rho']:>9.3f}{r['within_night_p']:>9.2g}"
                  f"{r['night_rho']:>9.3f}{r['night_p']:>8.2g}")
        print()

    write_report(out_dir / f"{dso}_conditions_report.md", dso, rows, nights, res)

    # ---------------- per-filter sanity ----------------
    print("Per-filter medians (shows the filter/night confound):")
    by_filt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_filt[r["filter"]].append(r)
    for filt, fr in sorted(by_filt.items()):
        nn = sorted({x['night'] for x in fr})
        print(f"  {filt:<6} n={len(fr):<4} FWHM {med([x['fwhm_arcsec'] for x in fr]):.2f}\"  "
              f"stars {med([x['star_count'] for x in fr]):.0f}  "
              f"ecc {med([x['eccentricity'] for x in fr]):.2f}  "
              f"nights {', '.join(nn)}")


if __name__ == "__main__":
    main()
