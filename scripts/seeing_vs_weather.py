"""Correlate a target's per-night median FWHM against the weather that night.

This is the analysis that set weather.SEEING_LEVEL_HPA to 850 hPa. Run it again
once there are more nights and re-check the thresholds in weather.seeing_from_wind
— nine nights is a thin calibration.

    python scripts/seeing_vs_weather.py sh2-92 [--plot out.png]

FWHM comes from <dso_dir>/frame_stats.json (the `stats` command's cache), so no
frames are re-measured. Weather comes from Open-Meteo, two endpoints:

  * surface fields   — archive-api (ERA5 reanalysis)
  * pressure levels  — historical-forecast-api (archived model runs). ERA5 on
    archive-api returns nulls for every *hPa field, which is why the upper air
    cannot come from the same call.

n is the number of NIGHTS, not frames, so it is small and p-values come from an
exact permutation test over all n! orderings. With ~20 variables tested, treat a
lone p < 0.05 as a lead rather than a result: the finding that mattered here was
that the correlation decays monotonically with altitude, which is a pattern noise
does not produce, not any single coefficient.
"""
import argparse
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import permutations
from math import factorial
from pathlib import Path

import numpy as np
import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

SURFACE = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
           "relative_humidity_2m", "temperature_2m", "dew_point_2m",
           "wind_speed_10m", "wind_gusts_10m"]
UPPER = ["wind_speed_200hPa", "wind_speed_250hPa", "wind_speed_300hPa",
         "wind_speed_500hPa", "wind_speed_700hPa", "wind_speed_850hPa",
         "temperature_500hPa", "temperature_850hPa"]

LEVELS = [("surface wind", "wind_speed_10m"), ("850 hPa (~1.5 km)", "wind_speed_850hPa"),
          ("700 hPa (~3 km)", "wind_speed_700hPa"), ("500 hPa (~5.5 km)", "wind_speed_500hPa"),
          ("300 hPa (~9 km)", "wind_speed_300hPa"),
          ("250 hPa JET STREAM", "wind_speed_250hPa"), ("200 hPa (~12 km)", "wind_speed_200hPa")]
OTHER = [("cloud cover", "cloud_cover"), ("cloud, low", "cloud_cover_low"),
         ("relative humidity 2m", "relative_humidity_2m"), ("dew-point spread", "dewpoint_spread"),
         ("surface gusts", "wind_gusts_10m"), ("temperature 2m", "temperature_2m"),
         ("850-500 lapse rate", "lapse_850_500"), ("airmass", "airmass"),
         ("star count", "stars")]


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rankdata(x) , rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def exact_p(x, y, perms):
    """Two-sided p from every permutation — n is small enough to enumerate."""
    rho = spearman(x, y)
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    null = (rx[perms] * ry).sum(axis=1) / denom
    return rho, float((np.abs(null) >= abs(rho) - 1e-12).mean())


def nights_from_cache(dso_dir: Path) -> dict:
    """Median FWHM (and friends) per night from the frame_stats cache."""
    rows = json.load(open(dso_dir / "frame_stats.json"))
    # Guard against the historical cross-contamination bug where another
    # target's frames were globbed into this cache.
    rows = [r for r in rows if dso_dir.name.lower() in r["path"].lower()]
    by_night = defaultdict(list)
    for r in rows:
        m = re.search(r"(\d{4}-\d{2}-\d{2})[\\/]LIGHT", r["path"])
        if m and r.get("fwhm_arcsec"):
            by_night[m.group(1)].append(r)
    out = {}
    for night, fr in by_night.items():
        times = sorted(datetime.fromisoformat(x["time"]) for x in fr)
        out[night] = {
            "n": len(fr),
            "fwhm": st.median(x["fwhm_arcsec"] for x in fr),
            "stars": st.median(x["star_count"] for x in fr if x.get("star_count")),
            "filter": ",".join(sorted({x["filter"] for x in fr})),
            "t0": times[0].isoformat(), "t1": times[-1].isoformat(),
        }
    return out


def fetch(url, lat, lon, start, end, fields):
    r = requests.get(url, params={"latitude": lat, "longitude": lon,
                                  "start_date": start, "end_date": end,
                                  "hourly": ",".join(fields), "timezone": "UTC",
                                  "wind_speed_unit": "kn"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data.get("reason"))
    return data["hourly"]


def window_mean(hourly, idx, t0, t1, var):
    t = datetime.fromisoformat(t0).replace(minute=0, second=0, microsecond=0)
    end = datetime.fromisoformat(t1).replace(minute=0, second=0, microsecond=0)
    vals = []
    while t <= end:
        i = idx.get(t.strftime("%Y-%m-%dT%H:00"))
        if i is not None and hourly.get(var) and hourly[var][i] is not None:
            vals.append(hourly[var][i])
        t += timedelta(hours=1)
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dso", nargs="?", default="sh2-92")
    ap.add_argument("--plot", help="write the 4-panel figure here")
    args = ap.parse_args()

    cfg = config.data()
    lat, lon = cfg["location"]["latitude"], cfg["location"]["longitude"]
    dso_dir = Path(cfg["nina"]["image_dir"]) / args.dso
    if not (dso_dir / "frame_stats.json").exists():
        print(f"no frame_stats.json under {dso_dir} — run `stats {args.dso}` first")
        return 1

    nights = nights_from_cache(dso_dir)
    if len(nights) < 4:
        print(f"only {len(nights)} nights — too few to correlate")
        return 1
    start = min(v["t0"][:10] for v in nights.values())
    end = max(v["t1"][:10] for v in nights.values())

    surf = fetch("https://archive-api.open-meteo.com/v1/archive", lat, lon, start, end, SURFACE)
    up = fetch("https://historical-forecast-api.open-meteo.com/v1/forecast",
               lat, lon, start, end, UPPER)
    si = {t: i for i, t in enumerate(surf["time"])}
    ui = {t: i for i, t in enumerate(up["time"])}

    for v in nights.values():
        for f in SURFACE:
            v[f] = window_mean(surf, si, v["t0"], v["t1"], f)
        for f in UPPER:
            v[f] = window_mean(up, ui, v["t0"], v["t1"], f)
        if v["temperature_2m"] is not None and v["dew_point_2m"] is not None:
            v["dewpoint_spread"] = v["temperature_2m"] - v["dew_point_2m"]
        if v["temperature_850hPa"] is not None and v["temperature_500hPa"] is not None:
            v["lapse_850_500"] = v["temperature_850hPa"] - v["temperature_500hPa"]
        v["airmass"] = None  # filled below if headers are readable

    # Airmass straight from the FITS headers — it is the one confounder that
    # would fake a weather correlation if the nights were shot at different
    # altitudes, so it is worth the header read to rule out.
    try:
        from astropy.io import fits
        rows = [r for r in json.load(open(dso_dir / "frame_stats.json"))
                if dso_dir.name.lower() in r["path"].lower()]
        am = defaultdict(list)
        for r in rows:
            m = re.search(r"(\d{4}-\d{2}-\d{2})[\\/]LIGHT", r["path"])
            if not m or m.group(1) not in nights:
                continue
            try:
                a = fits.getheader(r["path"]).get("AIRMASS")
            except Exception:
                continue
            if a:
                am[m.group(1)].append(float(a))
        for k, v in am.items():
            nights[k]["airmass"] = st.median(v)
    except Exception:
        pass

    ks = sorted(nights)
    fwhm = [nights[k]["fwhm"] for k in ks]
    print(f"{args.dso}: {len(ks)} nights, {sum(nights[k]['n'] for k in ks)} frames, "
          f"{start} -> {end}\n")
    print(f"{'night':12} {'n':>4} {'FWHM':>6} {'filter':>7} {'w850':>7} {'jet250':>7} "
          f"{'RH%':>5} {'cloud':>6}")
    for k in ks:
        v = nights[k]
        g = lambda key: f"{v[key]:7.1f}" if v.get(key) is not None else "    n/a"
        print(f"{k:12} {v['n']:4} {v['fwhm']:6.2f} {v['filter']:>7} {g('wind_speed_850hPa')} "
              f"{g('wind_speed_250hPa')} {g('relative_humidity_2m')[:5]:>5} {g('cloud_cover')}")

    perms = np.array(list(permutations(range(len(ks)))))
    print(f"\nSpearman rho vs median FWHM  (n={len(ks)}, exact permutation p over "
          f"{factorial(len(ks)):,} orderings)")
    print("\n  -- wind by altitude: the shape of this column is the finding --")
    for label, key in LEVELS:
        vals = [nights[k].get(key) for k in ks]
        if any(v is None for v in vals):
            print(f"    {label:24}   -- missing --")
            continue
        rho, p = exact_p(fwhm, vals, perms)
        print(f"    {label:24} {rho:+6.2f}  p={p:.4f}")
    print("\n  -- everything else --")
    for label, key in OTHER:
        vals = [nights[k].get(key) for k in ks]
        if any(v is None for v in vals):
            print(f"    {label:24}   -- missing --")
            continue
        rho, p = exact_p(fwhm, vals, perms)
        print(f"    {label:24} {rho:+6.2f}  p={p:.4f}")

    print("\n  NOTE: ~16 variables tested. Bonferroni for 0.05 across them is "
          f"p < {0.05/16:.4f}.")
    if args.plot:
        make_plot(nights, ks, args.plot)
        print(f"\nwrote {args.plot}")
    return 0


def make_plot(nights, ks, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    BG, PANEL, INK, MUTED = "#0d0d1a", "#1a1a2e", "#f2f2f7", "#9aa0b4"
    BLUE, GOLD, GRID = "#3d8fd6", "#e8b84b", "#3a3a55"
    g = lambda key: np.array([nights[k][key] for k in ks], float)
    fwhm = g("fwhm")

    fig = Figure(figsize=(15, 10.5))
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(BG)

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=11)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
        ax.set_title(title, color=INK, fontsize=13, pad=12, loc="left")
        for s in ax.spines.values():
            s.set_edgecolor(GRID)
        ax.grid(True, alpha=0.25, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    heights = [0.1, 1.5, 3.0, 5.5, 9.0, 10.4, 12.0]
    ax = fig.add_subplot(2, 2, 1)
    ax.axvline(0, color=MUTED, linewidth=1)
    for (label, key), y in zip(LEVELS, heights):
        r = spearman(fwhm, g(key))
        c = GOLD if "JET" in label else BLUE
        ax.plot([0, r], [y, y], color=c, linewidth=2.5, solid_capstyle="round", alpha=0.85)
        ax.plot([r], [y], "o", color=c, markersize=11)
        ax.annotate(f"{label}   ρ={r:+.2f}", (max(r, 0.0), y), textcoords="offset points",
                    xytext=(14, 0), ha="left", va="center",
                    color=INK if c == GOLD else MUTED, fontsize=10,
                    fontweight="bold" if c == GOLD else "normal")
    style(ax, "A · Where the seeing signal actually lives",
          "Spearman ρ  of wind speed vs median FWHM", "height")
    ax.set_yticks(heights)
    ax.set_yticklabels(["0.1", "1.5", "3", "5.5", "9", "10.4", "12 km"])
    ax.set_xlim(-0.45, 1.42)
    ax.invert_yaxis()

    def scatter(pos, key, title, xlabel, colour=BLUE):
        ax = fig.add_subplot(2, 2, pos)
        x = g(key)
        ax.plot(x, fwhm, "o", color=colour, markersize=11, markeredgecolor=PANEL,
                markeredgewidth=2, zorder=3)
        if len(set(x)) > 1:
            b, a = np.polyfit(x, fwhm, 1)
            xr = np.linspace(x.min(), x.max(), 50)
            ax.plot(xr, b * xr + a, "--", color=colour, alpha=0.45, linewidth=1.5)
        placed, xr_ = [], (x.max() - x.min()) or 1.0
        for xi, yi, k in zip(x, fwhm, ks):
            below = any(abs(xi - px) / xr_ < 0.09 and abs(yi - py) < 0.20
                        for px, py in placed)
            ax.annotate(k[5:], (xi, yi), textcoords="offset points",
                        xytext=(0, -20 if below else 13), ha="center",
                        color=MUTED, fontsize=9)
            placed.append((xi, yi))
        style(ax, f"{title}    ρ = {spearman(fwhm, x):+.2f}", xlabel,
              "median FWHM  (arcsec)")

    scatter(2, "wind_speed_850hPa", "B · 850 hPa wind", "mean 850 hPa wind (knots)")
    scatter(3, "relative_humidity_2m", "C · Relative humidity",
            "mean relative humidity (%)", GOLD)
    scatter(4, "cloud_cover", "D · Cloud cover", "mean cloud cover (%)")
    fig.suptitle(f"seeing vs weather — {len(ks)} nights", color=INK, fontsize=15,
                 x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=130, facecolor=BG)


if __name__ == "__main__":
    raise SystemExit(main())
