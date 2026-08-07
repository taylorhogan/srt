#!/usr/bin/env python3
"""Render the live all-sky chart and push it straight to the web host.

Deliberately does NOT go through git. The site's markup is versioned because it
is authored; this image is a reading, regenerated every few minutes, and putting
it in git would mean a commit per frame and a repo that grows without bound. It
is scp'd into /srv/iris-live on the web host, which Caddy serves at /live/.

The marked target is whatever ``rank_targets_tonight`` currently puts first --
the same ranking the scheduler uses, so the chart cannot disagree with what the
observatory intends to image. That ranking already scores hours against the
measured tree line rather than altitude zero, which is why the chart draws the
tree line too: the brown region is not decoration, it is the constraint doing
the sorting.

Usage:  python scripts/live_skymap.py [--no-push]
"""
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from configs import config

HOST = "taylor@100.91.17.119"          # web host, over the tailnet
DEST = "/srv/iris-live"
KEY = str(Path.home() / ".ssh" / "id_ed25519_iris")

BG, FG, DIM = "#0d0d1a", "#e0e6ed", "#8b9bb4"
ACCENT, TARGET, TREES = "#3e64ff", "#f50057", "#f5a623"

ANCHORS = [
    ("Polaris", 37.95, 89.26), ("Vega", 279.23, 38.78),
    ("Deneb", 310.36, 45.28), ("Altair", 297.70, 8.87),
    ("Arcturus", 213.92, 19.18), ("Capella", 79.17, 46.00),
    ("Aldebaran", 68.98, 16.51), ("Regulus", 152.09, 11.97),
    ("Dubhe", 165.93, 61.75), ("Alkaid", 206.89, 49.31),
]


def _star_cache(root: Path) -> np.ndarray:
    """Naked-eye stars, fetched once and cached. Never re-fetched on a timer."""
    cache = root / "local" / "bright_stars.npy"
    if cache.exists():
        return np.load(cache)
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    v = Vizier(columns=["RAJ2000", "DEJ2000", "Vmag"], row_limit=-1)
    v.ROW_LIMIT = -1
    cat = v.query_constraints(catalog="V/50/catalog", Vmag="<6.0")[0]
    c = SkyCoord(ra=cat["RAJ2000"], dec=cat["DEJ2000"], unit=(u.hourangle, u.deg))
    arr = np.column_stack([c.ra.deg.astype(float), c.dec.deg.astype(float),
                           np.asarray(cat["Vmag"], dtype=float)])
    arr = arr[np.isfinite(arr).all(axis=1)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def _top_target(root: Path):
    """(name, ra_deg, dec_deg, good_hours) for the current best target, or None."""
    from iris_astronomy import astro_dso_visibility as adv
    from control import instructions as instr
    cfg = config.data()
    path = os.path.join(root, cfg["location"]["instructions"])
    try:
        rows, _dark, _wx = adv.rank_targets_tonight(path)
    except Exception:
        return None
    if not rows:
        return None
    name, good_hours = rows[0][0], rows[0][1]
    obj = instr.resolve_target_by_name(name)
    if obj is None:
        return None
    return name, float(obj.coord.ra.deg), float(obj.coord.dec.deg), int(good_hours)


def main() -> None:
    root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import astropy.units as u
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    from astropy.time import Time
    from iris_astronomy.astro_dso_visibility import map_az_to_horizon

    loc_cfg = config.data()["location"]
    loc = EarthLocation.from_geodetic(loc_cfg["longitude"] * u.deg,
                                      loc_cfg["latitude"] * u.deg,
                                      loc_cfg["elevation"] * u.m)
    now = Time.now()
    frame = AltAz(obstime=now, location=loc)

    # Called before our figure exists: it draws into the current figure as a
    # side effect, which would otherwise land on top of the chart.
    haz_raw, hal_raw = map_az_to_horizon()
    plt.close("all")
    order = np.argsort(np.array(haz_raw) % 360.0)
    haz = list(np.array(haz_raw)[order] % 360.0)
    hal = list(np.array(hal_raw)[order])
    haz, hal = haz + [haz[0] + 360.0], hal + [hal[0]]

    # ALSO before the figure: rank_targets_tonight calls map_az_to_horizon
    # internally, and that draws into whatever figure is current. Doing it after
    # plt.figure() painted stray horizon lines across the chart.
    tgt = _top_target(root)
    plt.close("all")

    stars = _star_cache(root)
    sc = SkyCoord(ra=stars[:, 0] * u.deg, dec=stars[:, 1] * u.deg).transform_to(frame)
    alt, az, mag = sc.alt.deg, sc.az.deg, stars[:, 2]
    up = alt > 0
    alt, az, mag = alt[up], az[up], mag[up]

    fig = plt.figure(figsize=(8.6, 8.6), facecolor=BG)
    ax = fig.add_subplot(111, projection="polar", facecolor="#080a14")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)      # East on the left: this is the view looking up
    ax.set_rlim(0, 90)

    hth, hr = np.radians(np.array(haz)), 90.0 - np.array(hal)
    ax.fill_between(hth, hr, 90.0, color=TREES, alpha=0.16, zorder=1)
    ax.plot(hth, hr, color=TREES, lw=1.2, alpha=0.65, zorder=2)

    ax.scatter(np.radians(az), 90.0 - alt,
               s=np.clip(28.0 * (6.2 - mag) ** 1.6 / 10.0, 0.6, 60.0),
               c=FG, alpha=np.clip(0.35 + 0.11 * (6.2 - mag), 0.3, 1.0),
               linewidths=0, zorder=3)

    for d in (60, 30):
        ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 90.0 - d),
                color=DIM, lw=0.6, alpha=0.3, zorder=2)

    for nm, ara, adec in ANCHORS:
        c = SkyCoord(ra=ara * u.deg, dec=adec * u.deg).transform_to(frame)
        if c.alt.deg <= 0:
            continue
        ax.scatter([np.radians(c.az.deg)], [90.0 - c.alt.deg], s=52,
                   facecolors="none", edgecolors=ACCENT, linewidths=0.9,
                   alpha=0.75, zorder=4)
        ax.text(np.radians(c.az.deg), 90.0 - c.alt.deg + 4.0, nm, color=ACCENT,
                fontsize=7.5, ha="center", va="bottom", alpha=0.85, zorder=5)

    status = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if tgt:
        name, ra, dec, good_hours = tgt
        tc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(frame)
        talt, taz = float(tc.alt.deg), float(tc.az.deg)
        tree_alt = float(np.interp(taz % 360.0, np.array(haz) % 360.0, np.array(hal)))
        if talt <= 0:
            state = "below the horizon"
        elif talt < tree_alt:
            state = "behind the trees"
        else:
            state = "observable now"
        status.update(target=name, ra_deg=round(ra, 4), dec_deg=round(dec, 4),
                      alt_deg=round(talt, 1), az_deg=round(taz, 1),
                      good_hours=good_hours, state=state)
        if talt > 0:
            tth, tr = np.radians(taz), 90.0 - talt
            ax.scatter([tth], [tr], s=340, facecolors="none", edgecolors=TARGET,
                       linewidths=2.0, zorder=6)
            ax.scatter([tth], [tr], s=26, c=TARGET, zorder=7)
            ax.annotate(name + "\nalt " + ("%.1f" % talt) + "°  az "
                        + ("%.1f" % taz) + "°\n" + state,
                        xy=(tth, tr), xytext=(tth + np.radians(22), 96),
                        color=TARGET, fontsize=9, ha="center", va="center",
                        zorder=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor=BG,
                                  edgecolor=TARGET, alpha=0.93, linewidth=1.0),
                        arrowprops=dict(arrowstyle="-", color=TARGET, lw=1.0,
                                        alpha=0.8, shrinkA=2, shrinkB=10))
    else:
        status["target"] = None
        status["state"] = "no target selected"

    lat, lon = loc_cfg["latitude"], loc_cfg["longitude"]
    # Two decimals (~1 km): enough to say which sky, not which house. This is
    # a public page.
    where = ("%.2f°%s  %.2f°%s"
             % (abs(lat), "N" if lat >= 0 else "S",
                abs(lon), "E" if lon >= 0 else "W"))
    ax.set_rticks([])
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                       color=FG, fontsize=11, fontweight="bold")
    ax.grid(color=DIM, alpha=0.14, lw=0.6)
    ax.spines["polar"].set_color(DIM)
    ax.spines["polar"].set_alpha(0.5)
    ax.set_title("The sky over " + where + "  ·  "
                 + now.to_datetime().strftime("%Y-%m-%d %H:%M UTC") + "\n"
                 + "zenith at centre, horizon at the rim, shaded region blocked by trees",
                 color=FG, fontsize=11, pad=18)

    out_dir = root / "iris_astronomy" / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = out_dir / "live_skymap.jpg"
    fig.savefig(img, format="jpeg", dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    js = out_dir / "live_status.json"
    js.write_text(json.dumps(status, indent=1))
    print("rendered", img.name, "->", status.get("target"), status.get("state"))

    if "--no-push" in sys.argv:
        return

    # Write to .tmp and rename on the far side, so the site never serves a
    # half-copied image.
    for src, dest in ((img, "skymap.jpg"), (js, "status.json")):
        subprocess.run(["scp", "-i", KEY, "-o", "BatchMode=yes", "-o",
                        "ConnectTimeout=15", str(src),
                        HOST + ":" + DEST + "/." + dest + ".tmp"], check=True)
        subprocess.run(["ssh", "-i", KEY, "-o", "BatchMode=yes", HOST,
                        "mv " + DEST + "/." + dest + ".tmp " + DEST + "/" + dest],
                       check=True)
    print("pushed to", HOST + ":" + DEST)


if __name__ == "__main__":
    main()
