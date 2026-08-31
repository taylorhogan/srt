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
import sys
import warnings
from datetime import datetime, timedelta, timezone
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

# Transport (host/dest/key) lives in scripts/live_push.py -- the ONE copy.
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
    """Naked-eye stars, fetched once and cached. Never re-fetched on a timer.

    Thin wrapper kept for readability at the call site; the cache itself is
    shared with the plate solver, which has to agree with this chart about
    which stars are up.
    """
    from iris_astronomy import bright_stars
    return bright_stars.catalogue(root)


# The five that are naked-eye objects. Uranus at ~5.7 is technically within the
# chart's V<6 star limit but is not something anyone plans around, and Neptune
# at ~7.8 is not visible at all -- listing them would add clutter, not
# information. Colours are roughly the eye impression, so the chart can be read
# without depending on the labels.
PLANETS = [
    ("Mercury", "#c8c2b4"), ("Venus", "#fff4d6"), ("Mars", "#ff7043"),
    ("Jupiter", "#ffe0a3"), ("Saturn", "#e8d7a0"),
]


def _draw_solar_system(ax, fig, frame, loc):
    """Mark the sun, moon and naked-eye planets, wherever they are up.

    Drawn only above the horizon, so the chart never claims a body is somewhere
    it cannot be seen. The moon carries its illuminated fraction, which is the
    number that decides whether a narrowband night is worth running.

    Worth knowing when reading this against the sky camera: the camera sees only
    the upper half of the sky, and at this latitude the ecliptic runs low across
    the south, so planets shown here are often outside the camera's field.
    """
    import astropy.units as u
    from astropy.coordinates import get_body


    now = frame.obstime
    out = []
    try:
        sun = get_body("sun", now, loc).transform_to(frame)
        moon = get_body("moon", now, loc).transform_to(frame)
        elong = get_body("sun", now, loc).separation(get_body("moon", now, loc)).deg
    except Exception:
        return out
    illum = float((1 - np.cos(np.radians(elong))) / 2)

    def _label(th, r, text, colour):
        # Push the label towards the centre when the body is low, or it lands
        # outside the r-limit and matplotlib clips it away.
        dr = -7.0 if r > 78 else 7.0
        ax.text(th, r + dr, text, color=colour, fontsize=8, ha="center",
                va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=BG,
                          edgecolor="none", alpha=0.75))

    if sun.alt.deg > -2:
        th, r = np.radians(float(sun.az.deg)), 90.0 - float(sun.alt.deg)
        ax.scatter([th], [r], s=760, c="#ffca3a", alpha=0.18, linewidths=0,
                   zorder=7)                       # glow
        ax.scatter([th], [r], s=300, c="#ffca3a", edgecolors="#ff9f1c",
                   linewidths=1.2, zorder=8)
        _label(th, r, "Sun", "#ffca3a")
        out.append("sun %.0f°" % sun.alt.deg)

    if moon.alt.deg > -2:
        th, r = np.radians(float(moon.az.deg)), 90.0 - float(moon.alt.deg)
        # Phase as fill brightness rather than a drawn crescent. A crescent has
        # to be painted in the background colour, and the background here is
        # sometimes sky and sometimes the brown tree wedge -- against the wrong
        # one it reads as a smudge rather than a phase. Brightness is
        # unambiguous over either, and the percentage removes all doubt.
        lit = 0.16 + 0.84 * illum
        face = (0.91 * lit, 0.93 * lit, 0.96 * lit)
        ax.scatter([th], [r], s=300, c=[face], edgecolors="#e9ecf5",
                   linewidths=1.4, zorder=8)
        _label(th, r, "Moon %.0f%%" % (100 * illum), "#e9ecf5")
        out.append("moon %.0f° %.0f%%" % (moon.alt.deg, 100 * illum))

    for name, colour in PLANETS:
        try:
            b = get_body(name.lower(), now, loc).transform_to(frame)
        except Exception:
            continue
        alt = float(b.alt.deg)
        if alt <= 0:
            continue
        th, r = np.radians(float(b.az.deg)), 90.0 - alt
        ax.scatter([th], [r], s=110, c=colour, edgecolors=BG, linewidths=0.8,
                   zorder=7.5)
        # Labelled only well clear of the rim: down at the horizon the labels
        # pile into the compass points and each other, and a planet a couple of
        # degrees up is not an observing prospect anyway.
        if alt > 8:
            ax.text(th, r - 4.5, name, color=colour, fontsize=7.5, ha="center",
                    va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor=BG,
                              edgecolor="none", alpha=0.7))
        out.append("%s %.0f°" % (name.lower(), alt))
    return out


def _draw_camera_fov(ax):
    """Outline what the all-sky camera can see, from its plate solution.

    Drawn from the stored solution rather than a nominal field of view, so the
    outline is where the camera actually looks: 4 degrees off zenith at a roll
    nobody chose. Silently skipped if nothing has been solved yet.

    The zenith sits INSIDE the frame, so the border winds a full turn in
    azimuth. Left as raw angles that produces a line sweeping back across the
    chart at the 360/0 seam, hence the unwrap.
    """
    try:
        from sentry import plate_solve
    except ImportError:
        return
    sol = plate_solve.load()
    if sol is None:
        return
    w, h = 2560, 1440
    n = 120
    xs = np.concatenate([np.linspace(0, w, n), np.full(n, w),
                         np.linspace(w, 0, n), np.zeros(n)])
    ys = np.concatenate([np.zeros(n), np.linspace(0, h, n),
                         np.full(n, h), np.linspace(h, 0, n)])
    try:
        alt, az = plate_solve.pixel_to_altaz(sol, xs, ys)
    except Exception:
        return
    th = np.unwrap(np.radians(az))
    r = 90.0 - alt
    th = np.append(th, th[0])
    r = np.append(r, r[0])
    ax.plot(th, r, color=FG, lw=0.9, alpha=0.33, ls=(0, (5, 4)), zorder=3.5)
    ax.fill(th, r, color=FG, alpha=0.035, zorder=1.5, linewidth=0)


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



def _latest_frame(root: Path, out_dir: Path):
    """Annotate the most recent sub the way the `latest` webchat command does.

    Deliberately the same code path as cmd_processing/social_server.latest_cmd:
    get_latest_file -> measure_sky -> fitsfwhm.save_fwhm. Reimplementing the
    annotation would give the site a second opinion about FWHM and sky that
    could drift from the one the observatory reports, and two numbers that
    disagree are worse than one.

    Returns (jpg_path, meta) or (None, {}).
    """
    from astropy.io import fits as _fits
    from fits_processing import fitstojpg, fitsfwhm
    from fits_processing import sky_brightness as sb

    cfg = config.data()
    image_dir = cfg["nina"]["image_dir"]
    aps = cfg["nina"]["arc_sec_per_pixel"]
    latest = fitstojpg.get_latest_file(image_dir, "fits")
    if latest is None:
        return None, {}
    fp = Path(str(latest))

    filter_name, taken, dso = None, None, None
    try:
        hdr = _fits.getheader(fp)
        filter_name = str(hdr.get("FILTER", "")).strip() or None
        # DATE-OBS is when the shutter opened; mtime is when writing finished.
        # For "how long ago" the exposure start is the honest answer.
        taken = str(hdr.get("DATE-OBS", "")).strip() or None
        dso = str(hdr.get("OBJECT", "")).strip() or None
    except Exception:
        pass
    if not dso:
        # Fall back to the directory the frame sits in: subs live under
        # <image_dir>/<dso>/[rig/]<date>/LIGHT/.
        try:
            dso = fp.relative_to(Path(image_dir)).parts[0]
        except Exception:
            dso = None
    if not taken:
        taken = datetime.fromtimestamp(fp.stat().st_mtime, timezone.utc).isoformat(
            timespec="seconds")
    elif not taken.endswith("Z") and "+" not in taken:
        taken = taken + "+00:00"          # N.I.N.A writes DATE-OBS in UTC

    sky = None
    try:
        sky = sb.measure_sky(fp, arcsec_per_pixel=aps)
    except Exception:
        pass

    jpg = out_dir / "live_latest.jpg"
    try:
        out, mean_px, mean_ecc = fitsfwhm.save_fwhm(
            fp, jpg, arcsec_per_pixel=aps, annotate=False,
            filter_name=filter_name, sky_data=sky)
    except Exception as exc:
        return None, {"latest_error": str(exc)}

    meta = {"latest_dso": dso, "latest_taken": taken,
            "latest_filter": filter_name, "latest_file": fp.name,
            "latest_fwhm_arcsec": round(float(mean_px) * aps, 2) if mean_px else None,
            "latest_ecc": round(float(mean_ecc), 3) if mean_ecc else None}
    if sky and sky.get("sky_adu_per_s") is not None:
        meta["latest_sky_adu_per_s"] = round(float(sky["sky_adu_per_s"]), 4)
    return Path(str(out)), meta


def _night_of(ts):
    """The observing night a frame timestamp belongs to, or None.

    Shifted back 12 hours so a session that runs past midnight counts as one
    night rather than two -- which is most of them here.
    """
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return (t - timedelta(hours=12)).date()
    except Exception:
        return None


def _stats_chart(root, out_dir, dso):
    """Render the session stats chart for *dso*. Returns (path, meta).

    Meta carries the frame count and DSO so the page can caption the chart
    without parsing the image, and so it can hide the panel outright when
    there is nothing to show -- an empty axes frame under a heading reads as a
    fault rather than as an idle observatory.
    """
    if not dso:
        return None, {}
    try:
        from fits_processing import imaging_artifacts as ia
        cfg_nina = config.data()["nina"]
        cache = Path(cfg_nina["image_dir"]) / dso / "frame_stats.json"
        if not cache.exists():
            return None, {}
        # EVERY frame of this target, not just the current session -- the
        # equivalent of `stats <dso> all`. The web-chat card deliberately shows
        # only tonight, because there it sits beside a live ticker and answers
        # "how is this night going". The page is not that: a visitor arriving
        # cold wants what the target has accumulated, and a chart that silently
        # dropped every earlier night would understate the work.
        out = out_dir / "live_stats.jpg"
        res = ia.render_stats_plot_from_cache_path(cache_path=cache,
                                                   output_path=out,
                                                   latest_session_only=False)
        if res is None:
            return None, {}
        frames = ia.gather_dso_frames(cache.parent, latest_session_only=False)
        meta = {"stats_dso": dso, "stats_frames": len(frames)}
        filts = sorted({str(f.get("filter", "")).strip()
                        for f in frames if f.get("filter")})
        if filts:
            meta["stats_filters"] = filts
        # Nights, so the caption can say how much history the chart spans.
        # Counted off the frame timestamps rather than the day boundary,
        # because a session that runs past midnight is one night's work and
        # splitting it would inflate the count.
        nights = {_night_of(f.get("time")) for f in frames}
        nights.discard(None)
        if nights:
            meta["stats_nights"] = len(nights)
        return Path(str(res)), meta
    except Exception as exc:
        # Never let the chart cost the skymap push, which is the part the page
        # has depended on since long before this panel existed.
        print("stats chart skipped (%s: %s)" % (type(exc).__name__, exc))
        return None, {}


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

    _draw_camera_fov(ax)
    bodies = _draw_solar_system(ax, fig, frame, loc)

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

    # Publish the boot INSTANT, not an elapsed count. The page subtracts it
    # itself, so the figure stays right between the 5-minute pushes instead of
    # sitting frozen at whatever it was when this ran. It is safe to let the
    # browser extrapolate only because the same status carries `generated`:
    # once that goes stale the page stops rendering uptime altogether, which is
    # the whole point -- an elapsed count extrapolated from a dead feed climbs
    # forever and reports a machine that is off as having the best uptime yet.
    #
    # kernel32 rather than psutil: psutil is listed in requirements.txt but is
    # NOT installed in .venv, so the obvious version of this silently published
    # nothing. ctypes is stdlib and cannot go missing the same way. No
    # subprocess fallback either -- this script runs under live_skymap.bat's
    # 4-minute kill, where an extra process spawn is a hang waiting to happen
    # (see f07477a).
    #
    # GetTickCount64 excludes time spent asleep, which is the honest reading
    # for "up": the observatory PC does not sleep, and if it ever did, the
    # hours it was unavailable should not be counted as uptime. restype must be
    # set -- ctypes defaults to a 32-bit signed int, which wraps to negative
    # after 24.8 days and would report a boot in the future.
    try:
        import ctypes
        _tick = ctypes.windll.kernel32.GetTickCount64
        _tick.restype = ctypes.c_ulonglong
        up_s = _tick() / 1000.0
        status["boot"] = (datetime.now(timezone.utc)
                          - timedelta(seconds=up_s)).isoformat(timespec="seconds")
    except Exception:
        pass

    if bodies:
        status["bodies"] = ", ".join(bodies)
    if tgt:
        name, ra, dec, good_hours = tgt
        tc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(frame)
        talt, taz = float(tc.alt.deg), float(tc.az.deg)
        # haz is ALREADY normalised and sorted, with a wrap point appended as
        # haz[0] + 360. Do NOT take % 360 of it again: that folds the wrap point
        # back to 0, leaving [.. 343, 355, 0], which is not increasing. np.interp
        # does not check, and silently returned the profile's MAXIMUM (82.8 deg)
        # for every azimuth -- so every target below 82.8 deg read "behind the
        # trees" while the chart, which draws from the unmodified array, showed
        # it sitting in clear sky.
        tree_alt = float(np.interp(taz % 360.0, np.array(haz), np.array(hal)))
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

    # The most recent sub, annotated the same way `latest` does.
    latest_jpg, latest_meta = _latest_frame(root, out_dir)
    status.update(latest_meta)

    # The session stats chart, rendered by the same code the `stats` command
    # uses, so the site and the observatory cannot report different numbers --
    # the same reason latest.jpg is built by the `latest` renderer above.
    #
    # Rendered here rather than pushed from frame_watcher's artifact worker,
    # which already rebuilds this chart on every sub for the local web chat.
    # That worker runs on the imaging thread, and this job runs every 5 minutes
    # against 300 s subs, so the freshness is the same while the network I/O
    # stays out of the capture path.
    stats_jpg, stats_meta = _stats_chart(root, out_dir, latest_meta.get("latest_dso"))
    status.update(stats_meta)

    # The conductor's machine state: one string naming the lit node of
    # docs/night_machine.mmd, so the site can highlight it on the live state
    # diagram. Distinct keys from "state", which already means the TARGET's
    # visibility on this page. A down conductor publishes nothing and the
    # panel hides -- absence is honest, a guess is not. The 3 s timeout
    # matters: this whole script runs under live_skymap.bat's 4-minute kill.
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8096/v1/state",
                                    timeout=3) as r:
            _c = json.loads(r.read().decode("utf-8"))
        if _c.get("state"):
            status["machine_state"] = _c["state"]
            status["machine_shadow"] = bool(_c.get("shadow"))
            _dso = (_c.get("context") or {}).get("dso")
            if _dso:
                status["machine_dso"] = _dso
    except Exception:
        pass

    js = out_dir / "live_status.json"
    js.write_text(json.dumps(status, indent=1))
    print("rendered", img.name, "->", status.get("target"), status.get("state"))

    if "--no-push" in sys.argv:
        return

    # Shared transport, NOT an inline scp loop. This file used to duplicate
    # the copy with only BatchMode+ConnectTimeout -- no ServerAlive* and no
    # subprocess timeout -- which is exactly the unbounded-hang class that
    # live_push exists to prevent and that stalled this very job on
    # 2026-08-14 (three 240s .bat kills in one hour and a stale-feed page,
    # on a link that measured 3ms once anyone looked). One transport, one
    # place to get the bounding right.
    pushes = [(img, "skymap.jpg"), (js, "status.json")]
    if latest_jpg is not None and latest_jpg.exists():
        pushes.insert(1, (latest_jpg, "latest.jpg"))
    if stats_jpg is not None and stats_jpg.exists():
        pushes.insert(1, (stats_jpg, "stats.jpg"))
    # The generated night-machine diagram rides along so the site renders the
    # SAME structure CI enforces against the code -- the page never hardcodes
    # a copy that could drift. Tiny file, harmless to re-push every cycle.
    mmd = root / "docs" / "night_machine.mmd"
    if "machine_state" in status and mmd.exists():
        pushes.append((mmd, "night_machine.mmd"))
    from scripts import live_push
    live_push.push(pushes)
    print("pushed", len(pushes), "file(s) to", live_push.HOST + ":" + live_push.DEST)


if __name__ == "__main__":
    main()
