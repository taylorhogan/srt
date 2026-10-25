"""What the moon costs per filter at this site, measured on our own frames (2026-10-25).

    python scripts/moon_sky_cost.py      # table + fit -> local/analysis/moon_sky_{cost,fit}.json

Every frame_stats.json row with a measured pedestal carries the sky rate
(sky_adu_per_s). For each frame this works out the moon's altitude, its lit
fraction, its distance from the target and the sun's altitude (frames in
twilight are left out). Each frame's sky is divided by the same target's
DARK sky in the same filter (median of its frames with the moon down), so
the field's own brightness and the filter's throughput cancel. Targets with
fewer than MIN_DARK moon-down frames in a filter have no baseline there.

THE MODEL. The moon's sky brightness at the target is predicted with
Krisciunas & Schaefer (1991) (phase, moon-target distance, both altitudes,
V-band extinction 0.172), and each filter's measured ratio is fitted as
1 + c x B_moon -- one c per filter. A frame's worth against a dark one is the
noise-variance ratio (sky + read noise): read noise 5.82 ADU and gain 1.328
e-/ADU at gain 0, measured on a bias pair and an L flat pair 2026-10-25. The
read noise is why narrowband barely cares: a dark O-III sky adds ~0.5 ADU^2
in 300 s against 34 ADU^2 of read noise.

First fit, 2026-10-25 (2112 frames, 13 targets): c = L 1.8e-3, R 1.2e-3,
G 3.0e-3, B 2.0e-3, Ha 1.2e-2, O-III 2.1e-2 per nL; log scatter L 0.35,
RGB ~0.21, O-III 0.58. Worth at a 95% moon 60 deg away: L 0.45, G 0.37,
Ha 0.88, O-III 0.81. Without any moon the sky still varies ~2x night to night
(haze over the town's lights), which is most of the scatter.

Feeds the planner's moon rule (owner's question 2026-10-25: should galaxies
wait for a dark sky?).
"""
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

MIN_DARK = 5
FILTERS = ["L", "R", "G", "B", "Ha", "O-III", "S-II"]
K_V = 0.172                              # V-band extinction (KS 1991)
READ_NOISE_ADU, GAIN = 5.82, 1.328       # QHY600M at gain 0, measured 2026-10-25
BAD_BASELINE = {("ngc6946", "L")}        # dark baseline from 2 frames; ratios 0.19-0.38


def _airmass(alt_deg):
    z = math.radians(90 - alt_deg)
    return (1 - 0.96 * math.sin(z) ** 2) ** -0.5


def moon_brightness(illum, sep_deg, moon_alt, target_alt):
    """The moon's sky brightness at the target in nanoLamberts (Krisciunas & Schaefer 1991). Pure."""
    if moon_alt <= 0:
        return 0.0
    elong = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * illum))))
    a = 180.0 - elong
    istar = 10 ** (-0.4 * (3.84 + 0.026 * abs(a) + 4e-9 * a ** 4))
    rho = max(sep_deg, 1.0)
    f = 10 ** 5.36 * (1.06 + math.cos(math.radians(rho)) ** 2) + 10 ** (6.15 - rho / 40)
    return (f * istar * 10 ** (-0.4 * K_V * _airmass(moon_alt))
            * (1 - 10 ** (-0.4 * K_V * _airmass(max(target_alt, 5)))))


def frame_worth(dark_sky_adu_s, c, b_moon, exptime=300.0):
    """What a frame under moonlight b_moon is worth vs a dark one (noise-variance ratio). Pure."""
    s0 = dark_sky_adu_s * exptime / GAIN
    rn2 = READ_NOISE_ADU ** 2
    return (s0 + rn2) / (s0 * (1 + c * b_moon) + rn2)


def target_coord(dso_dir: Path):
    """The pointing from the first light frame's header (RA/DEC in degrees, or OBJCT* strings)."""
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    for p in sorted(dso_dir.rglob("*.fits")):
        if p.parent.name.upper() != "LIGHT":
            continue
        h = fits.getheader(p)
        if "RA" in h and "DEC" in h and isinstance(h["RA"], (int, float)):
            return SkyCoord(float(h["RA"]) * u.deg, float(h["DEC"]) * u.deg)
        if "OBJCTRA" in h and "OBJCTDEC" in h:
            return SkyCoord(h["OBJCTRA"], h["OBJCTDEC"], unit=(u.hourangle, u.deg))
    return None


def load_frames(root: Path):
    out = []
    for fs in sorted(root.glob("*/frame_stats.json")):
        d = json.loads(fs.read_text())
        rows = [r for r in (d.values() if isinstance(d, dict) else d)
                if isinstance(r, dict) and r.get("sky_adu_per_s") is not None
                and r.get("pedestal_source") == "measured" and r.get("time")]
        if rows:
            out.append((fs.parent, rows))
    return out


def main():
    import numpy as np
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.utils import iers
    from configs import config
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None

    cfg = config.data()
    loc = cfg["location"]
    site = EarthLocation.from_geodetic(float(loc["longitude"]) * u.deg, float(loc["latitude"]) * u.deg,
                                       float(loc.get("elevation", 0)) * u.m)
    root = Path(cfg["nina"]["image_dir"])

    recs = []
    for dso_dir, rows in load_frames(root):
        c = target_coord(dso_dir)
        if c is None:
            continue
        t = Time([r["time"] for r in rows], scale="utc")
        fr = AltAz(obstime=t, location=site)
        moon = get_body("moon", t, site)
        sun = get_body("sun", t, site)
        malt = moon.transform_to(fr).alt.deg
        salt = sun.transform_to(fr).alt.deg
        talt = c.transform_to(fr).alt.deg
        sep = moon.separation(c).deg
        illum = 0.5 * (1 - np.cos(moon.separation(sun).rad))
        for r, a, s, ta, sp, il in zip(rows, malt, salt, talt, sep, illum):
            if s > -18:
                continue                     # twilight brightens the sky by itself
            recs.append({"dso": dso_dir.name, "filter": r["filter"], "sky": float(r["sky_adu_per_s"]),
                         "moon_alt": float(a), "illum": float(il), "sep": float(sp), "t_alt": float(ta),
                         "night": Path(r["path"]).parent.parent.name})

    # Dark baseline per (target, filter): median sky with the moon down.
    groups = defaultdict(list)
    for r in recs:
        groups[(r["dso"], r["filter"])].append(r)
    for key, rs in groups.items():
        dark = [r["sky"] for r in rs if r["moon_alt"] < -2]
        base = float(np.median(dark)) if len(dark) >= MIN_DARK and key not in BAD_BASELINE else None
        for r in rs:
            r["ratio"] = r["sky"] / base if base and base > 0 else None

    usable = [r for r in recs if r["ratio"] is not None]
    print("frames: %d after the twilight cut, %d with a dark baseline (%d targets)"
          % (len(recs), len(usable), len({r["dso"] for r in usable})))

    # The fit: ratio = 1 + c * B_moon, one c per filter, from moonlit frames
    # bright enough to measure (B_moon > 20 nL).
    fit, dark_sky = {}, {}
    print("\nfit: sky ratio = 1 + c x moon brightness (KS 1991, nL)")
    for f in FILTERS:
        up = [r for r in usable if r["filter"] == f and r["moon_alt"] > 0]
        xs = np.array([moon_brightness(r["illum"], r["sep"], r["moon_alt"], r["t_alt"]) for r in up])
        ys = np.array([r["ratio"] - 1 for r in up])
        ok = xs > 20
        dark = [r["sky"] for r in usable if r["filter"] == f and r["moon_alt"] < -2]
        if ok.sum() < 8 or not dark:
            print("  %-5s too few moonlit frames (%d)" % (f, int(ok.sum())))
            continue
        c = float(np.median(ys[ok] / xs[ok]))
        scatter = float(np.std(np.log(np.clip(1 + ys[ok], 0.05, None) / (1 + c * xs[ok]))))
        fit[f], dark_sky[f] = c, float(np.median(dark))
        print("  %-5s n=%3d  c=%.2e  log scatter %.2f  dark sky %.4f ADU/s"
              % (f, int(ok.sum()), c, scatter, dark_sky[f]))

    print("\nworth of a 300 s frame vs a dark one (moon 30 deg up, target 50 deg up):")
    print("   lit  sep  " + "  ".join("%6s" % f for f in fit))
    for il in (0.25, 0.5, 0.75, 0.95):
        for sp in (30, 60, 90, 120):
            b = moon_brightness(il, sp, 30, 50)
            print("   %3d%% %3d  " % (il * 100, sp)
                  + "  ".join("%6.2f" % frame_worth(dark_sky[f], fit[f], b) for f in fit))

    out = Path(__file__).resolve().parent.parent / "local" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "moon_sky_cost.json").write_text(json.dumps({"frames": usable}, indent=1))
    (out / "moon_sky_fit.json").write_text(json.dumps(
        {"c": fit, "dark_sky_adu_s": dark_sky, "read_noise_adu": READ_NOISE_ADU, "gain": GAIN,
         "k_v": K_V}, indent=1))
    print("\nwrote", out / "moon_sky_cost.json", "and moon_sky_fit.json")


if __name__ == "__main__":
    main()
