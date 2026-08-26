#!/usr/bin/env python3
"""Draw compass bearings on a sky frame, from the plate solution.

The directions are computed, not assumed. The camera sits 4 degrees off zenith
with a rolled-up orientation nobody chose deliberately, so "north is up" is
simply false here -- the solution says where north actually is, and this puts a
label there.

Annotates a COPY. The archived frame stays clean, because the archive is
training data for the weather detector and burnt-in text would poison it: a
label sitting in the same pixels every frame is exactly the kind of constant a
classifier learns instead of the sky.

The horizon is not in shot -- the frame reaches about 45 degrees from zenith on
the long axis and 25 on the short -- so a bearing cannot be drawn where it meets
the ground. Instead each label goes at the frame edge along the direction that
bearing lies in, which is what a compass rose on an all-sky image means anyway.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from sentry import plate_solve

_logger = logging.getLogger(__name__)

CARDINALS = [("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0)]
PROBE_ALT = 45.0        # a point this far up, used only to get the bearing's
                        # direction on the image; never drawn itself.


def _edge_hit(origin, direction, w, h, margin):
    """Where a ray from origin leaves the frame, inset by margin."""
    ox, oy = origin
    dx, dy = direction
    ts = []
    if abs(dx) > 1e-9:
        for bx in (margin, w - margin):
            t = (bx - ox) / dx
            if t > 0:
                y = oy + t * dy
                if margin <= y <= h - margin:
                    ts.append(t)
    if abs(dy) > 1e-9:
        for by in (margin, h - margin):
            t = (by - oy) / dy
            if t > 0:
                x = ox + t * dx
                if margin <= x <= w - margin:
                    ts.append(t)
    if not ts:
        return None
    t = min(ts)
    return ox + t * dx, oy + t * dy


def compass_positions(sol, shape, margin=90):
    """{'N': (x, y), ...} at the frame edge, plus the zenith pixel."""
    h, w = shape
    zx, zy = plate_solve.altaz_to_pixel(sol, 90.0, 0.0)
    zenith = (float(zx[0]), float(zy[0]))
    out = {}
    for name, az in CARDINALS:
        px, py = plate_solve.altaz_to_pixel(sol, PROBE_ALT, az)
        v = np.array([float(px[0]) - zenith[0], float(py[0]) - zenith[1]])
        n = np.hypot(*v)
        if n < 1e-6:
            continue
        hit = _edge_hit(zenith, v / n, w, h, margin)
        if hit:
            out[name] = hit
    return out, zenith


def imaging_target():
    """(name, ra_deg, dec_deg) for the DSO being imaged, or None.

    Two sources, in order of authority:

    1. An instruction marked ``in process``. That is the queue saying so
       outright, but nothing currently sets it -- as of 2026-08-26 the live
       queue holds only ``waiting`` and ``completed`` -- so it is tried first
       and expected to miss.
    2. The generated sequence N.I.N.A actually runs. It carries the target name
       AND the resolved coordinates that were patched into it, so it needs no
       name lookup and cannot disagree with what the camera is pointed at.

    Returns None rather than guessing. A dot in the wrong place is worse than
    no dot: it would be read as a plate-solve error.
    """
    try:
        from control import instructions
        for row in instructions.get_sorted_instructions(apply_convergence=False):
            if str(row.get("status", "")).lower() == "in process":
                ra, dec = row.get("ra_deg"), row.get("dec_deg")
                if ra is not None and dec is not None:
                    return row.get("dso"), float(ra), float(dec)
                tgt = instructions.resolve_target_by_name(row.get("dso"))
                if tgt is not None:
                    return (row.get("dso"), float(tgt.coord.ra.deg),
                            float(tgt.coord.dec.deg))
    except Exception:
        _logger.debug("in-process lookup failed", exc_info=True)

    try:
        import json
        from configs import config
        path = config.data()["nina"]["sequence_output"]
        with open(path, encoding="utf-8-sig") as fh:
            doc = json.load(fh)

        def walk(o):
            if isinstance(o, dict):
                yield o
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)

        for node in walk(doc):
            if "InputTarget" not in str(node.get("$type", "")):
                continue
            c = node.get("InputCoordinates") or {}
            if not c:
                continue
            ra = (float(c.get("RAHours", 0)) + float(c.get("RAMinutes", 0)) / 60.0
                  + float(c.get("RASeconds", 0)) / 3600.0) * 15.0
            dec = (abs(float(c.get("DecDegrees", 0)))
                   + float(c.get("DecMinutes", 0)) / 60.0
                   + float(c.get("DecSeconds", 0)) / 3600.0)
            if c.get("NegativeDec") or float(c.get("DecDegrees", 0)) < 0:
                dec = -dec
            return node.get("TargetName"), ra, dec
    except Exception:
        _logger.debug("sequence target lookup failed", exc_info=True)
    return None


def target_pixel(sol, ra_deg, dec_deg, shape, when=None, inset=20.0):
    """Where the target lands in the frame, or None if it is not in it.

    None covers three separate cases and deliberately does not distinguish
    them, because the drawing code does the same thing for all three: the
    target is below the horizon, it is above the horizon but outside this
    camera's field, or the transform failed.
    """
    import pytz
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from configs import config

    loc = config.data()["location"]
    site = EarthLocation.from_geodetic(loc["longitude"] * u.deg,
                                       loc["latitude"] * u.deg,
                                       loc["elevation"] * u.m)
    t = Time(when or datetime.now(pytz.timezone(loc["timezone"])))
    altaz = SkyCoord(ra_deg * u.deg, dec_deg * u.deg).transform_to(
        AltAz(obstime=t, location=site))
    alt, az = float(altaz.alt.deg), float(altaz.az.deg)
    # Below the horizon the projection still returns a pixel -- the fisheye
    # model does not know the ground is there -- so this has to be checked
    # explicitly or a set target gets a dot painted on the trees.
    if alt <= 0.0:
        return None
    x, y = plate_solve.altaz_to_pixel(sol, alt, az)
    x, y = float(np.atleast_1d(x)[0]), float(np.atleast_1d(y)[0])
    h, w = shape
    if not (inset <= x <= w - inset and inset <= y <= h - inset):
        return None
    return x, y, alt, az


def annotate(src, dst, sol=None, shape=None, profile=None, when=None):
    """Write an annotated copy of src to dst. Returns dst, or None if it can't.

    *when* is the instant the target dot is computed for; it defaults to now,
    which is right for the live path. Pass the frame's own timestamp to
    re-annotate an archived frame, or the dot lands where the object is now
    rather than where it was.
    """
    from PIL import Image, ImageDraw, ImageFont
    profile = profile or plate_solve.DEFAULT_PROFILE
    sol = sol or plate_solve.load(profile=profile)
    if sol is None:
        return None
    im = Image.open(str(src)).convert("RGB")
    w, h = im.size
    # Everything below was sized by eye on the 2560-wide Kasa frame. The all-sky
    # camera is half that, so fixed pixel sizes would put a compass letter across
    # an eighth of the picture. Scaled, the annotation looks the same size on
    # both -- which is what "sized by eye" meant in the first place.
    k = w / 2560.0
    marks, zenith = compass_positions(sol, (h, w), margin=max(20, int(90 * k)))
    if not marks:
        return None

    dr = ImageDraw.Draw(im)
    font = None
    for cand in ("arialbd.ttf", "arial.ttf", "seguisb.ttf"):
        try:
            font = ImageFont.truetype(cand, max(18, int(58 * k)))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # The region the star statistics are actually measured inside. Outside it is
    # still sky and still photographed, but the lens does not deliver stars there
    # at any brightness, so those catalogue stars are dropped from the
    # completeness denominator rather than counted as misses.
    #
    # Not a plain circle, and drawing one was wrong: at r=800 in a 2560x1440
    # frame the disc reaches y=-80 and y=1520, overshooting the picture top and
    # bottom by 80 px. What is measured is the disc INTERSECTED with the frame --
    # a disc with both caps sliced flat -- and an arc alone left the shape open
    # exactly where it leaves the picture, which reads as though the strips above
    # and below were excluded when they are counted. So the boundary is drawn
    # wherever it runs: arc where the circle bounds the region, straight where
    # the frame does.
    #
    # Faint on purpose. It is a note about the measurement, not a feature of the
    # sky, and it must not compete with the picture.
    #
    # Skipped entirely for a camera whose radius has not been measured yet:
    # there is no measured region to outline, and drawing one at a borrowed
    # radius would label an area the counts were not taken in.
    r = plate_solve.measured_radius(profile)
    cx, cy = sol["cx"], sol["cy"]
    if r:
        ring = Image.new("RGBA", im.size, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        INSET = 20.0            # must match the inframe cut in plate_solve
        x0, y0, x1, y1 = INSET, INSET, w - INSET, h - INSET
        ang = np.linspace(0, 2 * np.pi, 1441)
        ax, ay = cx + r * np.cos(ang), cy + r * np.sin(ang)
        inside = (ax >= x0) & (ax <= x1) & (ay >= y0) & (ay <= y1)
        for i in range(len(ang) - 1):
            if inside[i] and inside[i + 1]:
                rd.line([ax[i], ay[i], ax[i + 1], ay[i + 1]],
                        fill=(255, 255, 255, 70), width=3)
        # Where the frame is the boundary rather than the circle: the chord the
        # circle cuts across each edge it crosses.
        for const, horiz in ((y0, True), (y1, True), (x0, False), (x1, False)):
            d2 = r * r - (const - (cy if horiz else cx)) ** 2
            if d2 <= 0:
                continue
            half = float(np.sqrt(d2))
            c = cx if horiz else cy
            lo = max(c - half, x0 if horiz else y0)
            hi = min(c + half, x1 if horiz else y1)
            if hi <= lo:
                continue
            rd.line([lo, const, hi, const] if horiz else [const, lo, const, hi],
                    fill=(255, 255, 255, 70), width=3)
        im = Image.alpha_composite(im.convert("RGBA"), ring).convert("RGB")
        dr = ImageDraw.Draw(im)
        small = None
        for cand in ("arial.ttf", "seguisb.ttf", "arialbd.ttf"):
            try:
                small = ImageFont.truetype(cand, max(11, int(26 * k)))
                break
            except OSError:
                continue
        if small is not None:
            label = "star counts measured inside this outline"
            # r is larger than half the frame height, so the outline is clipped
            # top and bottom and "just below the arc" lands outside the picture.
            # Clamp to the bottom edge rather than let the caption disappear.
            lx = min(max(cx - 230 * k, 20), w - 560 * k)
            ly = min(cy + r + 12, h - 62 * k)
            for ox in (-2, 0, 2):
                for oy in (-2, 0, 2):
                    dr.text((lx + ox, ly + oy), label, fill=(0, 0, 0), font=small)
            dr.text((lx, ly), label, fill=(210, 215, 225), font=small)

    for name, (x, y) in marks.items():
        v = np.array([x - zenith[0], y - zenith[1]])
        v = v / max(np.hypot(*v), 1e-6)
        # A short tick pointing outward along the bearing, then the letter
        # inboard of it, so the glyph never runs off the edge.
        tip = (x, y)
        tail = (x - v[0] * 46 * k, y - v[1] * 46 * k)
        dr.line([tail, tip], fill=(255, 210, 60), width=max(2, int(5 * k)))
        tw, th = 30 * k, 34 * k
        lx, ly = x - v[0] * 96 * k, y - v[1] * 96 * k
        # Outline first so the letter stays readable over a bright cloud.
        for ox in (-3, 0, 3):
            for oy in (-3, 0, 3):
                dr.text((lx - tw + ox, ly - th + oy), name, fill=(0, 0, 0),
                        font=font)
        dr.text((lx - tw, ly - th), name, fill=(255, 210, 60), font=font)

    # The target being imaged, if it falls inside the picture. Drawn last so it
    # sits above the compass marks and the measured-region outline.
    #
    # Cyan, not the compass yellow: this is a different KIND of thing -- the
    # compass is a fixed property of the mounted camera, the dot moves with the
    # sky and changes target to target. Sharing a colour would suggest they can
    # be read the same way.
    #
    # A ring rather than a filled blob, so the star it marks stays visible
    # inside it. The whole point is to let someone check the dot really is on
    # the object.
    try:
        tgt = imaging_target()
        if tgt is not None:
            name, ra_deg, dec_deg = tgt
            hit = target_pixel(sol, ra_deg, dec_deg, (h, w), when=when)
            if hit is not None:
                tx, ty, alt, az = hit
                cyan, black = (90, 235, 255), (0, 0, 0)
                rr = max(9, int(26 * k))
                wdt = max(2, int(4 * k))
                # Black halo first: over a bright cloud or the Milky Way a thin
                # cyan ring alone disappears.
                dr.ellipse([tx - rr - 1, ty - rr - 1, tx + rr + 1, ty + rr + 1],
                           outline=black, width=wdt + 2)
                dr.ellipse([tx - rr, ty - rr, tx + rr, ty + rr],
                           outline=cyan, width=wdt)
                dot = max(2, int(3 * k))
                dr.ellipse([tx - dot, ty - dot, tx + dot, ty + dot], fill=cyan)
                label = "%s  %.0f° alt" % (name or "target", alt)
                lfont = font
                for cand in ("arialbd.ttf", "arial.ttf", "seguisb.ttf"):
                    try:
                        lfont = ImageFont.truetype(cand, max(13, int(32 * k)))
                        break
                    except OSError:
                        continue
                # Label below the ring, nudged back inside if the target is
                # near an edge.
                lxp = min(max(tx - 60 * k, 8), w - 300 * k)
                lyp = min(ty + rr + 6 * k, h - 46 * k)
                for ox in (-2, 0, 2):
                    for oy in (-2, 0, 2):
                        dr.text((lxp + ox, lyp + oy), label, fill=black, font=lfont)
                dr.text((lxp, lyp), label, fill=cyan, font=lfont)
    except Exception:
        # Never let the target marker cost the compass annotation, which is the
        # part that has been trusted on the live page for weeks.
        _logger.debug("target dot failed", exc_info=True)

    im.save(str(dst), quality=88)
    return Path(dst)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    src = Path(argv[0])
    dst = Path(argv[1]) if len(argv) > 1 else src.with_name(src.stem + "_compass.jpg")
    sol = plate_solve.load()
    if sol is None:
        print("no stored plate solution; run sentry/plate_solve.py <frame> --save")
        return 1
    from PIL import Image
    w, h = Image.open(str(src)).size
    marks, zenith = compass_positions(sol, (h, w))
    print("zenith at (%.0f, %.0f)" % zenith)
    for k, (x, y) in marks.items():
        print("  %s at (%.0f, %.0f)" % (k, x, y))
    out = annotate(src, dst, sol)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
