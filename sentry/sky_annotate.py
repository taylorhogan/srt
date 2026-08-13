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
import os
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from sentry import plate_solve

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


def annotate(src, dst, sol=None, shape=None, profile=None):
    """Write an annotated copy of src to dst. Returns dst, or None if it can't."""
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
