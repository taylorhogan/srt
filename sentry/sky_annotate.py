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


def annotate(src, dst, sol=None, shape=None):
    """Write an annotated copy of src to dst. Returns dst, or None if it can't."""
    from PIL import Image, ImageDraw, ImageFont
    sol = sol or plate_solve.load()
    if sol is None:
        return None
    im = Image.open(str(src)).convert("RGB")
    w, h = im.size
    marks, zenith = compass_positions(sol, (h, w))
    if not marks:
        return None

    dr = ImageDraw.Draw(im)
    font = None
    for cand in ("arialbd.ttf", "arial.ttf", "seguisb.ttf"):
        try:
            font = ImageFont.truetype(cand, 58)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for name, (x, y) in marks.items():
        v = np.array([x - zenith[0], y - zenith[1]])
        v = v / max(np.hypot(*v), 1e-6)
        # A short tick pointing outward along the bearing, then the letter
        # inboard of it, so the glyph never runs off the edge.
        tip = (x, y)
        tail = (x - v[0] * 46, y - v[1] * 46)
        dr.line([tail, tip], fill=(255, 210, 60), width=5)
        tw, th = 30, 34
        lx, ly = x - v[0] * 96, y - v[1] * 96
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
