"""Where to cut a finished picture: the subject by type, then a box.

Built 2026-11-02 ("can you write an algorithm/model to auto crop an image
based on its type and the picture"), in SHADOW: the morning render writes
`<dso>_<recipe>_<tag>_crop.jpg` beside every full composite, records the box
in the render meta and pushes the crop after the full view. Nothing
downstream uses it until the owner has judged a run of them.

THE IDEA. Cropping is two decisions: what the subject is, and where the box
goes. Only the first depends on the object's type.

Subject map by type, on the composite's luminance brought to ~800 px with
the stars taken out by a median filter (`starless`):
  nebula          diffuse light above an isophote: sky + ISOPHOTE * (bright - sky),
                  bright = the 99.5th percentile of the starless map. A fixed
                  sigma would catch the whole field (a smooth map has almost
                  no noise, and IC 410's dim sulphur haze covers its lower
                  half); an isophote relative to the subject's own level does
                  not. A ladder (x2, x1, x0.5) finds a level where the subject
                  is 3-60 % of the field.
  galaxy          the same at a fainter isophote, so the outer disc counts.
  cluster         diffuse light is the wrong signal for an open or globular
                  cluster: the subject is STAR DENSITY, counted per block from
                  the luminance's peaks, where it stands clearly above the
                  field's. Unioned with any diffuse subject in band, because
                  a cluster in a nebula (Melotte 15) is a picture of both.
  galaxy_cluster  the diffuse map at the galaxy isophote, closed by a wide
                  disc so the members merge into one region.
The region is the largest connected component plus any carrying a tenth of
its flux, so a second nebula across the field is kept and a stray wisp is not.

Box fit, the same for every type: the subject's bounding box grown by a
type-specific margin on each side (galaxies get the most: a galaxy with no
sky around it looks cramped), then widened to the nearest of four aspects
(3:2, 16:10, 4:3, 1:1), clamped inside the footprint minus an edge band
(the common footprint's rim carries fringe), and nudged off any bright star
within a few pixels of an edge. Each aspect is scored by subject coverage
minus half the box's empty share; the best wins, ties to the aspect nearest
the subject's own shape. A subject covering more than 70 % of the field, or
none found, means no crop: the full frame is the answer.

Validated on the two hand crops of 2026-11-02 (IC 410 and Melotte 15 SHO,
see tests/test_auto_crop.py for the synthetic cases). Known weak cases: two
subjects of equal weight far apart (both kept, the box spans them), a faint
nebula whose ladder finds no level in band (full frame), and taste.

Coordinates: boxes are [x0, y0, x1, y1] on the array as stored, which is the
picture's own orientation (stacker.sky_parity is the identity on this rig).
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np

ASPECTS = (("3:2", 1.5), ("16:10", 1.6), ("4:3", 4.0 / 3.0), ("1:1", 1.0))
TYPES = ("galaxy", "nebula", "cluster", "galaxy_cluster", "unknown")
WORK_PX = 800                 # the long side the subject is found at
MEDIAN_PX = 7                 # star removal at work scale (a star is <= 3 px there)
SMOOTH_PX = 2.0
# each side, as a share of the subject's extent on that axis
MARGIN = {"galaxy": 0.5, "nebula": 0.08, "cluster": 0.35, "galaxy_cluster": 0.3, "unknown": 0.15}
# the subject is what sits above sky + ISOPHOTE * (bright - sky)
ISOPHOTE = {"galaxy": 0.06, "nebula": 0.15, "cluster": 0.15, "galaxy_cluster": 0.06, "unknown": 0.12}
ISOPHOTE_LADDER = (2.0, 1.0, 0.5)   # bright to faint: the core first, then the halo while it grows like one
HALO_GROWTH = 2.0             # a fainter level is accepted while the region is under this times the CORE's area
AREA_BAND = (0.001, 0.60)     # the subject's share of the field a level may give
# the box's long side is at least this share of the field's: a small object needs its context
MIN_BOX_SHARE = {"galaxy": 0.35, "nebula": 0.40, "cluster": 0.40, "galaxy_cluster": 0.50, "unknown": 0.40}
SQUARE_PENALTY = 0.05         # the gallery is landscape; 1:1 has to earn it
SHAPE_WEIGHT = 0.10           # an aspect far from the subject's own shape costs this per e-fold
# closing radius, as a share of the long side: merges a nebula's dark bays, a cluster's members
CLOSE = {"galaxy": 0.01, "nebula": 0.03, "cluster": 0.04, "galaxy_cluster": 0.05, "unknown": 0.02}
KEEP_SHARE = {"galaxy": 0.30, "nebula": 0.10, "cluster": 0.10, "galaxy_cluster": 0.10, "unknown": 0.15}
KEEP_REACH = 2.0              # ... and only within this many of the main one's long sides
MEMBER_AREA = (3, 400)        # px^2 at work scale: a cluster member is small, and not a bright star's halo
MEMBER_PEAK_MAX = 0.5         # luminance: brighter is a star
MEMBER_SPREAD_MIN = 0.12      # of the long side: the half-extent a cluster of galaxies gets at least
MEMBER_SPREAD_MAX = 0.35      # of each side: a wider half-spread is field galaxies, not a cluster
MEMBERS_MIN = 10
MEMBER_MIN_CONTRAST = 0.015   # luminance above the local background a member must show (3 MADs of a smooth map is nothing)
GALAXY_CLUSTER_DIFFUSE_MAX = 0.25   # a diffuse subject larger than this on a galaxy-cluster field is not it
EDGE = 0.01                   # of the long side, kept clear of the footprint's edge (fringe)
FULL_FRAME_AREA = 0.70        # a subject this large is not cropped
FRAME_IS_THE_BOX = 0.85       # a box this much of the frame is the frame
STAR_EDGE = 0.015             # of the long side: an edge this close to a bright star is moved
BRIGHT_STAR = 0.85            # luminance of a star an edge must not cut
EMPTY_WEIGHT = 0.25
DENSITY_BLOCK = 0.04          # of the long side: the star-count block
DENSITY_FRAC = 0.3            # density above field median + this * (peak - median) is the cluster
STAR_PEAK_SIGMA = 5.0

MIRROR_QUEUE = os.path.expanduser("~/Desktop/iris_state/live/local/my_instructions.json")
MIRROR_TYPES = os.path.expanduser("~/Desktop/iris_state/live/local/dso_types.json")


# ------------------------------------------------------------------ the type

_KIND_WORDS = {"galaxy": "galaxy", "nebula": "nebula", "reflection": "nebula", "pn": "nebula",
               "snr": "nebula", "cluster": "cluster", "galaxy_cluster": "galaxy_cluster",
               "galaxy cluster": "galaxy_cluster"}
# SIMBAD otype codes the grid files under G that are clusters or groups OF galaxies
_GALAXY_GROUP_CODES = {"ClG", "GrG", "CGG", "PaG", "IG", "GiC", "GiG"}
_CLUSTER_CODES = {"GlC", "Gl?", "OpC", "Cl*", "As*"}
_NEBULA_CODES = {"PN", "HII", "SNR", "RNe", "MoC", "DNe", "glb", "HH", "SFR", "EmO", "Cld", "ISM"}


def type_from_code(code) -> Optional[str]:
    """A SIMBAD otype code (the type cache's value) -> a TYPES word, or None."""
    c = str(code or "").strip()
    if not c:
        return None
    if c in _GALAXY_GROUP_CODES:
        return "galaxy_cluster"
    if c in _CLUSTER_CODES:
        return "cluster"
    if c in _NEBULA_CODES:
        return "nebula"
    if c in ("G", "C", "N"):                   # the first cache wrote letters
        return {"G": "galaxy", "C": "cluster", "N": "nebula"}[c]
    try:
        from iris_astronomy.astro_dso_visibility import _letter_for_otype
        return {"G": "galaxy", "C": "cluster", "N": "nebula"}.get(_letter_for_otype(c))
    except Exception:                          # noqa: BLE001 -- no astropy here
        return None


def type_for(dso: str, root: str, queue_path: str = MIRROR_QUEUE,
             catalogue: Optional[str] = None, caches: tuple = ()) -> str:
    """The object's type for the cropper, OFFLINE: the queue row's obj_type
    (the mirrored queue on the Spark), then configs/candidates.json's kind,
    then the type cache (the repo's, then the mirror's). "unknown" otherwise.
    A monitor's "star" is a cluster field for cropping purposes (the
    sequence stars are the subject)."""
    key = str(dso or "").lower().replace(" ", "")
    try:
        with open(queue_path, encoding="utf-8") as fh:
            for row in json.load(fh):
                if str(row.get("dso") or "").lower().replace(" ", "") == key:
                    t = str(row.get("obj_type") or "").lower()
                    if t in ("galaxy", "nebula"):
                        return t
                    if t == "star":
                        return "cluster"
                    break
    except Exception:                          # noqa: BLE001
        pass
    try:
        with open(catalogue or os.path.join(root, "configs", "candidates.json"), encoding="utf-8") as fh:
            cat = json.load(fh)
        items = cat if isinstance(cat, list) else cat.get("candidates", [])
        for it in items:
            if str(it.get("name") or "").lower().replace(" ", "") == key:
                w = _KIND_WORDS.get(str(it.get("kind") or "").lower())
                if w:
                    return w
    except Exception:                          # noqa: BLE001
        pass
    for path in caches or (os.path.join(root, "local", "dso_types.json"), MIRROR_TYPES):
        try:
            with open(path, encoding="utf-8") as fh:
                w = type_from_code(json.load(fh).get(key))
            if w:
                return w
        except Exception:                      # noqa: BLE001
            continue
    return "unknown"


# ------------------------------------------------------------------ the subject

def luminance(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        return a
    return a[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_work(lum: np.ndarray, work_px: int = WORK_PX) -> tuple[np.ndarray, int]:
    """(luminance at work scale, the integer factor it was reduced by): block means."""
    h, w = lum.shape
    k = max(1, int(math.ceil(max(h, w) / work_px)))
    if k == 1:
        return np.asarray(lum, dtype=np.float32), 1
    hh, ww = h // k * k, w // k * k
    return lum[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3)).astype(np.float32), k


def starless(lum: np.ndarray, median_px: int = MEDIAN_PX, smooth_px: float = SMOOTH_PX) -> np.ndarray:
    from scipy import ndimage
    m = ndimage.median_filter(np.nan_to_num(lum), size=median_px, mode="nearest")
    return ndimage.gaussian_filter(m, smooth_px) if smooth_px > 0 else m


def star_points(lum: np.ndarray, sigma: float = STAR_PEAK_SIGMA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ys, xs, peaks) of the luminance's point sources at work scale: local
    maxima of the luminance minus its 5 px median, above *sigma* MADs."""
    from scipy import ndimage
    a = np.nan_to_num(lum)
    peaks = a - ndimage.median_filter(a, size=5, mode="nearest")
    mad = float(np.median(np.abs(peaks - np.median(peaks)))) * 1.4826
    thr = max(sigma * mad, 1e-4)
    is_max = peaks >= ndimage.maximum_filter(peaks, size=3, mode="nearest")
    ys, xs = np.nonzero(is_max & (peaks > thr))
    return ys, xs, a[ys, xs]


def _disc(r: int) -> np.ndarray:
    r = max(1, int(r))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def region(mask: np.ndarray, weight: np.ndarray, close_px: int, keep_share: float = 0.1,
           keep_reach: float = KEEP_REACH) -> dict:
    """Close *mask* by a disc, keep the main component and any with
    *keep_share* of its flux (*weight* summed inside) whose centre lies
    within *keep_reach* of the main one's long sides: a companion galaxy
    across the field (NGC 1961's) is a different picture. {} when empty."""
    from scipy import ndimage
    if not mask.any():
        return {}
    closed = ndimage.binary_closing(mask, structure=_disc(close_px)) | mask
    labels, n = ndimage.label(closed)
    if n == 0:
        return {}
    idx = np.arange(1, n + 1)
    flux = ndimage.sum(np.clip(weight, 0, None), labels, index=idx)
    im = int(np.argmax(flux)) + 1
    main = float(flux.max())
    objs = ndimage.find_objects(labels)
    my, mx = objs[im - 1]
    long_side = max(my.stop - my.start, mx.stop - mx.start)
    mc = ((my.start + my.stop) / 2.0, (mx.start + mx.stop) / 2.0)
    keep = [im]
    for i, f in enumerate(flux):
        lab = i + 1
        if lab == im or f < keep_share * main:
            continue
        sy, sx = objs[i]
        c = ((sy.start + sy.stop) / 2.0, (sx.start + sx.stop) / 2.0)
        if math.hypot(c[0] - mc[0], c[1] - mc[1]) <= keep_reach * long_side:
            keep.append(lab)
    kept = np.isin(labels, keep)
    ys, xs = np.nonzero(kept)
    w = np.clip(weight[kept], 1e-6, None)
    return {"mask": kept, "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            "centroid": (float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())),
            "area": float(kept.mean()), "components": len(keep)}


def diffuse_subject(sm: np.ndarray, obj_type: str, long_px: int) -> dict:
    """The diffuse-light subject of a starless map: the core at the brightest
    ladder level that finds anything, then each fainter level is taken while
    the region grows like a halo (by less than HALO_GROWTH) and stays in
    band. Faint-first flooded M76's 2 % core to 16 % of background, and a
    galaxy's isophote at 3 % of its core caught every gradient in the field."""
    sky = float(np.median(sm))
    bright = float(np.percentile(sm, 99.5))
    if bright <= sky:
        return {}
    base = ISOPHOTE.get(obj_type, ISOPHOTE["unknown"])
    close_px = max(1, int(CLOSE.get(obj_type, CLOSE["unknown"]) * long_px))
    share = KEEP_SHARE.get(obj_type, KEEP_SHARE["unknown"])
    found, core_area = {}, None
    for mult in ISOPHOTE_LADDER:
        iso = base * mult
        r = region(sm > sky + iso * (bright - sky), sm - sky, close_px, keep_share=share)
        if not r or r["area"] < AREA_BAND[0]:
            continue
        r["isophote"] = iso
        if not found:
            if r["area"] <= AREA_BAND[1]:
                found, core_area = r, r["area"]
            continue
        if r["area"] <= AREA_BAND[1] and r["area"] <= HALO_GROWTH * core_area:
            found = r
        else:
            break
    return found


def density_subject(lum: np.ndarray, long_px: int) -> dict:
    """The star-density subject: blocks whose star count stands above the field's."""
    from scipy import ndimage
    ys, xs, _ = star_points(lum)
    if len(ys) < 20:
        return {}
    blk = max(2, int(DENSITY_BLOCK * long_px))
    h, w = lum.shape
    counts = np.zeros((h // blk + 1, w // blk + 1), dtype=np.float32)
    np.add.at(counts, (ys // blk, xs // blk), 1.0)
    dens = ndimage.gaussian_filter(counts, 1.0)
    med, peak = float(np.median(dens)), float(dens.max())
    if peak <= med * 1.5:
        return {}
    high = dens > med + DENSITY_FRAC * (peak - med)
    full = np.kron(high, np.ones((blk, blk), dtype=bool))[:h, :w]
    dfull = np.kron(dens - med, np.ones((blk, blk), dtype=np.float32))[:h, :w]
    r = region(full, dfull, max(1, int(CLOSE["cluster"] * long_px)))
    if r:
        r["isophote"] = None
    return r


def galaxy_density_subject(sm: np.ndarray, lum: np.ndarray, long_px: int) -> dict:
    """A cluster of galaxies: its members are the SMALL diffuse sources that
    survive the star-removing median (MEMBER_AREA px^2 at work scale, and
    nothing in them as bright as a star, MEMBER_PEAK_MAX: a bright star's
    halo survives the median too). At this depth Abell 2218 gives 14 of
    them, too few for a density map (one or two per block, the "cluster"
    landed wherever two fell together), so the subject is the members'
    robust centre and spread: the median position, and 1.5 times the
    interquartile range on each axis, floored at MEMBER_SPREAD_MIN of the
    long side."""
    from scipy import ndimage
    back = ndimage.gaussian_filter(sm, max(2.0, 0.1 * long_px))
    res = sm - back
    mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
    labels, n = ndimage.label(res > max(3.0 * mad, MEMBER_MIN_CONTRAST))
    if n < 5:
        return {}
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(np.ones_like(res), labels, index=idx)
    peaks = ndimage.maximum(lum, labels, index=idx)
    small = [i + 1 for i, (a, pk) in enumerate(zip(areas, peaks))
             if MEMBER_AREA[0] <= a <= MEMBER_AREA[1] and pk < MEMBER_PEAK_MAX]
    if len(small) < 5:
        return {}
    cents = ndimage.center_of_mass(res, labels, small)
    ys = np.array([c[0] for c in cents])
    xs = np.array([c[1] for c in cents])
    h, w = sm.shape
    cx, cy = float(np.median(xs)), float(np.median(ys))
    floor = MEMBER_SPREAD_MIN * long_px
    hx = max(1.5 * (np.percentile(xs, 75) - np.percentile(xs, 25)), floor)
    hy = max(1.5 * (np.percentile(ys, 75) - np.percentile(ys, 25)), floor)
    if len(small) < MEMBERS_MIN or hx > MEMBER_SPREAD_MAX * w or hy > MEMBER_SPREAD_MAX * h:
        return {}                              # scattered: not a cluster the members can place
    x0, x1 = int(max(0, cx - hx)), int(min(w, cx + hx))
    y0, y1 = int(max(0, cy - hy)), int(min(h, cy + hy))
    mask = np.isin(labels, small)
    mask[:y0, :] = mask[y1:, :] = False
    mask[:, :x0] = mask[:, x1:] = False
    return {"mask": mask, "bbox": (x0, y0, x1, y1), "centroid": (cx, cy),
            "area": (x1 - x0) * (y1 - y0) / float(h * w), "components": len(small),
            "isophote": None, "members": len(small)}


def subject(lum: np.ndarray, obj_type: str) -> dict:
    """The subject region of a work-scale luminance for *obj_type*. {} if none."""
    long_px = max(lum.shape)
    sm = starless(lum)
    if obj_type == "cluster":
        d = density_subject(lum, long_px)
        n = diffuse_subject(sm, "nebula", long_px)
        if d and n and AREA_BAND[0] <= n["area"] <= AREA_BAND[1]:
            mask = d["mask"] | n["mask"]
            r = region(mask, np.where(mask, 1.0, 0.0), 1)
            r["isophote"] = n.get("isophote")
            r["source"] = "stars+nebula"
            return r
        r = d or n
        if r:
            r["source"] = "stars" if d else "nebula"
        return r
    if obj_type == "galaxy_cluster":
        r = galaxy_density_subject(sm, lum, long_px)
        if r:
            r["source"] = "members"
            return r
        # No members to place it by: the diffuse map is bright stars' halos
        # and gradients on such a field (Abell 2218 at this depth), so only
        # a compact diffuse subject counts; a flood is "no subject".
        r = diffuse_subject(sm, obj_type, long_px)
        if r and r["area"] <= GALAXY_CLUSTER_DIFFUSE_MAX:
            r["source"] = "diffuse"
            return r
        return {}
    r = diffuse_subject(sm, obj_type, long_px)
    if r:
        r["source"] = "diffuse"
    return r


# ------------------------------------------------------------------ the box

def _clamp_box(cx: float, cy: float, bw: float, bh: float, W: int, H: int, edge: int) -> list:
    """Shrink to fit the frame KEEPING THE ASPECT (the margin gives on both
    axes), then shift inside. Shrinking one axis alone made Melotte 15's
    4:3 box taller than the field, and the shift put the cluster at the
    bottom of a box of empty sky."""
    scale = min(1.0, (W - 2 * edge) / max(bw, 1e-6), (H - 2 * edge) / max(bh, 1e-6))
    bw, bh = bw * scale, bh * scale
    x0 = min(max(cx - bw / 2, edge), W - edge - bw)
    y0 = min(max(cy - bh / 2, edge), H - edge - bh)
    return [int(round(x0)), int(round(y0)), int(round(x0 + bw)), int(round(y0 + bh))]


def _nudge_off_stars(box: list, stars: tuple, W: int, H: int, edge: int, near: int) -> list:
    """Move any edge that passes within *near* px of a bright star: outward
    when the frame allows it within 2 * near, else inward past the star.
    Outward only grew NGC 2146's box by a quarter in a star-rich field."""
    x0, y0, x1, y1 = box

    def low(v, sv, lim):             # a low edge (x0 / y0) near a star at sv
        out = int(sv - near - 1)
        return max(lim, out) if out >= lim and v - out <= 2 * near else int(sv + near + 2)

    def high(v, sv, lim):            # a high edge (x1 / y1)
        out = int(sv + near + 2)
        return min(lim, out) if out <= lim and out - v <= 2 * near else int(sv - near - 1)

    ys, xs = stars
    for sy, sx in zip(ys, xs):
        if y0 - near <= sy <= y1 + near:
            if abs(sx - x0) <= near:
                x0 = low(x0, sx, edge)
            if abs(sx - x1) <= near:
                x1 = high(x1, sx, W - edge)
        if x0 - near <= sx <= x1 + near:
            if abs(sy - y0) <= near:
                y0 = low(y0, sy, edge)
            if abs(sy - y1) <= near:
                y1 = high(y1, sy, H - edge)
    return [int(x0), int(y0), int(x1), int(y1)]


def fit_box(sub: dict, shape: tuple, obj_type: str, stars: tuple = ((), ())) -> dict:
    """The best box for *sub* (from subject()) on a field of *shape* (H, W)
    at work scale: the subject's box grown by the type's margin, widened to
    each aspect, clamped, nudged off bright stars, scored. Pure."""
    H, W = shape
    long_px = max(H, W)
    edge = max(1, int(EDGE * long_px))
    near = max(1, int(STAR_EDGE * long_px))
    m = MARGIN.get(obj_type, MARGIN["unknown"])
    sx0, sy0, sx1, sy1 = sub["bbox"]
    bw0, bh0 = (sx1 - sx0) * (1 + 2 * m), (sy1 - sy0) * (1 + 2 * m)
    floor = MIN_BOX_SHARE.get(obj_type, MIN_BOX_SHARE["unknown"]) * long_px
    if max(bw0, bh0) < floor:                  # a small subject keeps its context
        f = floor / max(bw0, bh0, 1e-6)
        bw0, bh0 = bw0 * f, bh0 * f
    cx, cy = (sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0
    own = bw0 / max(bh0, 1e-6)
    mask = sub["mask"]
    total = float(mask.sum()) or 1.0
    best = None
    for name, a in ASPECTS:
        bw, bh = bw0, bh0
        if bw / max(bh, 1e-6) < a:
            bw = bh * a
        else:
            bh = bw / a
        box = _clamp_box(cx, cy, bw, bh, W, H, edge)
        box = _nudge_off_stars(box, stars, W, H, edge, near)
        x0, y0, x1, y1 = box
        inside = mask[y0:y1, x0:x1]
        coverage = float(inside.sum()) / total
        empty = 1.0 - float(inside.mean()) if inside.size else 1.0
        gap = abs(math.log(a / own))
        score = coverage - EMPTY_WEIGHT * empty - SHAPE_WEIGHT * gap - (SQUARE_PENALTY if a == 1.0 else 0.0)
        ratio = (x1 - x0) / max(y1 - y0, 1)
        cand = {"aspect": name if abs(ratio / a - 1) <= 0.03 else f"{name}~{ratio:.2f}",
                "box": box, "coverage": round(coverage, 3),
                "empty": round(empty, 3), "score": round(score, 4)}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def propose(rgb: np.ndarray, obj_type: str = "unknown", work_px: int = WORK_PX) -> dict:
    """The crop for a finished composite (H, W, 3) in 0..1, or a luminance.

    Returns {"box": [x0, y0, x1, y1] on the full array, "full": bool, "type",
    "aspect", "coverage", "empty", "subject": {bbox, area, source, isophote,
    components}, "why"}. "full" True means the full frame IS the answer and
    the box is the whole array."""
    obj_type = obj_type if obj_type in TYPES else "unknown"
    lum = luminance(rgb)
    H, W = lum.shape
    out = {"type": obj_type, "shape": [int(H), int(W)], "box": [0, 0, int(W), int(H)],
           "full": True, "aspect": None, "coverage": None, "empty": None, "subject": None, "why": ""}
    try:
        wl, k = to_work(lum, work_px)
        sub = subject(wl, obj_type)
    except Exception as e:                     # noqa: BLE001 -- a crop must never fail a render
        out["why"] = f"no crop: {type(e).__name__}: {e}"
        return out
    if not sub:
        out["why"] = "no crop: no subject found"
        return out
    out["subject"] = {"bbox": [int(v * k) for v in sub["bbox"]], "area": round(sub["area"], 3),
                      "source": sub.get("source"), "isophote": sub.get("isophote"),
                      "components": sub.get("components")}
    if sub["area"] >= FULL_FRAME_AREA:
        out["why"] = f"no crop: the subject is {100 * sub['area']:.0f}% of the field"
        return out
    ys, xs, pk = star_points(wl)
    bright = pk >= BRIGHT_STAR
    fit = fit_box(sub, wl.shape, obj_type, (ys[bright], xs[bright]))
    x0, y0, x1, y1 = fit["box"]
    box = [x0 * k, y0 * k, min(x1 * k, W), min(y1 * k, H)]
    if (box[2] - box[0]) * (box[3] - box[1]) >= FRAME_IS_THE_BOX * W * H:
        out["why"] = "no crop: the box is the frame"
        return out
    out.update({"box": [int(v) for v in box], "full": False, "aspect": fit["aspect"],
                "coverage": fit["coverage"], "empty": fit["empty"],
                "why": f"{obj_type} subject {100 * sub['area']:.0f}% of the field ({sub.get('source')})"})
    return out


def crop(rgb: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in box]
    return np.asarray(rgb)[y0:y1, x0:x1]


def scaled_box(box, shape, max_px: Optional[int]) -> list:
    """The box on a picture written with save_rgb(max_px=): the same scale rule."""
    H, W = shape[:2]
    s = min(1.0, max_px / max(H, W)) if max_px and max(H, W) > max_px else 1.0
    return [int(round(v * s)) for v in box]


def describe(rec: dict) -> str:
    """One log / push line."""
    if rec.get("full"):
        return f"crop (shadow): full frame -- {rec.get('why', '')}"
    x0, y0, x1, y1 = rec["box"]
    return (f"crop (shadow): {rec['aspect']} {x1 - x0}x{y1 - y0} at ({x0},{y0}), "
            f"{rec['why']}, coverage {rec['coverage']:.2f}, empty {rec['empty']:.2f}")


def iou(a, b) -> float:
    """Intersection over union of two boxes [x0, y0, x1, y1]. Pure."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
