"""
Color–magnitude (H–R) diagram from imaging subs, calibrated against Gaia DR3.

A true Hertzsprung–Russell diagram plots luminosity vs temperature; from images
we build its observable cousin, a *color–magnitude diagram* (CMD): a colour index
(temperature proxy, x-axis) vs an apparent magnitude (brightness proxy, y-axis).
That needs two filters.

Pipeline (see :func:`build_cmd`):
  1. Stack the blue and red filter subs separately (registered).
  2. Plate-solve each stack with ASTAP → a WCS.
  3. Detect + aperture-measure stars in each stack (``sep``) → instrumental mags.
  4. Cross-match the two filters' star lists by sky position → one colour per star.
  5. One Gaia DR3 cone query over the field; cross-match to set a per-filter zero
     point (blue→BP, red→RP), so the axes sit on the Gaia photometric system.
  6. Plot calibrated colour vs magnitude (brightest at top), with the Gaia field
     CMD drawn faintly behind as a reference sequence.

Everything is fail-safe at the edges (missing solver, no Gaia, too few stars):
the caller gets a clear message rather than a traceback. The best results come
from a star *cluster* (open or globular), where all stars share one distance and
the main sequence / giant branch / turnoff actually line up.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_logger = logging.getLogger(__name__)

# Rough blue→red ordering by effective wavelength, used to decide which of two
# filters is the "blue" and which the "red" when the caller doesn't say.
_FILTER_WAVELENGTH_RANK = {
    "B": 1, "OIII": 2, "G": 3, "CLEAR": 4, "L": 4, "R": 5, "HA": 6, "SII": 7,
}


def filter_rank(name: str) -> int:
    """Blue→red ordering rank for a filter name (lower = bluer). Unknown → 4."""
    return _FILTER_WAVELENGTH_RANK.get(name.strip().upper(), 4)


def choose_filters(
    by_filter: dict[str, list],
    min_frames: int = 1,
) -> Optional[tuple[str, str]]:
    """Pick a (blue, red) filter pair from the available filter→paths mapping.

    Prefers the two filters with the most subs, ordered blue→red by effective
    wavelength. Returns None if fewer than two usable filters are present.
    """
    usable = {f: p for f, p in by_filter.items() if len(p) >= min_frames}
    if len(usable) < 2:
        return None
    # Two most-imaged filters, then order by wavelength rank.
    top2 = sorted(usable, key=lambda f: len(usable[f]), reverse=True)[:2]
    blue, red = sorted(top2, key=filter_rank)
    if filter_rank(blue) == filter_rank(red):
        return None
    return blue, red


def _plate_solve_wcs(
    data: np.ndarray,
    ra_deg: Optional[float],
    dec_deg: Optional[float],
    fov_deg: float,
    astap_exe: str,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Plate-solve a stacked image array with ASTAP and return an astropy WCS.

    Writes the array to a throwaway FITS (the imaging data is never touched),
    passing the field centre + field-of-view as hints. Returns None on any
    failure (missing solver, no solution, unreadable WCS).
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    if not astap_exe or not os.path.exists(astap_exe):
        if progress_cb:
            progress_cb("ASTAP not found — cannot plate-solve")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="cmd_solve_")
    tmp_fits = Path(tmp_dir) / "stack.fits"
    try:
        fits.writeto(tmp_fits, np.asarray(data, dtype=np.float32), overwrite=True)
        cmd = [astap_exe, "-f", str(tmp_fits), "-fov", f"{fov_deg:.4f}",
               "-r", "5", "-update"]
        if ra_deg is not None and dec_deg is not None:
            cmd += ["-ra", f"{ra_deg / 15.0:.6f}", "-spd", f"{dec_deg + 90.0:.6f}"]
        subprocess.run(cmd, capture_output=True, timeout=300)
        solved = fits.getheader(tmp_fits)
        if "CRVAL1" not in solved:
            if progress_cb:
                progress_cb("plate solve failed (no WCS in solution)")
            return None
        return WCS(solved)
    except Exception:
        _logger.exception("cmd_diagram plate solve failed")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _detect_and_measure(data: np.ndarray, thresh_sigma: float = 5.0):
    """Detect stars and aperture-measure background-subtracted flux with sep.

    Returns (x, y, instrumental_mag) arrays for stars with positive flux. The
    aperture radius is scaled to the typical stellar profile in the frame.

    Dense globular clusters (e.g. M13) put a very large number of connected
    pixels above threshold; sep's deblend pixel buffer (default 300k) overflows
    on a multi-megapixel stack. We raise that ceiling up front, and if extract
    still overflows we step the threshold up and retry rather than aborting the
    whole CMD job.
    """
    import sep

    # 4M-pixel deblend buffer: comfortably covers a crowded globular core in a
    # ~60 Mpx stack without a meaningful memory cost.
    try:
        sep.set_extract_pixstack(4_000_000)
    except Exception:
        pass

    data = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(data)
    data_sub = data - bkg.back()

    objs = None
    thresh = thresh_sigma
    for _attempt in range(4):
        try:
            objs = sep.extract(data_sub, thresh=thresh, err=bkg.globalrms)
            break
        except Exception as exc:
            # The pixstack-overflow case is recoverable: a higher threshold
            # admits far fewer core pixels. Anything else is a real failure.
            if "pixel buffer full" not in str(exc):
                raise
            thresh *= 1.5
            _logger.warning(
                "sep.extract pixstack overflow; retrying at thresh=%.1fσ", thresh)
    if objs is None:
        raise RuntimeError(
            "star detection failed: field too crowded even at "
            f"{thresh / 1.5:.1f}σ (sep pixel buffer exhausted)")
    if len(objs) == 0:
        return np.array([]), np.array([]), np.array([])

    # Aperture ~ a couple of profile sigmas; sep 'a','b' are profile RMS sizes.
    ab = np.sqrt(np.clip(objs["a"] * objs["b"], 1e-6, None))
    aper_r = float(np.median(ab)) * 2.5
    aper_r = max(2.5, min(aper_r, 12.0))

    flux, _flux_err, flag = sep.sum_circle(
        data_sub, objs["x"], objs["y"], aper_r, gain=1.0
    )
    flux = np.asarray(flux, dtype=np.float64)
    good = np.isfinite(flux) & (flux > 0) & (np.asarray(flag) == 0)
    x = np.asarray(objs["x"])[good]
    y = np.asarray(objs["y"])[good]
    mag = -2.5 * np.log10(flux[good])
    return x, y, mag


def _robust_zero_point(offsets: np.ndarray, n_sigma: float = 3.0) -> Optional[float]:
    """Sigma-clipped median of (catalog − instrumental) magnitude offsets."""
    vals = offsets[np.isfinite(offsets)]
    if vals.size < 5:
        return None
    for _ in range(3):
        med = np.median(vals)
        sd = np.std(vals)
        if sd == 0:
            break
        keep = np.abs(vals - med) < n_sigma * sd
        if keep.all() or keep.sum() < 5:
            break
        vals = vals[keep]
    return float(np.median(vals))


def _select_members(pmra, pmdec, parallax) -> "np.ndarray":
    """Boolean mask of likely cluster members from Gaia astrometry.

    Cluster members share a common proper motion and parallax (distance); field
    stars scatter across both. Seed the proper-motion clump from the densest bin
    of the 2-D PM histogram, then iteratively sigma-clip in (pmra, pmdec,
    parallax) to converge on that dominant concentration. Returns an all-False
    mask when there's too little astrometry or no clear clump (e.g. a non-cluster
    field), so callers can fall back gracefully.
    """
    pmra = np.asarray(pmra, dtype=np.float64)
    pmdec = np.asarray(pmdec, dtype=np.float64)
    plx = np.asarray(parallax, dtype=np.float64)
    out = np.zeros(pmra.shape, dtype=bool)
    valid = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(plx)
    if valid.sum() < 20:
        return out

    a, d, p = pmra[valid], pmdec[valid], plx[valid]

    # Seed the PM centre at the densest histogram bin, over a robust range.
    lo_a, hi_a = np.percentile(a, [1, 99])
    lo_d, hi_d = np.percentile(d, [1, 99])
    if hi_a <= lo_a or hi_d <= lo_d:
        return out
    hist, ea, ed = np.histogram2d(a, d, bins=64, range=[[lo_a, hi_a], [lo_d, hi_d]])
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    ca, cd = 0.5 * (ea[i] + ea[i + 1]), 0.5 * (ed[j] + ed[j + 1])

    # Start from stars near that peak, then refine by joint sigma-clipping.
    sel = (np.abs(a - ca) < max(2.0, 0.1 * (hi_a - lo_a))) & \
          (np.abs(d - cd) < max(2.0, 0.1 * (hi_d - lo_d)))
    if sel.sum() < 10:
        return out

    def _sigma(x, floor):
        s = 1.4826 * np.median(np.abs(x - np.median(x)))
        return max(float(s), floor)

    for _ in range(6):
        ca, cd, cp = np.median(a[sel]), np.median(d[sel]), np.median(p[sel])
        sa, sd, sp = _sigma(a[sel], 0.3), _sigma(d[sel], 0.3), _sigma(p[sel], 0.05)
        keep = ((np.abs(a - ca) < 3 * sa) & (np.abs(d - cd) < 3 * sd)
                & (np.abs(p - cp) < 3 * sp))
        if keep.sum() < 10:
            break
        if keep.sum() == sel.sum():
            sel = keep
            break
        sel = keep

    if sel.sum() < 10:
        return out
    out[np.where(valid)[0][sel]] = True
    return out


def build_cmd(
    blue_paths: list[Path],
    red_paths: list[Path],
    blue_name: str,
    red_name: str,
    dso_name: str,
    output_path: Path,
    astap_exe: str,
    arcsec_per_pixel: float,
    match_radius_arcsec: float = 2.0,
    gaia_mag_limit: float = 19.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    precomputed_fwhm_stars: Optional[dict] = None,
    observatory_name: str = "this telescope",
) -> dict:
    """Build a Gaia-calibrated colour–magnitude diagram and save it as a JPEG.

    Returns a stats dict: ``{n_stars, n_gaia, zp_blue, zp_red, output}``.
    Raises ``RuntimeError`` with a human-readable message on any step that makes
    a diagram impossible (no WCS, too few stars, no Gaia matches).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from stacking import stacker

    def _say(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def _ckpt() -> None:
        if cancel_cb and cancel_cb():
            from utils.cancellation import Cancelled
            raise Cancelled()

    # ── 1. Stack both filters ────────────────────────────────────────────────
    stacks: dict[str, np.ndarray] = {}
    headers: dict[str, fits.Header] = {}
    for name, paths in ((blue_name, blue_paths), (red_name, red_paths)):
        _ckpt()
        _say(f"stacking {len(paths)} {name} subs…")
        data, _info = stacker.stack(
            paths, register=True, progress_cb=lambda m, n=name: _say(f"[{n}] {m}"),
            cancel_cb=cancel_cb, precomputed_fwhm_stars=precomputed_fwhm_stars,
        )
        stacks[name] = data
        headers[name] = fits.getheader(paths[0])

    # ── 2. Plate-solve each stack → WCS ──────────────────────────────────────
    h, w = stacks[red_name].shape
    fov_deg = (max(h, w) * arcsec_per_pixel) / 3600.0
    wcs: dict = {}
    for name in (blue_name, red_name):
        _ckpt()
        _say(f"plate-solving {name} stack with ASTAP…")
        hdr = headers[name]
        try:
            ra_deg = float(hdr.get("RA"))
            dec_deg = float(hdr.get("DEC"))
        except (TypeError, ValueError):
            ra_deg = dec_deg = None
        solved = _plate_solve_wcs(
            stacks[name], ra_deg, dec_deg, fov_deg, astap_exe, progress_cb
        )
        if solved is None:
            raise RuntimeError(
                f"Could not plate-solve the {name} stack — check ASTAP and its "
                f"star database. A CMD needs a WCS to match stars to Gaia."
            )
        wcs[name] = solved

    # ── 3. Detect + measure stars in each stack ──────────────────────────────
    stars: dict = {}
    for name in (blue_name, red_name):
        _ckpt()
        x, y, mag = _detect_and_measure(stacks[name])
        if x.size < 10:
            raise RuntimeError(f"Only {x.size} stars measured in the {name} "
                               f"stack — too few for a diagram.")
        sky = wcs[name].pixel_to_world(x, y)
        stars[name] = {"mag": mag, "sky": SkyCoord(sky)}
        _say(f"{name}: measured {x.size} stars")

    # ── 4. Cross-match blue ↔ red by sky position ────────────────────────────
    _ckpt()
    idx, sep2d, _ = stars[blue_name]["sky"].match_to_catalog_sky(stars[red_name]["sky"])
    paired = sep2d.arcsec <= match_radius_arcsec
    if paired.sum() < 10:
        raise RuntimeError(f"Only {int(paired.sum())} stars matched between "
                           f"{blue_name} and {red_name}. Are the frames of the "
                           f"same field, and did both solve?")
    blue_mag = stars[blue_name]["mag"][paired]
    red_mag = stars[red_name]["mag"][idx[paired]]
    pair_sky = stars[blue_name]["sky"][paired]
    _say(f"matched {paired.sum()} stars across both filters")

    # ── 5. Gaia field query + per-filter zero point (blue→BP, red→RP) ─────────
    _ckpt()
    _say("querying Gaia DR3 for the field…")
    center = SkyCoord(pair_sky.ra.mean(), pair_sky.dec.mean())
    radius_deg = fov_deg * 0.6  # half-FOV plus margin
    gaia_tbl = _query_gaia(center, radius_deg, gaia_mag_limit, progress_cb)
    if gaia_tbl is None or len(gaia_tbl) < 5:
        raise RuntimeError("Gaia query returned too few sources — no calibration "
                           "possible (check network / astroquery).")
    gaia_sky = SkyCoord(ra=np.asarray(gaia_tbl["ra"]) * u.deg,
                        dec=np.asarray(gaia_tbl["dec"]) * u.deg)
    g_bp = np.asarray(gaia_tbl["phot_bp_mean_mag"], dtype=np.float64)
    g_rp = np.asarray(gaia_tbl["phot_rp_mean_mag"], dtype=np.float64)
    g_g = np.asarray(gaia_tbl["phot_g_mean_mag"], dtype=np.float64)
    g_pmra = np.asarray(gaia_tbl["pmra"], dtype=np.float64)
    g_pmdec = np.asarray(gaia_tbl["pmdec"], dtype=np.float64)
    g_plx = np.asarray(gaia_tbl["parallax"], dtype=np.float64)

    g_idx, g_sep, _ = pair_sky.match_to_catalog_sky(gaia_sky)
    has_gaia = g_sep.arcsec <= match_radius_arcsec
    zp_blue = _robust_zero_point(g_bp[g_idx[has_gaia]] - blue_mag[has_gaia])
    zp_red = _robust_zero_point(g_rp[g_idx[has_gaia]] - red_mag[has_gaia])
    if zp_blue is None or zp_red is None:
        raise RuntimeError(f"Only {int(has_gaia.sum())} stars matched to Gaia — "
                           f"not enough to calibrate. A star cluster gives the "
                           f"cleanest result.")

    color = (blue_mag + zp_blue) - (red_mag + zp_red)
    mag = red_mag + zp_red
    _say(f"calibrated: ZP({blue_name})={zp_blue:.2f}, ZP({red_name})={zp_red:.2f}; "
         f"{int(has_gaia.sum())} Gaia anchors")

    # ── 6. Gaia cluster membership (proper motion + parallax) ─────────────────
    member = _select_members(g_pmra, g_pmdec, g_plx)
    mem_color, mem_g = (g_bp - g_rp)[member], g_g[member]
    _say(f"Gaia membership: {int(member.sum())}/{len(g_g)} field stars share the "
         f"cluster proper motion + parallax")

    # ── 7. Morphology → rough age from the Δmag(turn-off − HB) gap ────────────
    anchors = _cluster_age(_branch_anchors(mem_color, mem_g), band=red_name)
    if anchors.get("age") is not None:
        _say(f"morphology: Δ{red_name}(TO−HB) ≈ {anchors['delta_mag']:.2f} mag "
             f"→ age ≈ {anchors['age']:.0f} Gyr (rough)")

    # ── 8. Plot ──────────────────────────────────────────────────────────────
    _ckpt()
    _plot_cmd(color, mag, g_bp - g_rp, g_g, mem_color, mem_g, blue_name, red_name,
              dso_name, int(has_gaia.sum()), output_path,
              observatory_name=observatory_name, anchors=anchors)

    return {
        "n_stars": int(color.size),
        "n_gaia": int(has_gaia.sum()),
        "n_members": int(member.sum()),
        "age_gyr": anchors.get("age"),
        "delta_mag": anchors.get("delta_mag"),
        "zp_blue": round(zp_blue, 3),
        "zp_red": round(zp_red, 3),
        "output": str(output_path),
    }


def _query_gaia(center, radius_deg: float, mag_limit: float, progress_cb):
    """Cone-search Gaia DR3 for sources in the field, brighter than mag_limit."""
    try:
        from astroquery.gaia import Gaia
        adql = (
            "SELECT source_id, ra, dec, phot_g_mean_mag, "
            "phot_bp_mean_mag, phot_rp_mean_mag, pmra, pmdec, parallax "
            "FROM gaiadr3.gaia_source "
            "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {center.ra.deg:.6f}, {center.dec.deg:.6f}, {radius_deg:.5f})) "
            f"AND phot_g_mean_mag < {mag_limit} "
            "AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL"
        )
        job = Gaia.launch_job_async(adql)
        return job.get_results()
    except Exception:
        _logger.exception("Gaia query failed")
        if progress_cb:
            progress_cb("Gaia query failed")
        return None


def _branch_anchors(m_color, m_mag) -> dict:
    """Robust morphological anchors for the member CMD (heuristic, not a fit).

    Returns a dict that may contain:
      - ``hb_mag``: horizontal-branch / RR-Lyrae magnitude (the pile-up of the
        bluer-half members),
      - ``rgb`` (colour, mag): a point on the red giant branch (red ridge above
        the HB),
      - ``to`` (colour, mag): the main-sequence turn-off — the bright, blue
        corner of the sequence below the HB.
    Empty when the member sample is too sparse to be meaningful.
    """
    anchors: dict = {}
    finite = np.isfinite(m_color) & np.isfinite(m_mag)
    c, m = np.asarray(m_color)[finite], np.asarray(m_mag)[finite]
    if c.size < 30:
        return anchors

    # Horizontal branch: among the bluer members, the HB is the *brightest*
    # concentration. The lower main sequence is also blue-ish but ~3 mag fainter
    # and far more numerous, so it forms the dominant (faint) histogram peak,
    # with a sparse gap above it (the instability strip). The HB is therefore the
    # brightest peak on the bright side of that dominant faint peak — a floor
    # tied to the global max fails because the faint MS dwarfs the HB.
    bm = m[c < np.percentile(c, 35)]
    if bm.size >= 15:
        h, edges = np.histogram(bm, bins=25)
        centers = 0.5 * (edges[:-1] + edges[1:])
        gmax = int(np.argmax(h))
        if centers[gmax] <= np.median(bm):
            anchors["hb_mag"] = float(centers[gmax])        # bright peak dominates → it's the HB
        else:
            # The faint MS pile (peak + its rising flank) dwarfs the HB, so look
            # well above it — brightward of ~1.5 mag over the gap — for the HB peak.
            region = centers < (centers[gmax] - 1.5)
            if region.any():
                masked = np.where(region, h, 0)
                if masked.max() >= max(8.0, 0.1 * float(h.max())):
                    anchors["hb_mag"] = float(centers[int(np.argmax(masked))])
    hb = anchors.get("hb_mag", float(np.median(m)))

    # Red giant branch: red ridge brighter than the HB (for the label).
    red = (c > np.percentile(c, 75)) & (m < hb)
    if red.sum() >= 5:
        anchors["rgb"] = (float(np.median(c[red])), float(np.median(m[red])))

    # Main-sequence turn-off: the bright edge of the dense main-sequence pile
    # well below the HB (the brightest MS stars, where the MS bends toward the
    # subgiants). A bright percentile of the sub-HB members is robust to the
    # bluest-straggler outliers that an extreme-blue cut would latch onto.
    sub = m > hb + 1.5
    if sub.sum() >= 10:
        cs, ms = c[sub], m[sub]
        to_m = float(np.percentile(ms, 20))
        near = np.abs(ms - to_m) < 0.3
        to_c = float(np.median(cs[near])) if near.any() else float(np.median(cs))
        anchors["to"] = (to_c, to_m)
    return anchors


def _cluster_age(anchors: dict, band: str = "mag") -> dict:
    """Augment *anchors* with a rough age from the Δmag(turn-off − HB) gap.

    The vertical method: the HB is a near-standard candle, so the turn-off's
    magnitude *relative* to it is distance/reddening-independent and tracks age.
    Calibration is approximate (V-band: ΔV≈3.5 ↔ ~13 Gyr, ~6.7 Gyr/mag) and is
    applied here to an instrumental band, so treat the age as a ballpark only.
    """
    anchors["band"] = band
    hb, to = anchors.get("hb_mag"), anchors.get("to")
    if hb is None or to is None:
        return anchors
    dv = to[1] - hb            # turn-off is fainter than the HB → positive
    if dv <= 0:
        return anchors
    anchors["delta_mag"] = float(dv)
    anchors["age"] = float(min(15.0, max(1.0, 13.0 + 6.7 * (dv - 3.5))))
    return anchors


def _annotate_branches(ax, anchors: dict) -> None:
    """Draw the branch labels and the Δmag(TO−HB) age gauge from *anchors*."""
    LABEL, ARROW, AGE = "#e6edf3", "#8b949e", "#f0a500"
    hb = anchors.get("hb_mag")
    band = anchors.get("band", "mag")
    if hb is not None:
        ax.axhline(hb, color=ARROW, ls="--", lw=0.8, alpha=0.7)
        ax.text(0.02, hb, "horizontal branch · RR Lyrae level", color=LABEL,
                fontsize=8, va="bottom", ha="left",
                transform=ax.get_yaxis_transform())
    if "rgb" in anchors:
        rc, rm = anchors["rgb"]
        ax.annotate("red giant branch", xy=(rc, rm), xytext=(rc + 0.35, rm),
                    color=LABEL, fontsize=8, va="center", ha="left",
                    arrowprops=dict(arrowstyle="->", color=ARROW, lw=0.8))
    if "to" in anchors:
        tc, tm = anchors["to"]
        ax.annotate("main-sequence turn-off", xy=(tc, tm),
                    xytext=(tc + 0.4, tm + 0.4), color=LABEL, fontsize=8,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="->", color=ARROW, lw=0.8))
    # Δmag(TO−HB) age gauge: a vertical caliper between the two levels.
    if hb is not None and "to" in anchors and anchors.get("age") is not None:
        tc, tm = anchors["to"]
        ax.annotate("", xy=(tc, hb), xytext=(tc, tm),
                    arrowprops=dict(arrowstyle="<->", color=AGE, lw=1.3))
        ax.text(tc + 0.06, 0.5 * (hb + tm),
                f"Δ{band}(TO−HB) ≈ {anchors['delta_mag']:.2f}\n≈ {anchors['age']:.0f} Gyr (rough)",
                color=AGE, fontsize=8, va="center", ha="left")


def _plot_cmd(color, mag, gaia_color, gaia_g, mem_color, mem_g, blue_name, red_name,
              dso_name, n_gaia, output_path: Path,
              observatory_name: str = "this telescope", anchors: "Optional[dict]" = None) -> None:
    """Render the colour–magnitude diagram to a JPEG (dark theme).

    Four panels (2×2) sharing axes: the measured stars, the full Gaia DR3 field,
    the Gaia stars selected as cluster members by proper motion + parallax (the
    field stripped away → the clean cluster locus), and an annotated copy of the
    member sequence with the main evolutionary branches labelled.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, GAIA, FG = "#0d1117", "#30475e", "#c9d1d9"
    CMAP = "RdYlBu_r"  # blue (small B−R, hot) → red (large B−R, cool)

    finite_g = np.isfinite(gaia_color) & np.isfinite(gaia_g)
    g_color, g_mag = gaia_color[finite_g], gaia_g[finite_g]
    finite_m = np.isfinite(mem_color) & np.isfinite(mem_g)
    m_color, m_mag = mem_color[finite_m], mem_g[finite_m]
    color, mag = np.asarray(color), np.asarray(mag)

    # Shared colour/magnitude ranges over all populations: the panels line up,
    # and a point's hue matches its x-position (same scale as the axis).
    all_color = np.concatenate([color, g_color]) if g_color.size else color
    all_mag = np.concatenate([mag, g_mag]) if g_mag.size else mag
    cmin, cmax = (float(np.nanmin(all_color)), float(np.nanmax(all_color))) if all_color.size else (0.0, 1.0)
    mmin, mmax = (float(np.nanmin(all_mag)), float(np.nanmax(all_mag))) if all_mag.size else (0.0, 1.0)

    def _draw_temp(ax, x, y, alpha=0.9):
        # False-colour each star by its colour index: blue = hot, red = cool.
        return ax.scatter(x, y, s=12, c=x, cmap=CMAP, vmin=cmin, vmax=cmax,
                          alpha=alpha, linewidths=0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=110,
                             sharex=True, sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(BG)
    a_meas, a_field, a_mem, a_ann = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    sc = _draw_temp(a_meas, color, mag)
    a_meas.set_title(f"Measured ({observatory_name}) — {color.size} stars", color="#e6edf3")

    if g_color.size:
        a_field.scatter(g_color, g_mag, s=6, c=GAIA, alpha=0.6, linewidths=0)
    a_field.set_title(f"Gaia DR3 field — {g_color.size} stars", color="#e6edf3")

    if m_color.size:
        _draw_temp(a_mem, m_color, m_mag)
        a_mem.set_title(f"Gaia members (PM + parallax) — {m_color.size} stars",
                        color="#e6edf3")
        _draw_temp(a_ann, m_color, m_mag, alpha=0.55)
        if anchors is None:
            anchors = _cluster_age(_branch_anchors(m_color, m_mag), band=red_name)
        _annotate_branches(a_ann, anchors)
        a_ann.set_title("Annotated members", color="#e6edf3")
    else:
        for ax in (a_mem, a_ann):
            ax.text(0.5, 0.5, "no cluster members\n(no PM/parallax clump)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#8b949e", fontsize=11)
        a_mem.set_title("Gaia members (PM + parallax) — none", color="#e6edf3")
        a_ann.set_title("Annotated members — none", color="#e6edf3")

    if all_color.size:
        cpad, mpad = 0.05 * (cmax - cmin + 1e-6), 0.05 * (mmax - mmin + 1e-6)
        a_meas.set_xlim(cmin - cpad, cmax + cpad)
        a_meas.set_ylim(mmax + mpad, mmin - mpad)  # inverted: brighter at top

    for ax in axes.ravel():
        ax.set_facecolor(BG)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.6)
    for ax in (a_mem, a_ann):       # bottom row
        ax.set_xlabel(f"colour  {blue_name} − {red_name}  (Gaia BP/RP scale)", color=FG)
    for ax in (a_meas, a_mem):      # left column
        ax.set_ylabel(f"magnitude  {red_name}  (Gaia RP scale)", color=FG)

    # Shared colourbar for the temperature false-colour (measured + members).
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.04, pad=0.01)
    cbar.set_label(f"star colour  {blue_name} − {red_name}   "
                   f"(blue = hot  →  red = cool)", color=FG)
    cbar.ax.tick_params(colors="#8b949e")
    cbar.outline.set_edgecolor("#30363d")

    fig.suptitle(f"Colour–magnitude diagram — {dso_name}   ·   "
                 f"{n_gaia} Gaia calibration stars", color="#e6edf3", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
