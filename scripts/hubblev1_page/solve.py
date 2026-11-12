"""Match a frame's stars to Gaia and return pixel <-> sky. No ASTAP on this machine."""
import json, math, numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import astroalign as aa
from skimage.transform import SimilarityTransform

def gnomonic(ra, dec, ra0, dec0):
    ra, dec, ra0, dec0 = map(np.radians, (np.asarray(ra, float), np.asarray(dec, float), ra0, dec0))
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.degrees(xi) * 3600, np.degrees(eta) * 3600          # arcsec, east and north positive

def inv_gnomonic(xi, eta, ra0, dec0):
    xi = np.radians(np.asarray(xi, float) / 3600); eta = np.radians(np.asarray(eta, float) / 3600)
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    rho = np.hypot(xi, eta); c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * math.sin(dec0) + np.where(rho > 0, eta * np.sin(c) * math.cos(dec0) / rho, 0))
        ra = ra0 + np.arctan2(xi * np.sin(c), rho * math.cos(dec0) * np.cos(c) - eta * math.sin(dec0) * np.sin(c))
    return np.degrees(ra) % 360, np.degrees(dec)

def detect(arr, bin_=4, nmax=400, fwhm_px=3.0):
    h, w = arr.shape; h -= h % bin_; w -= w % bin_
    b = arr[:h, :w].reshape(h // bin_, bin_, w // bin_, bin_).mean(axis=(1, 3)).astype(np.float32)
    mean, med, std = sigma_clipped_stats(b[::7, ::7], sigma=3.0)
    f = DAOStarFinder(fwhm=fwhm_px, threshold=8 * std, exclude_border=True, peakmax=None)
    t = f(b - med)
    if t is None or len(t) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    t.sort("flux"); t = t[::-1][:nmax]
    xy = np.vstack([t["xcentroid"], t["ycentroid"]]).T * bin_ + (bin_ - 1) / 2.0
    return xy, np.asarray(t["flux"])

def solve(arr, ra0, dec0, gaia, g_limit=15.0, nmax=150, bin_=4):
    """-> dict(T: 2x3 affine pixel->(xi,eta) arcsec, flip, n_match, rms_arcsec, ra0, dec0)"""
    xy, flux = detect(arr, bin_=bin_, nmax=nmax)
    cat = [g for g in gaia if g["g"] < g_limit]
    xi, eta = gnomonic([g["ra"] for g in cat], [g["dec"] for g in cat], ra0, dec0)
    best = None
    for flip in (1, -1):
        dst = np.vstack([flip * xi, eta]).T
        # keep the catalogue inside a generous box so the triangle matching is not swamped
        keep = (np.abs(dst[:, 0]) < 1500) & (np.abs(dst[:, 1]) < 1100)
        try:
            T, (s, d) = aa.find_transform(xy, dst[keep], max_control_points=100)
        except Exception as e:  # noqa: BLE001
            continue
        if best is None or len(s) > best[2]:
            best = (T, flip, len(s), s, d)
    if best is None:
        raise RuntimeError("no match")
    T, flip, n, s, d = best
    # refine with a least-squares similarity on all matched pairs
    tr = SimilarityTransform(); tr.estimate(s, d)
    res = tr(s) - d
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    M = tr.params
    return {"M": M.tolist(), "flip": flip, "n_match": int(n), "rms_arcsec": rms, "ra0": ra0, "dec0": dec0,
            "scale_arcsec_px": float(tr.scale), "rotation_deg": float(np.degrees(tr.rotation))}

def pix_to_sky(sol, x, y):
    M = np.array(sol["M"]); x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float)); p = M @ np.vstack([x, y, np.ones_like(x)])
    xi, eta = p[0] * sol["flip"], p[1]
    return inv_gnomonic(xi, eta, sol["ra0"], sol["dec0"])

def sky_to_pix(sol, ra, dec):
    xi, eta = gnomonic(ra, dec, sol["ra0"], sol["dec0"])
    M = np.array(sol["M"]); Mi = np.linalg.inv(M)
    p = Mi @ np.vstack([np.atleast_1d(xi * sol["flip"]), np.atleast_1d(eta), np.ones_like(np.atleast_1d(xi))])
    return p[0], p[1]

if __name__ == "__main__":
    import sys, time
    from astropy.io import fits
    S = sys.argv[1]
    gaia = json.load(open(f"{S}/gaia.json"))
    f = "/home/taylor/Desktop/Targets/hubblev1/cdk17/2026-11-10/LIGHT/2026-11-10_18-46-39_L_446_300.00s_0072.fits"
    arr = fits.getdata(f).astype(np.float32)
    t0 = time.time()
    xy, fl = detect(arr); print("detect %.1f s, %d stars" % (time.time() - t0, len(xy)), flush=True)
    t0 = time.time()
    sol = solve(arr, 10.36375, 41.16956, gaia)
    print("solved in %.1f s:" % (time.time() - t0), {k: v for k, v in sol.items() if k != "M"})
    x, y = sky_to_pix(sol, 10.36375, 41.16956)
    print("V1 at pixel", float(x), float(y), "frame", arr.shape)
    h, w = arr.shape
    for name, (px, py) in {"TL": (0, 0), "TR": (w, 0), "BL": (0, h), "BR": (w, h), "C": (w / 2, h / 2)}.items():
        ra, dec = pix_to_sky(sol, px, py); print(" ", name, round(float(ra), 4), round(float(dec), 4))
    # check: a bright Gaia star's predicted pixel vs the nearest detected star
    xy, _ = detect(arr)
    gx, gy = sky_to_pix(sol, [g["ra"] for g in gaia if g["g"] < 13], [g["dec"] for g in gaia if g["g"] < 13])
    d = [float(np.min(np.hypot(xy[:, 0] - a, xy[:, 1] - b))) for a, b in zip(gx, gy) if 0 < a < w and 0 < b < h]
    print("bright Gaia stars inside: %d, median offset to nearest detection %.2f px" % (len(d), np.median(d)))
