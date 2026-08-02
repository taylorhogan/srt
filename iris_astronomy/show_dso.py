from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
import numpy as np

# ── Your setup ────────────────────────────────────────────────────────────────
# 2939 mm is what the FITS headers report (FOCALLEN), not the CDK17's 2563 mm
# native figure this used to carry. The stale number made every `show` preview
# frame 48.2' x 32.1' when the camera actually sees 41.5' x 27.7' — 16% wider
# than reality, which is misleading precisely when you are using the preview to
# judge whether a target fits.
FOCAL_LENGTH_MM = 2939  # measured: matches FOCALLEN in the light frames
SENSOR_WIDTH_MM = 35.9  # QHY600M (Sony IMX455, full-frame width)
SENSOR_HEIGHT_MM = 23.9
SENSOR_WIDTH_PIX = 9576
SENSOR_HEIGHT_PIX = 6388


# ─────────────────────────────────────────────────────────────────────────────

def pixel_scale_arcsec() -> float:
    """Arcsec per pixel for the imaging train.

    Prefers nina.arc_sec_per_pixel from config: that is the number the stacker,
    the FWHM measurements and the plate solver all work from, so the preview
    agrees with the rest of the system by construction instead of by someone
    remembering to update a constant here. Falls back to the optics when config
    is not importable (this module is runnable standalone).
    """
    try:
        from configs import config
        aspp = float(config.data()["nina"]["arc_sec_per_pixel"])
        if aspp > 0:
            return aspp
    except Exception:
        pass
    return (SENSOR_WIDTH_MM / SENSOR_WIDTH_PIX) / FOCAL_LENGTH_MM * 206265


def field_of_view(focal_length_mm=None, sensor_w_mm=None, sensor_h_mm=None):
    """Returns (fov_width_deg, fov_height_deg) for the real imaging train.

    Arguments are accepted for backwards compatibility but ignored unless config
    is unavailable — see pixel_scale_arcsec.
    """
    if focal_length_mm and sensor_w_mm and sensor_h_mm:
        try:
            from configs import config
            config.data()["nina"]["arc_sec_per_pixel"]
        except Exception:
            fov_w = 2 * np.degrees(np.arctan(sensor_w_mm / (2 * focal_length_mm)))
            fov_h = 2 * np.degrees(np.arctan(sensor_h_mm / (2 * focal_length_mm)))
            return fov_w, fov_h
    aspp = pixel_scale_arcsec()
    return (SENSOR_WIDTH_PIX * aspp / 3600.0, SENSOR_HEIGHT_PIX * aspp / 3600.0)


# Colour survey cutout services, tried in order by get_preview_image. Measured
# against NGC 3628 at the real 41.5' x 27.7' frame on 2026-08-02:
#   legacy  DESI Legacy Imaging Surveys (grz). Deepest, and the only one whose
#           pixel scale is adjustable, so any FOV fits in one request. Roughly
#           dec -20..+85. Was returning HTTP 503 the day this was written, which
#           is exactly why this is a cascade and not a swap.
#   sdss    SDSS DR18 (gri). Excellent where it reaches, but ~1/3 of sky: of 17
#           real Iris targets, 13 covered. It misses the galactic plane, so
#           sh2-92 — the current narrowband target — is NOT in it.
#   dss2    SkyView DSS2 Red. Photographic, shallow, monochrome, all-sky. The
#           fallback that always answers.
# Pan-STARRS1 is deliberately absent: its cutout service is fixed at 0.25"/px
# and capped at 6000 px = 25', so it cannot cover this frame without mosaicking
# skycells, and what it does return is mis-centred with visible seams.
#   dss2rgb DSS2 Blue+Red+IR composited. Photographic and shallow, but all-sky —
#           and all-sky is the whole point, because both deep colour surveys are
#           *extragalactic* and deliberately mask the Milky Way. NGC 7380 (b =
#           -0.9) and IC 405 (b = -2.0) are invisible to SDSS and to Legacy, so
#           without this tier every galactic-plane target — which is most of the
#           narrowband work — is permanently monochrome.
PREVIEW_SOURCES = ("legacy", "sdss", "dss2rgb", "dss2")
_PREVIEW_MAX_PX = 1500


def _fetch_jpeg(url, params, timeout=120, attempts=2):
    """GET a JPEG cutout and return it as an (H, W, 3) uint8 array, or None.

    Retries once: a single dropped request would otherwise be indistinguishable
    from "this survey doesn't cover the target" and quietly downgrade the
    preview to a worse survey. Observed once in six calls while testing.
    """
    import requests
    from PIL import Image
    import io, time
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": "iris-observatory/1.0"})
            if r.status_code == 503:
                return None            # service down — don't burn time retrying
            if r.status_code == 200 and len(r.content) >= 5000:
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(1.5)
    return None


def _legacy_preview(ra, dec, fov_w_deg, fov_h_deg):
    px = fov_w_deg * 3600 / _PREVIEW_MAX_PX          # arcsec/px
    img = _fetch_jpeg("https://www.legacysurvey.org/viewer/cutout.jpg", {
        "ra": ra, "dec": dec, "layer": "ls-dr10", "pixscale": px,
        "width": _PREVIEW_MAX_PX, "height": int(_PREVIEW_MAX_PX * fov_h_deg / fov_w_deg),
    })
    # Outside the footprint it answers with a flat tile rather than an error.
    if img is None or float(img.std()) < 1.0:
        return None
    return img, "DESI Legacy DR10 (grz)"


def _sdss_preview(ra, dec, fov_w_deg, fov_h_deg):
    img = _fetch_jpeg("https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg", {
        "ra": ra, "dec": dec, "scale": fov_w_deg * 3600 / _PREVIEW_MAX_PX,
        "width": _PREVIEW_MAX_PX, "height": int(_PREVIEW_MAX_PX * fov_h_deg / fov_w_deg),
    })
    if img is None:
        return None
    # Out of footprint SDSS returns HTTP 200 with a black tile carrying the words
    # "is outside the SDSS footprint". It is 91.3% exact-zero pixels; a real
    # field, even empty sky, is under 8%.
    if float((img.max(axis=2) == 0).mean()) > 0.5:
        return None
    return img, "SDSS DR18 (gri)"


def _dss2_rgb_preview(coord, fov_w_deg, fov_h_deg):
    """Composite the DSS2 Red and Blue plates into colour. All-sky, always answers.

    Red -> R, Blue -> B, green synthesised as their mean. The obvious mapping is
    IR/Red/Blue into R/G/B, but DSS2's Red plate is where H-alpha lands, so that
    puts every emission nebula in the green channel and IC 405 comes out teal.
    Dropping IR and synthesising green renders emission red, reflection blue, and
    stars roughly their real colours — and costs one fewer SkyView request.

    Each plate is percentile-normalised before stacking, not noise-normalised:
    the IR and Red plates differ ~6x in background noise, so scaling by noise
    would have made one channel dominate on brightness alone.
    """
    from astropy.visualization import AsinhStretch, PercentileInterval
    h = int(_PREVIEW_MAX_PX * fov_h_deg / fov_w_deg)
    try:
        images = SkyView.get_images(
            position=coord, survey=["DSS2 Red", "DSS2 Blue"],
            width=fov_w_deg * u.deg, height=fov_h_deg * u.deg,
            pixels=f"{_PREVIEW_MAX_PX},{h}", cache=False,
        )
    except Exception:
        return None
    if not images or len(images) < 2:
        return None

    planes = []
    for hdul in images:
        data = np.asarray(hdul[0].data, dtype=float)
        if not np.isfinite(data).any():
            return None
        lo, hi = PercentileInterval(99.5).get_limits(data)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            return None
        planes.append(np.nan_to_num(np.clip((data - lo) / (hi - lo), 0, 1)))

    red, blue = planes
    stretch = AsinhStretch(a=0.1)
    rgb = np.dstack([stretch(red), stretch(0.5 * (red + blue)), stretch(blue)])
    rgb = (rgb * 255).astype(np.uint8)
    if float(rgb.std()) < 1.0:
        return None
    return rgb, "DSS2 R/B composite"


def get_preview_image(target_name: str, sources=PREVIEW_SOURCES):
    """Best available preview of *target_name*: (array, survey_label, is_colour).

    Walks PREVIEW_SOURCES and takes the first that actually returns sky, so a
    colour survey being down or not covering the target degrades to the next one
    rather than to nothing.
    """
    fov_w, fov_h = field_of_view()
    result = Simbad.query_object(target_name)
    if result is None:
        raise ValueError(f"Could not resolve '{target_name}' via SIMBAD")
    coord = SkyCoord(result["ra"][0], result["dec"][0], unit=(u.deg, u.deg))
    ra, dec = float(coord.ra.deg), float(coord.dec.deg)

    for name in sources:
        if name == "legacy":
            got = _legacy_preview(ra, dec, fov_w, fov_h)
        elif name == "sdss":
            got = _sdss_preview(ra, dec, fov_w, fov_h)
        elif name == "dss2rgb":
            got = _dss2_rgb_preview(coord, fov_w, fov_h)
        elif name == "dss2":
            data, _hdr = get_dso_image(target_name, show=False)
            return data, "DSS2 Red", False
        else:
            continue
        if got is not None:
            return got[0], got[1], True
    raise RuntimeError(f"No survey returned an image for {target_name}")


def get_dso_image(target_name: str, survey="DSS2 Red", show=True):
    fov_w, fov_h = field_of_view(FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM)

    print(f"FOV: {fov_w * 60:.2f}' x {fov_h * 60:.2f}'")
    print(f"Image scale: {pixel_scale_arcsec():.3f} \"/px")

    # Resolve name → coordinates
    result = Simbad.query_object(target_name)
    if result is None:
        raise ValueError(f"Could not resolve '{target_name}' via SIMBAD")
    coord = SkyCoord(result["ra"][0], result["dec"][0], unit=(u.deg, u.deg))
    print(f"Resolved {target_name}: RA={coord.ra.to_string(unit=u.hour)}, Dec={coord.dec:.4f}")

    # Fetch image from SkyView.
    #
    # cache=False deliberately. SkyView downloads through astropy's shared
    # ~/.astropy/cache, and on 2026-08-02 one entry there had unreadable ACLs —
    # the directory itself denied access — so every `show` died with
    # PermissionError on a file it only wanted to write. A survey preview is
    # fetched once and looked at once; there is nothing to gain from caching it,
    # and plenty to lose by depending on a cache this process cannot repair.
    images = SkyView.get_images(
        position=coord,
        survey=[survey],
        width=fov_w * u.deg,
        height=fov_h * u.deg,
        pixels=f"{min(SENSOR_WIDTH_PIX // 4, 1500)},{min(SENSOR_HEIGHT_PIX // 4, 1000)}",
        cache=False,
    )

    if not images:
        raise RuntimeError(f"No image returned from SkyView for {target_name}")

    data = images[0][0].data
    header = images[0][0].header

    if show:
        norm = ImageNormalize(data, interval=ZScaleInterval())
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
        ax.imshow(data, origin="lower", cmap="gray", norm=norm, aspect="equal")
        ax.set_title(
            f"{target_name}  |  {survey}\n"
            f"FOV {fov_w * 60:.1f}' × {fov_h * 60:.1f}'  "
            f"({fov_w:.3f}° × {fov_h:.3f}°)",
            color="white", pad=10
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return data, header


if __name__ == "__main__":
    data, hdr = get_dso_image("NGC 891")
    # data, hdr = get_dso_image("M94", survey="DSS2 Blue")
    # data, hdr = get_dso_image("NGC 6888", survey="H-Alpha Composite Survey")
