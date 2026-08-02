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
