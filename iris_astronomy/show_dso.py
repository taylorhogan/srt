from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
import numpy as np

# ── Your setup ────────────────────────────────────────────────────────────────
FOCAL_LENGTH_MM = 2563  # CDK17 native focal length
SENSOR_WIDTH_MM = 35.9  # QHY600M (Sony IMX455, full-frame width)
SENSOR_HEIGHT_MM = 23.9
SENSOR_WIDTH_PIX = 9576
SENSOR_HEIGHT_PIX = 6388


# ─────────────────────────────────────────────────────────────────────────────

def field_of_view(focal_length_mm, sensor_w_mm, sensor_h_mm):
    """Returns (fov_width_deg, fov_height_deg)"""
    fov_w = 2 * np.degrees(np.arctan(sensor_w_mm / (2 * focal_length_mm)))
    fov_h = 2 * np.degrees(np.arctan(sensor_h_mm / (2 * focal_length_mm)))
    return fov_w, fov_h


def get_dso_image(target_name: str, survey="DSS2 Red", show=True):
    fov_w, fov_h = field_of_view(FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM)

    print(f"FOV: {fov_w * 60:.2f}' x {fov_h * 60:.2f}'")
    print(f"Image scale: {(SENSOR_WIDTH_MM / SENSOR_WIDTH_PIX) / FOCAL_LENGTH_MM * 206265:.3f} \"/px")

    # Resolve name → coordinates
    result = Simbad.query_object(target_name)
    if result is None:
        raise ValueError(f"Could not resolve '{target_name}' via SIMBAD")
    coord = SkyCoord(result["ra"][0], result["dec"][0], unit=(u.deg, u.deg))
    print(f"Resolved {target_name}: RA={coord.ra.to_string(unit=u.hour)}, Dec={coord.dec:.4f}")

    # Fetch image from SkyView
    images = SkyView.get_images(
        position=coord,
        survey=[survey],
        width=fov_w * u.deg,
        height=fov_h * u.deg,
        pixels=f"{min(SENSOR_WIDTH_PIX // 4, 1500)},{min(SENSOR_HEIGHT_PIX // 4, 1000)}",
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
