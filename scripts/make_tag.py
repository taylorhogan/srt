"""make_tag.py — a printable AprilTag at an exact physical size.

    python scripts/make_tag.py 1 --mm 76 -o local/tag1_76mm.png

The size you ask for is the size of the WHOLE IMAGE including its white quiet
zone, because that is the number you can measure on the printed sheet and the
number the OpenSCAD plates are dimensioned by (openscad/roof_marker_plate.scad
and marker_plate.scad both call their parameter `tag_size` and both mean this).

The quiet zone is not decoration. An AprilTag's outer ring is BLACK, so with no
light margin the tag's border merges into whatever it is stuck to and the
detector never finds a quad to begin with. One module of white on every side is
the documented minimum and is what this writes.

Printing:
  * Print at 100% / "actual size" — every "fit to page" setting silently
    rescales, and then the tag is no longer the size the plate was cut for.
  * Matte, not glossy. A gloss finish returns a specular highlight straight
    back at the IR illuminator that sits beside the lens, and a blown highlight
    across part of the pattern costs the decode.
  * Measure the printed square before sticking it down. Two minutes with a
    rule here saves discovering a 96% scale after the roof is shut.
  * DENSE BLACK MATTERS MORE THAN IT LOOKS. Some black inks are near
    transparent in the near infrared, and this camera runs IR most of the
    night, so a tag that reads perfectly in daylight can vanish after dark.
    The scope's tag is read on 84% of IR frames, so whatever printed that one
    is known to work here; use the same printer and paper if you can.

Verification is built in: after rendering, the file is read back and run
through the same detector the observatory uses, and the id it returns is
printed. If that line does not say the id you asked for, do not mount it.
"""
import argparse
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2
import numpy as np

DICTS = {
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}
DEFAULT_DICT = "APRILTAG_36h11"     # what is mounted on the scope today


def build(tag_id, mm, dpi, dict_name=DEFAULT_DICT, quiet_modules=1):
    d = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
    tag_modules = d.markerSize + 2          # data + the black border ring
    total_modules = tag_modules + 2 * quiet_modules

    px = int(round(mm / 25.4 * dpi))
    # Round the module size to a whole number of pixels and derive the image
    # from THAT, so every module is identical. Letting cv2 scale a small marker
    # up to an arbitrary pixel count leaves some modules a pixel wider than
    # others, which shows up as a bias in the decoded corner positions.
    ppm = max(1, round(px / total_modules))
    marker_px = ppm * tag_modules
    img = cv2.aruco.generateImageMarker(d, tag_id, marker_px, borderBits=1)
    pad = ppm * quiet_modules
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=255)
    actual_mm = img.shape[0] / dpi * 25.4
    return img, actual_mm, ppm / dpi * 25.4


def write_pdf(png, out, mm, tag_id, dict_name):
    """A page with the tag at an exact physical size, and a ruler to prove it.

    A PNG carries no physical size unless something writes one, so every
    viewer and print dialog is free to guess -- and they guess differently,
    which is why "print at 100%" so often does not. A PDF has no such freedom:
    the page is a physical object and the image is placed on it in millimetres.

    The ruler is not decoration. It is the only way to find a scaling error
    AFTER printing and BEFORE the tag is glued to a plate and lifted onto a
    roof. If the 100 mm line does not measure 100 mm, nothing else on the page
    is the size it claims either.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PAGE_W, PAGE_H = 215.9, 279.4          # US Letter, mm
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    # cv2, NOT matplotlib.image.imread: the latter normalises an 8-bit PNG to
    # floats in 0..1, and combined with the explicit vmin/vmax below every
    # pixel then mapped to the bottom of the scale and the tag printed as a
    # solid black square. Reading as uint8 keeps the data in the same units
    # the limits are written in.
    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)

    # Axes placed in FIGURE FRACTIONS that work out to exact millimetres.
    x0 = (PAGE_W - mm) / 2.0
    y0 = PAGE_H - 45.0 - mm
    ax = fig.add_axes([x0 / PAGE_W, y0 / PAGE_H, mm / PAGE_W, mm / PAGE_H])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # WHERE TO CUT. The quiet zone is white on white paper, so the edge of the
    # sticker is invisible and the 76.20 mm cannot be found by eye. Two marks,
    # for two different jobs:
    #
    #   a hairline ON the boundary, so the size is visible and measurable;
    #   corner crop marks OUTSIDE it, which survive the cut and are what you
    #   actually align scissors to once the hairline is being removed.
    #
    # The hairline sits a full module (7.62 mm at this size) clear of the tag's
    # black border, so it cannot merge with it or be read as part of the
    # pattern. Cut ON the hairline: the pocket in the plate is sized for
    # 76.20 mm, so trimming wide will not fit and trimming narrow eats the
    # quiet zone the detector needs.
    axc = fig.add_axes([0, 0, 1, 1], facecolor="none")
    axc.set_xlim(0, PAGE_W); axc.set_ylim(0, PAGE_H); axc.axis("off")
    axc.add_patch(plt.Rectangle((x0, y0), mm, mm, fill=False,
                                edgecolor="0.55", lw=0.4))
    GAP, LEN = 2.0, 9.0
    for cx_, cy_, sx, sy in ((x0, y0, -1, -1), (x0 + mm, y0, 1, -1),
                             (x0, y0 + mm, -1, 1), (x0 + mm, y0 + mm, 1, 1)):
        axc.plot([cx_ + sx * GAP, cx_ + sx * (GAP + LEN)], [cy_, cy_],
                 color="black", lw=0.6)
        axc.plot([cx_, cx_], [cy_ + sy * GAP, cy_ + sy * (GAP + LEN)],
                 color="black", lw=0.6)
    axc.text(x0 + mm / 2, y0 - 13.0,
             "cut on the thin line / crop marks -- %.2f mm square" % mm,
             ha="center", va="top", fontsize=7.5, color="0.35")

    ry = y0 - 30.0
    rl = 100.0
    rx = (PAGE_W - rl) / 2.0
    axr = fig.add_axes([0, 0, 1, 1], facecolor="none")
    axr.set_xlim(0, PAGE_W); axr.set_ylim(0, PAGE_H)
    axr.axis("off")
    axr.plot([rx, rx + rl], [ry, ry], color="black", lw=1.0)
    for t in range(0, 101, 10):
        h = 3.5 if t % 50 else 6.0
        axr.plot([rx + t, rx + t], [ry, ry + h], color="black", lw=1.0)
    axr.text(PAGE_W / 2, ry - 6, "this line is exactly 100 mm -- measure it before you cut",
             ha="center", va="top", fontsize=8)
    axr.text(PAGE_W / 2, y0 + mm + 21,
             "AprilTag %s  id %d" % (dict_name, tag_id),
             ha="center", va="bottom", fontsize=13)
    axr.text(PAGE_W / 2, y0 + mm + 14,
             "%.2f mm square including its white quiet zone -- do not trim the white"
             % mm, ha="center", va="bottom", fontsize=8)
    axr.text(PAGE_W / 2, ry - 16,
             "Print at 100% / Actual Size. Matte, not glossy. Dense black.",
             ha="center", va="top", fontsize=8)
    fig.savefig(out)
    # Render the SAME figure to a raster and put it through the detector. A
    # PDF whose geometry is right but whose image is unreadable still fails,
    # and that is precisely the failure this function shipped with once.
    png_proof = os.path.splitext(out)[0] + "_proof.png"
    fig.savefig(png_proof, dpi=200)
    plt.close(fig)
    proof = cv2.imread(png_proof, cv2.IMREAD_GRAYSCALE)
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTS[dict_name]))
    _, ids, _ = det.detectMarkers(proof)
    got = None if ids is None else ids.ravel().tolist()
    ok = got == [tag_id]
    print("  page proof: detector reads %s  %s"
          % (got, "OK" if ok else "*** THE PAGE IS NOT READABLE, DO NOT PRINT ***"))
    if ok:
        os.remove(png_proof)
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", type=int, help="marker id (scope is 0, roof is 1)")
    ap.add_argument("--mm", type=float, default=76.0,
                    help="width of the whole image incl. quiet zone (default 76)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--dict", default=DEFAULT_DICT, choices=sorted(DICTS))
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--pdf", default=None,
                    help="also write a Letter page with the tag at exact size "
                         "and a 100 mm ruler to verify the print")
    args = ap.parse_args()

    img, actual_mm, module_mm = build(args.id, args.mm, args.dpi, args.dict)
    out = args.out or "tag%d_%dmm.png" % (args.id, round(actual_mm))
    cv2.imwrite(out, img)

    print("%s  dict %s  id %d" % (out, args.dict, args.id))
    print("  image      %d x %d px at %d dpi" % (img.shape[1], img.shape[0], args.dpi))
    print("  PRINT SIZE %.2f mm square (asked %.1f)" % (actual_mm, args.mm))
    print("  module     %.2f mm" % module_mm)

    # Read it back through the observatory's own detector. A tag that does not
    # survive this round trip must never reach the roof.
    # Stamp the physical size into the PNG too (pHYs). Not a substitute for
    # the PDF -- plenty of software ignores it -- but it stops the viewers
    # that DO read it from inventing 72 or 96 dpi.
    try:
        from PIL import Image
        im = Image.open(out)
        im.save(out, dpi=(args.dpi, args.dpi))
        print("  stamped %d dpi into the PNG" % args.dpi)
    except Exception as exc:              # noqa: BLE001
        print("  (could not stamp dpi: %s)" % type(exc).__name__)

    if args.pdf:
        write_pdf(out, args.pdf, actual_mm, args.id, args.dict)
        print("  wrote %s  (Letter page, tag at %.2f mm, with a 100 mm ruler)"
              % (args.pdf, actual_mm))

    check = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTS[args.dict]))
    _, ids, _ = det.detectMarkers(check)
    got = None if ids is None else ids.ravel().tolist()
    print("  detector reads back: %s  %s"
          % (got, "OK" if got == [args.id] else "*** MISMATCH, DO NOT MOUNT ***"))
    return 0 if got == [args.id] else 1


if __name__ == "__main__":
    sys.exit(main())
