"""Check a printable black-cells AprilTag (.scad) before it is printed (2026-10-29).

    python scripts/scope_tag_scad_check.py [openscad/scope_tag_black_id0.scad]

Reads the bit matrix, corner patches and sizes out of the .scad, draws the part
as it will be seen (black on a white background with the quiet zone), and
  * checks the black is ONE piece (4-connected pixels, so a corner touch alone
    does not count -- it would not hold together when printed),
  * decodes it with the detector the observatory uses, sharp, blurred, small
    and tilted, and says the id each one returns.
The connecting patches have to be invisible to the decode; this is the proof.
"""
import ast
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PX_PER_MM = 10


def _value(src, name):
    m = re.search(r"^%s\s*=\s*(.+?);" % re.escape(name), src, re.M | re.S)
    if not m:
        raise SystemExit("no %s in the .scad" % name)
    text = re.sub(r"//[^\n]*", "", m.group(1))
    return ast.literal_eval(text.replace("\n", " ").replace(",]", "]").replace(", ]", "]"))


def render(scad_path):
    src = open(scad_path, encoding="utf-8").read()
    bits = _value(src, "bits")
    patches = _value(src, "corner_patches")
    tag_size, modules, patch = _value(src, "tag_size"), _value(src, "modules"), _value(src, "patch")
    mod = tag_size / modules
    n = len(bits)
    q = (modules - n) // 2                    # quiet modules on each side
    size = int(round(tag_size * PX_PER_MM))
    img = np.full((size, size), 255, np.uint8)
    o = q * mod
    for r in range(n):
        for c in range(n):
            if bits[r][c]:
                x0, y0 = o + c * mod, o + r * mod
                img[int(round(y0 * PX_PER_MM)):int(round((y0 + mod) * PX_PER_MM)),
                    int(round(x0 * PX_PER_MM)):int(round((x0 + mod) * PX_PER_MM))] = 0
    for r, c in patches:
        x, y = o + c * mod, o + r * mod
        img[int(round((y - patch / 2) * PX_PER_MM)):int(round((y + patch / 2) * PX_PER_MM)),
            int(round((x - patch / 2) * PX_PER_MM)):int(round((x + patch / 2) * PX_PER_MM))] = 0
    return img


def decode(img):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    pad = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    _c, ids, _r = det.detectMarkers(pad)
    return None if ids is None else [int(i) for i in ids.ravel()]


def main(argv):
    path = argv[1] if len(argv) > 1 else os.path.join(ROOT, "openscad", "scope_tag_black_id0.scad")
    img = render(path)
    n_parts, _ = cv2.connectedComponents((img == 0).astype(np.uint8), connectivity=4)
    print("black pieces (4-connected): %d" % (n_parts - 1))
    h = img.shape[0]
    tilt = cv2.warpPerspective(img, cv2.getPerspectiveTransform(
        np.float32([[0, 0], [h, 0], [h, h], [0, h]]),
        np.float32([[h * .12, h * .05], [h * .9, 0], [h, h], [0, h * .92]])), (h, h), borderValue=255)
    views = {
        "sharp": img,
        "blurred": cv2.GaussianBlur(img, (0, 0), 6),
        "small (60 px across)": cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA),
        "tilted": tilt,
    }
    ok = n_parts - 1 == 1
    for name, v in views.items():
        ids = decode(v)
        print("%-22s decodes as %s" % (name, ids))
        ok &= ids == [0]
    print("OK" if ok else "NOT OK -- do not print")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
