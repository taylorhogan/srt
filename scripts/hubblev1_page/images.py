import sys, json, math, numpy as np
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # solve.py beside this file
import solve as sv
from PIL import Image, ImageDraw, ImageFont
S = Path(sys.argv[1])
gaia = json.load(open(S / "gaia.json"))
V1 = (10.36375, 41.16956)
den = np.load(S / "hubblev1_L_den.npy"); raw = np.load(S / "hubblev1_L_raw.npy", mmap_mode="r")
h, w = den.shape
sol = sv.solve(np.asarray(raw), *V1, gaia)        # solve on the raw stack: the denoiser moves faint star shapes
print("stack solved:", {k: v for k, v in sol.items() if k != "M"}, flush=True)
json.dump(sol, open(S / "stack_solution.json", "w"))
vx, vy = (float(a[0]) for a in sv.sky_to_pix(sol, *V1))
print("V1 at", vx, vy, "of", w, h)
corners = {k: tuple(float(c[0]) for c in sv.pix_to_sky(sol, px, py)) for k, (px, py) in
           {"TL": (0, 0), "TR": (w, 0), "BR": (w, h), "BL": (0, h), "C": (w / 2, h / 2)}.items()}
print("corners", corners)
json.dump({"v1_xy": [vx, vy], "shape": [h, w], "corners": corners}, open(S / "geometry.json", "w"), indent=1)

def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except Exception:
        import matplotlib; return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), sz)

def stretch(a, soft_sig=6.0, white_pct=99.97, gamma=1.0):
    sub = a[::9, ::9]; med = np.median(sub); mad = 1.4826 * np.median(np.abs(sub - med))
    black = med - 0.5 * mad; white = np.percentile(sub, white_pct); soft = soft_sig * mad
    y = np.arcsinh(np.clip((a - black), 0, None) / soft) / math.asinh((white - black) / soft)
    y = np.clip(y, 0, 1) ** gamma
    return (y * 255).astype(np.uint8)

# ---- 1. the whole field, V1 circled, scaled to 2200 px wide
img8 = stretch(den)
pil = Image.fromarray(img8, "L").convert("RGB")
scale = 2200 / w
disp = pil.resize((2200, int(h * scale)), Image.LANCZOS)
d = ImageDraw.Draw(disp)
cx, cy = vx * scale, vy * scale
R = 36
d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(255, 90, 60), width=3)
d.line([cx + R + 4, cy, cx + R + 60, cy], fill=(255, 90, 60), width=3)
d.text((cx + R + 66, cy - 16), "V1", fill=(255, 90, 60), font=font(30))
# north / east arrows from the solution: direction of +eta and +xi in display pixels
M = np.array(sol["M"]); Mi = np.linalg.inv(M[:2, :2])
def arrow(dx_arcsec, dy_arcsec, label, col):
    v = Mi @ np.array([dx_arcsec * sol["flip"], dy_arcsec]); v = v / np.hypot(*v) * 110
    x0, y0 = 150, disp.height - 150
    d.line([x0, y0, x0 + v[0], y0 + v[1]], fill=col, width=4)
    d.text((x0 + v[0] * 1.25 - 10, y0 + v[1] * 1.25 - 14), label, fill=col, font=font(28))
arrow(0, 1, "N", (255, 255, 255)); arrow(1, 0, "E", (255, 255, 255))
# 5 arcminute bar
bar = 300 / sol["scale_arcsec_px"] * scale
d.line([disp.width - 80 - bar, disp.height - 70, disp.width - 80, disp.height - 70], fill=(255, 255, 255), width=4)
d.text((disp.width - 80 - bar, disp.height - 110), "5 arcminutes", fill=(255, 255, 255), font=font(26))
d.text((30, 24), "M31 V1 field, Iris CDK17, 208 x 300 s luminance, stacked and denoised", fill=(255, 255, 255), font=font(30))
disp.save(S / "field_v1.jpg", quality=86)
print("field written", disp.size, flush=True)

# ---- 2. the 1:1 crop around V1 with comparison stars labelled
cw, ch = 1500, 1000
x0, y0 = int(vx - cw / 2), int(vy - ch / 2)
crop = Image.fromarray(stretch(den[y0:y0 + ch, x0:x0 + cw], soft_sig=5.0, white_pct=99.9), "L").convert("RGB")
d = ImageDraw.Draw(crop)
R = 22
d.ellipse([vx - x0 - R, vy - y0 - R, vx - x0 + R, vy - y0 + R], outline=(255, 90, 60), width=3)
d.text((vx - x0 + R + 6, vy - y0 - 14), "V1", fill=(255, 90, 60), font=font(26))
comps = json.load(open(S / "comps.json"))["targets"][1:] if (S / "comps.json").exists() else []
labels = []
for c in comps:
    px, py = (float(a[0]) for a in sv.sky_to_pix(sol, c["ra"], c["dec"]))
    if 20 < px - x0 < cw - 20 and 20 < py - y0 < ch - 20:
        d.ellipse([px - x0 - 16, py - y0 - 16, px - x0 + 16, py - y0 + 16], outline=(90, 200, 255), width=2)
        d.text((px - x0 + 19, py - y0 - 12), "%s  G %.1f" % (c["name"], c["g"]), fill=(90, 200, 255), font=font(20))
        labels.append(c["name"])
bar = 60 / sol["scale_arcsec_px"]
d.line([cw - 40 - bar, ch - 32, cw - 40, ch - 32], fill=(255, 255, 255), width=3)
d.text((cw - 40 - bar, ch - 62), "1 arcminute", fill=(255, 255, 255), font=font(22))
d.text((16, 12), "native pixels, 0.26\" each", fill=(255, 255, 255), font=font(20))
crop.save(S / "crop_v1.jpg", quality=90)
print("crop written; comps labelled:", labels, flush=True)

# ---- 3. the footprint on the DSS2 image of M31 (SkyView, TAN, 8.1"/px, north up east left)
dss = Image.open(S / "dl/m31_c.jpg").convert("RGB")
W, H = dss.size; ra0, dec0, sc = 10.6847, 41.2687, 3.6 * 3600 / W
def to_dss(ra, dec):
    xi, eta = sv.gnomonic(ra, dec, ra0, dec0)
    return float(W / 2 - xi / sc), float(H / 2 - eta / sc)
d = ImageDraw.Draw(dss, "RGBA")
poly = [to_dss(*corners[k]) for k in ("TL", "TR", "BR", "BL")]
d.polygon(poly, outline=(255, 170, 40, 255), fill=(255, 170, 40, 28))
d.line(poly + [poly[0]], fill=(255, 170, 40, 255), width=3)
px, py = to_dss(*V1)
d.ellipse([px - 9, py - 9, px + 9, py + 9], outline=(255, 70, 50, 255), width=3)
d.text((px + 14, py - 12), "V1", fill=(255, 70, 50, 255), font=font(26))
d.text((poly[1][0] + 12, poly[1][1] - 30), "the Iris frame, 42' x 28'", fill=(255, 170, 40, 255), font=font(24))
for name, (ra, dec), dx in (("M32", (10.6743, 40.8652), 14), ("M110", (10.0921, 41.6853), 14), ("nucleus", (10.6847, 41.2687), -110)):
    qx, qy = to_dss(ra, dec)
    d.text((qx + dx, qy - 12), name, fill=(235, 235, 235, 255), font=font(22))
d.text((20, 16), "M31, the Andromeda Galaxy: 3.6 x 2.5 degrees (DSS2 red, north up, east left)", fill=(255, 255, 255, 255), font=font(26))
bar = 1800 / sc
d.line([W - 60 - bar, H - 50, W - 60, H - 50], fill=(255, 255, 255, 255), width=3)
lab = "30 arcminutes, the width of the full Moon"; tw = font(22).getlength(lab)
d.text((W - 60 - tw, H - 82), lab, fill=(255, 255, 255, 255), font=font(22))
dss.save(S / "m31_footprint.jpg", quality=88)
print("footprint written; V1 on DSS at", round(px), round(py), flush=True)
