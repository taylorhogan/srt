"""Nightly differential photometry of V1 and a few Gaia comparison stars on the raw subs (3 frames a night)."""
import sys, json, glob, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # solve.py beside this file
import solve as sv
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

def main():
    S = Path(sys.argv[1])
    gaia = json.load(open(S / "gaia.json"))
    V1 = (10.36375, 41.16956)
    xi, eta = sv.gnomonic([g["ra"] for g in gaia], [g["dec"] for g in gaia], *V1)
    r = np.hypot(xi, eta)
    # comparison candidates: 15 < G < 17, not flagged variable, within 5', no neighbour brighter than G+2 within 12"
    cand = []
    for i, g in enumerate(gaia):
        if not (15.0 < g["g"] < 17.0) or g["var"] == "VARIABLE" or r[i] > 300:
            continue
        near = [j for j in range(len(gaia)) if j != i and np.hypot(xi[j] - xi[i], eta[j] - eta[i]) < 12 and gaia[j]["g"] < g["g"] + 2]
        if near:
            continue
        cand.append(dict(g, sep_arcsec=float(r[i])))
    cand.sort(key=lambda g: g["g"])
    comps = cand[:8]
    print("comps:", [(c["id"], round(c["g"], 2), round(c["sep_arcsec"])) for c in comps], flush=True)
    targets = [{"name": "V1", "ra": V1[0], "dec": V1[1]}] + [{"name": "C%d" % (k + 1), "ra": c["ra"], "dec": c["dec"], "g": c["g"], "id": c["id"]} for k, c in enumerate(comps)]
    out = {"targets": targets, "nights": []}
    for nd in sorted(glob.glob("/home/taylor/Desktop/Targets/hubblev1/cdk17/*/LIGHT")):
        files = sorted(glob.glob(nd + "/*.fits"))
        if not files:
            continue
        pick = [files[len(files) // 4], files[len(files) // 2], files[(3 * len(files)) // 4]]
        per_frame = []
        for f in pick:
            try:
                arr = fits.getdata(f).astype(np.float32)
                sol = sv.solve(arr, *V1, gaia)
                x, y = sv.sky_to_pix(sol, [t["ra"] for t in targets], [t["dec"] for t in targets])
                pos = np.vstack([x, y]).T
                ap = CircularAperture(pos, r=9.0); an = CircularAnnulus(pos, r_in=16, r_out=26)
                ph = aperture_photometry(arr, ap)
                # local sky: sigma-clipped median in the annulus
                sky = []
                for m in an.to_mask(method="center"):
                    d = m.multiply(arr); d = d[m.data > 0]
                    med = np.median(d); mad = 1.4826 * np.median(np.abs(d - med)); d = d[np.abs(d - med) < 3 * mad]
                    sky.append(np.median(d))
                flux = np.asarray(ph["aperture_sum"]) - np.asarray(sky) * ap.area
                per_frame.append(-2.5 * np.log10(np.maximum(flux, 1.0)))
            except Exception as e:  # noqa: BLE001
                print("  skip", Path(f).name, type(e).__name__, str(e)[:80], flush=True)
        if not per_frame:
            continue
        inst = np.median(np.vstack(per_frame), axis=0)             # per target, this night
        # ensemble zero point from the comps (their Gaia G minus instrumental), each comp judged against the OTHERS
        gmag = np.array([t.get("g", np.nan) for t in targets])
        zp_each = gmag - inst
        zp = np.nanmedian(zp_each[1:])
        cal = inst + zp
        loo = []
        for k in range(1, len(targets)):
            others = np.delete(zp_each[1:], k - 1)
            loo.append(float(inst[k] + np.nanmedian(others)))
        night = Path(nd).parent.name
        out["nights"].append({"night": night, "frames": [Path(f).name for f in pick], "mag": [float(v) for v in cal],
                              "mag_loo": [float(cal[0])] + loo, "zp": float(zp), "zp_scatter": float(np.nanstd(zp_each[1:]))})
        print(night, "V1 %.3f" % cal[0], "comps", " ".join("%.3f" % v for v in loo), "zp sc %.3f" % np.nanstd(zp_each[1:]), flush=True)
    json.dump(out, open(S / "comps.json", "w"), indent=1)
    print("done", flush=True)

if __name__ == "__main__":
    main()
