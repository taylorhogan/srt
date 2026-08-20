#!/usr/bin/env python3
"""Re-score the LRGB ladder inside the region every stack actually covers.

    python scripts/n2n_ladder_rescore.py            # all arms present
    python scripts/n2n_ladder_rescore.py --arm pooled-scenes

The ladder's own numbers are measured over the whole frame, and on some groups
that frame contains pixels no light frame covered.

`stacker.stack` normally crops to its high-coverage region, but it **skips that
crop when `shared_reference` is passed** — correctly, because cropping each group
to its own coverage would put the groups back on different grids, which is the
bug step 18 fixed. What it does instead is fill residual NaN with the global sky
median. So every stack keeps a full-size frame in which the uncovered part is a
flat constant, and the two halves of a pair have *different* uncovered
footprints, because they hold different dithered frames.

Measured on ngc5907 G: 0.28% of stack A and 0.47% of stack B are fill, all of it
in one band at y < 1000, and `sep` finds ~460 detections in it — 48% of that
stack's 952 sources. The consequences, both of which produced confident wrong
numbers:

- **Source survival read 57%** for a model that had destroyed nothing. Nearly
  half the "sources" it was charged with losing were edge artefacts.
- **Repeatability between the two independent test stacks read 51%**, against
  97-99% for L, R and B. Masked to the covered region it is **99%** — the G
  stacks agree as well as any other filter's.

This script recomputes the same three metrics with those pixels excluded, from
the arrays `evaluate` already saved, so nothing has to be retrained.

The mask is the union of the fill in the raw stack, the second (ceiling) stack,
and the denoised output, dilated by 8 px because `sep`'s background mesh spreads
the discontinuity at a fill boundary beyond the fill itself.

This is a measurement fix, not a pipeline fix. The models were still *trained* on
patches that can contain fill, and `N2NDataset` samples patch origins without any
knowledge of coverage. That is the real defect and it is not addressed here.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

LADDER = Path(_root) / "local" / "n2n_ladder"
ARMS = ("per-filter", "pooled-filters", "pooled-scenes")


def log(m: str = "") -> None:
    print(m, flush=True)


def fill_mask(a: np.ndarray) -> np.ndarray:
    """Pixels equal to the frame's exact median — stacker's NaN fill value.

    An exact float equality is the right test here rather than a tolerance: the
    fill is written as one constant, so genuine sky pixels landing on it are a
    negligible handful (22 of 9.6M on ngc5907 G stack A, against 290,422 filled).
    """
    return a == np.median(a)


def masked_metrics(raw: np.ndarray, other: np.ndarray, den: np.ndarray,
                   nsigma: float = 5.0, aperture: float = 3.0,
                   dilate: int = 8) -> dict:
    import sep
    from scipy.ndimage import binary_dilation
    from scipy.spatial import cKDTree

    sep.set_extract_pixstack(5_000_000)
    raw = np.ascontiguousarray(raw.astype(np.float32))
    other = np.ascontiguousarray(other.astype(np.float32))
    den = np.ascontiguousarray(den.astype(np.float32))

    bad = fill_mask(raw) | fill_mask(other) | fill_mask(den)
    frac_bad = float(bad.mean())
    if dilate:
        bad = binary_dilation(bad, iterations=dilate)
    good = ~bad

    bg_r, bg_o, bg_d = sep.Background(raw), sep.Background(other), sep.Background(den)
    r_s, o_s, d_s = raw - bg_r.back(), other - bg_o.back(), den - bg_d.back()
    rms = float(bg_r.globalrms)

    # asinh-normalise before correlating, exactly as collapse_check does. This
    # is not cosmetic. These frames run ~4000:1 — stars reach 66,000 ADU against
    # a 3.6 ADU sky — so a raw Pearson r is set almost entirely by whether the
    # handful of bright stars line up, and any two stacks of the same field
    # agree on those. Measured: without this step every arm and filter scored
    # corr 0.9996-0.9998 against a "ceiling" of 0.9987-0.9993, i.e. 100% of
    # ceiling everywhere, which says nothing about denoising at all. asinh is
    # linear where the sky noise lives, which is the part being judged.
    from nn import denoiser

    g = good.ravel()
    rv = denoiser.normalise(r_s)[0].ravel()[g]
    ov = denoiser.normalise(o_s)[0].ravel()[g]
    dv = denoiser.normalise(d_s)[0].ravel()[g]
    r12 = float(np.corrcoef(rv, ov)[0, 1])
    ceiling = float(np.sqrt(max(r12, 0.0)))
    corr = float(np.corrcoef(rv, dv)[0, 1])

    # std_ratio needs ONE scale for both frames, or dividing each by its own sky
    # sigma forces the ratio to ~1 whatever the network did.
    r_n, scale = denoiser.normalise(r_s)
    d_shared = np.arcsinh(d_s / (scale if scale else 1.0))
    std_ratio = float(d_shared.ravel()[g].std() / max(r_n.ravel()[g].std(), 1e-12))

    thresh = nsigma * rms
    src_r = sep.extract(r_s, thresh, minarea=5)
    src_d = sep.extract(d_s, thresh, minarea=5)

    def inside(src):
        if not len(src):
            return np.zeros(0, dtype=bool)
        yy = np.clip(src["y"].astype(int), 0, good.shape[0] - 1)
        xx = np.clip(src["x"].astype(int), 0, good.shape[1] - 1)
        return good[yy, xx]

    kr, kd = inside(src_r), inside(src_d)
    n_raw, n_den = int(kr.sum()), int(kd.sum())

    out = {
        "fill_fraction": frac_bad,
        "masked_fraction": float(bad.mean()),
        "ceiling": ceiling,
        "corr_in_out": corr,
        "fraction_of_ceiling": corr / ceiling if ceiling else float("nan"),
        "std_ratio": std_ratio,
        "std_ratio_ideal": ceiling,
        "threshold_adu": thresh,
        "n_raw": n_raw,
        "n_denoised": n_den,
        "source_survival": n_den / max(n_raw, 1),
    }

    if n_raw:
        x = src_r["x"][kr]
        y = src_r["y"][kr]
        f_r, _, _ = sep.sum_circle(r_s, x, y, aperture)
        f_d, _, _ = sep.sum_circle(d_s, x, y, aperture)
        keep = f_r > 0
        if keep.any():
            ratio = f_d[keep] / f_r[keep]
            order = np.argsort(f_r[keep])
            out["flux_retained_median"] = float(np.median(ratio))
            out["flux_retained_quintiles"] = [
                float(np.median(ratio[q])) for q in np.array_split(order, 5)
            ]
        # How many of the raw detections repeat in the independent stack. This
        # is the sanity check that caught the artefact: real sources repeat,
        # fill-edge detections do not.
        src_o = sep.extract(o_s, nsigma * float(bg_o.globalrms), minarea=5)
        ko = inside(src_o)
        if len(src_o) and ko.any():
            xy_o = np.column_stack([src_o["x"][ko], src_o["y"][ko]])
            d, _ = cKDTree(xy_o).query(np.column_stack([x, y]),
                                       distance_upper_bound=3.0)
            out["repeatability"] = float(np.isfinite(d).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", choices=ARMS)
    ap.add_argument("--test", default="ngc5907")
    ap.add_argument("--filters", default="L,R,G,B")
    args = ap.parse_args()

    arms = args.arm or [a for a in ARMS if (LADDER / a / "results.json").exists()]
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    stacks_dir = LADDER / "stacks"
    all_rows = []

    for filt in filters:
        raw = np.load(stacks_dir / f"{args.test}__{filt}_train0.npy")
        other = np.load(stacks_dir / f"{args.test}__{filt}_train1.npy")
        for arm in arms:
            p = LADDER / arm / f"{args.test}_{filt}_denoised.npy"
            if not p.exists():
                log(f"{filt} {arm}: no denoised array yet — skipped")
                continue
            den = np.load(p)
            m = masked_metrics(raw, other, den)
            m.update({"arm": arm, "filter": filt})
            all_rows.append(m)
            q = " ".join(f"{v:.3f}" for v in m.get("flux_retained_quintiles", []))
            log(f"{filt:3s} {arm:15s} masked {100 * m['masked_fraction']:5.2f}%  "
                f"{100 * m['fraction_of_ceiling']:4.0f}% of ceiling  "
                f"src {m['n_denoised']}/{m['n_raw']} "
                f"({100 * m['source_survival']:3.0f}%)  "
                f"flux {m.get('flux_retained_median', float('nan')):.4f}  "
                f"[{q}]  repeat {100 * m.get('repeatability', float('nan')):.0f}%")
            del den
        del raw, other
        log("")

    out = LADDER / "rescored.json"
    out.write_text(json.dumps(all_rows, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
