#!/usr/bin/env python3
"""Score every ladder arm on extended-structure retention.

    python scripts/n2n_halo_rescore.py

The ladder was chosen on point-source metrics — `source_survival`, aperture
flux, corr against the ceiling. All of them are blind to the defect that decides
whether a model is usable on a galaxy: it can flatten a low-surface-brightness
halo while scoring 97-99% on sources, and because the flattening differs per
channel it lands as a colour cast (lab manual step 21, the green halo on
ngc5907). `extended_retention` measures it directly; this applies it to every
arm so the ladder can be re-read with that column present.

Scored on ngc5907, which is held out of training in every arm, using the same
`__L_train0` stacks the original rescore used. The number that matters is not
any single channel's retention but the SPREAD across channels, because the cast
is a ratio error.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

_spec = importlib.util.spec_from_file_location(
    "hr", os.path.join(_root, "scripts", "n2n_holdout_run.py"))
hr = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["x"]
_spec.loader.exec_module(hr)
sys.argv = _argv

TEST_DSO = "ngc5907"
FILTERS = ("L", "R", "G", "B")
MODELS = Path(_root) / "local" / "models"
STACKS = Path(_root) / "local" / "n2n_ladder" / "stacks"
OUT = Path(_root) / "local" / "n2n_ladder" / "halo.json"

# per-filter is four checkpoints, one per channel; the rest are one model applied
# to every channel.
ARMS = {
    "per-filter":          lambda f: MODELS / f"n2n_ladder_per-filter_{f}_300s.pt",
    "pooled-filters":      lambda f: MODELS / "n2n_ladder_pooled-filters_300s.pt",
    "pooled-scenes":       lambda f: MODELS / "n2n_ladder_pooled-scenes_300s.pt",
    "pooled-scenes_p7000": lambda f: MODELS / "n2n_ladder_pooled-scenes_p7000_300s.pt",
}


def main() -> int:
    from nn import denoiser
    dev = denoiser.best_device()
    rows = []
    for filt in FILTERS:
        src = STACKS / f"{TEST_DSO}__{filt}_train0.npy"
        if not src.exists():
            print(f"{filt}: missing {src.name}")
            continue
        raw = np.ascontiguousarray(np.load(src).astype(np.float32))
        print(f"\n=== {filt}  {src.name}  {raw.shape} ===", flush=True)
        for arm, pick in ARMS.items():
            mp = pick(filt)
            if not mp.exists():
                print(f"  {arm:22s} missing {mp.name}")
                continue
            t0 = time.time()
            model = denoiser.load_model(mp)
            den = denoiser.denoise_frame(raw, model, device=dev)
            ext = hr.extended_retention(raw, den)
            rows.append(dict(arm=arm, filter=filt, model=mp.name,
                             halo_kept=ext.get("halo_kept"),
                             core_kept=ext.get("core_kept"),
                             halo_raw=ext.get("halo_raw"),
                             halo_den=ext.get("halo_den"),
                             far_sky_raw=ext.get("far_sky_raw"),
                             far_sky_den=ext.get("far_sky_den"),
                             halo_frac=ext.get("halo_frac"),
                             sky_sigma=ext.get("sky_sigma")))
            print(f"  {arm:22s} halo {100*ext['halo_kept']:5.1f}%  "
                  f"core {100*ext['core_kept']:5.1f}%  "
                  f"far sky {ext['far_sky_raw']:+.3f}/{ext['far_sky_den']:+.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            del den, model
        del raw

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT}")

    print(f"\n{'arm':<22} " + "".join(f"{f:>8}" for f in FILTERS) + f"{'spread':>9}")
    best = None
    for arm in ARMS:
        vals = {r["filter"]: r["halo_kept"] for r in rows if r["arm"] == arm}
        if len(vals) < len(FILTERS):
            continue
        sp = 100 * (max(vals.values()) - min(vals.values()))
        print(f"{arm:<22} " + "".join(f"{100*vals[f]:7.1f}%" for f in FILTERS)
              + f"{sp:8.1f}p")
        if best is None or sp < best[1]:
            best = (arm, sp)
    if best:
        print(f"\nsmallest cross-channel spread: {best[0]} ({best[1]:.1f} points) "
              f"— least likely to cast colour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
