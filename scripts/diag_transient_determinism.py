#!/usr/bin/env python3
"""Bisect where the transient search stops being reproducible.

Background in docs/TRANSIENT_DETERMINISM_HANDOFF.md. Short version: two runs of
the same search on the same frames gave 824 vs 893 residuals and a difference
image whose background RMS moved 13%, which moved the measured 50% recovery
floor from 400 to 800 ADU. The cause is NOT yet identified.

This logs a fingerprint at every stage so two runs can be diffed and the FIRST
stage that disagrees can be found. That is the whole point: the earlier
investigation reasoned backwards from the symptom and got it wrong.

Usage (run twice, then diff):

    python scripts/diag_transient_determinism.py ngc5907 R > /tmp/run1.txt
    python scripts/diag_transient_determinism.py ngc5907 R > /tmp/run2.txt
    diff /tmp/run1.txt /tmp/run2.txt

Peak memory is ~11 GB for a 46-frame set (each frame is 9576x6388 float32,
~245 MB). Run one process at a time.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs import config
from transient_search import difference as tr


def fp(a) -> str:
    """Fingerprint an array: hash plus stats, so a mismatch says how big it is."""
    x = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    fin = x[np.isfinite(x)]
    h = hashlib.sha1(np.ascontiguousarray(x.astype(np.float32)).tobytes()).hexdigest()[:12]
    if fin.size == 0:
        return f"sha={h} (all non-finite)"
    return (f"sha={h} n={fin.size} mean={fin.mean():.9g} std={fin.std():.9g} "
            f"min={fin.min():.9g} max={fin.max():.9g}")


def main() -> None:
    dso = sys.argv[1] if len(sys.argv) > 1 else "ngc5907"
    filt = sys.argv[2] if len(sys.argv) > 2 else "R"
    print(f"# quantum={getattr(tr, '_FWHM_QUANTUM_PX', 'ABSENT')}")

    # --- stage 1: registration -------------------------------------------- #
    from transit_search.transit import _register_only as _orig_reg

    def reg_spy(paths, progress_cb=None):
        frames, accepted = _orig_reg(paths, progress_cb=None)
        print(f"REG  n_frames={len(frames)} n_accepted={len(accepted)}")
        for i, f in enumerate(frames):
            print(f"REG  frame[{i:03d}] {fp(f)}")
        return frames, accepted

    import transit_search.transit as _t
    _t._register_only = reg_spy

    # --- stage 2: the FWHM actually used, and the kernel it selects -------- #
    _orig_fwhm = tr._frame_fwhm

    def fwhm_spy(image, default=3.0):
        v = _orig_fwhm(image, default)
        print(f"FWHM value={v!r} on {fp(image)}")
        return v

    tr._frame_fwhm = fwhm_spy

    # --- stage 3: the scale/background fit --------------------------------- #
    _orig_sb = tr._robust_scale_bg

    def sb_spy(science, template, mask):
        # Log the fit population size, not just the answer. _robust_scale_bg
        # subsamples with np.random.default_rng(0).choice(xs.size, 200_000):
        # the seed is fixed but the DRAW is a function of xs.size, so a coverage
        # change of a few thousand pixels does not perturb the subsample, it
        # replaces it (measured overlap: 0.33%). Without this line the bisection
        # cannot tell that path apart from "the values shifted slightly", and
        # they have very different fixes.
        m = mask & np.isfinite(science) & np.isfinite(template)
        print(f"SCALE n_fit={int(m.sum())}")
        s, b = _orig_sb(science, template, mask)
        print(f"SCALE s={s!r} b={b!r}")
        return s, b

    tr._robust_scale_bg = sb_spy

    # --- stage 4: the difference image and its detections ------------------ #
    _orig_sub = tr._psf_match_and_subtract

    def sub_spy(science, template, coverage):
        print(f"IN   science  {fp(science)}")
        print(f"IN   template {fp(template)}")
        d = _orig_sub(science, template, coverage)
        print(f"DIFF {fp(d)}")
        return d

    tr._psf_match_and_subtract = sub_spy

    _orig_det = tr._sep_detect

    def det_spy(image, thresh_sigma):
        o, r = _orig_det(image, thresh_sigma)
        print(f"DET  thresh={float(thresh_sigma):.1f} n={0 if o is None else len(o)} rms={r!r}")
        return o, r

    tr._sep_detect = det_spy

    cfg = config.data()
    scratch = Path(os.path.join(_root, cfg["scratch"]["directory"])).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    tr.run_transient_search(
        dso, filt, Path(cfg["nina"]["image_dir"]),
        scratch / f"determinism_{dso}_{filt}.jpg",
        progress_cb=None,
        identify=False,          # network lookups are irrelevant and slow here
    )
    print("# done")


if __name__ == "__main__":
    main()
