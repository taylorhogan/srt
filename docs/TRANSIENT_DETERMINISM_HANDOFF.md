# Transient search: run-to-run non-determinism — handoff

**Status: cause NOT identified.** An earlier diagnosis was wrong and is recorded
below so it is not repeated. Continue on the Spark.

## The symptom

The same transient search, on the same frames, with no code change between
invocations, produced materially different results:

| run | raw residuals | difference-image RMS |
|-----|---------------|----------------------|
| 1   | 824           | 1.566897             |
| 2   | 893           | 1.365186             |

An 8% change in detections and **13% in background RMS**. Downstream, the
injection test's measured 50% recovery floor moved between runs:

| run | 400 ADU | 800 ADU | floor    |
|-----|---------|---------|----------|
| A   | 3/3     | 3/3     | 400 ADU  |
| B   | 0/3     | 3/3     | 800 ADU  |

A factor of two in the quoted sensitivity. This blocks publishing a limiting
magnitude: there is no stable number to convert into one. The monotonicity
assertion held in both runs, which is why the tests assert curve *shape* rather
than a pinned value (see `utils/injection_checks.py`).

## Dataset

`ngc5907`, filter `R` — 46 LIGHT frames over 5 nights. The pipeline splits it
into a 36-frame template (4 nights, 2026-06-02 → 06-08) and a 10-frame science
night (06-09); baseline 7.0 d. About 5.6 GB at ~122 MB/frame.

On the Spark: `/home/taylor/Desktop/Targets/ngc5907/cdk17/*/LIGHT/`, already
populated by `scripts/sync_nina_targets_to_spark.bsh`. `configs/config_public.py`
maps host `spark-3129` to that root, so no path edits are needed.

**Peak memory ~11 GB** (each frame is 9576x6388 float32 ≈ 245 MB once
registered). This is why the test kept dying on the observatory PC — run one
process at a time and do not run two searches in one interpreter.

## Established facts (measured, trust these)

1. **astroalign is not bit-reproducible for a minority of frames.** Repeating
   `find_transform` on the same inputs: 2 of 13 frames differed, by up to
   `9.1e-13` in the transform matrix. A *robust* pair repeats exactly, so a
   single-pair test is not sufficient — that mistake was made here first.
2. Its RNG is not controlled by `np.random.seed` in current scikit-image, so
   seeding does nothing.
3. `ois` is **not installed**, so `_psf_match_and_subtract` takes the scipy
   Gaussian path. Do not chase Alard–Lupton.
4. `_robust_scale_bg` subsamples with `np.random.default_rng(0)` — **seeded**,
   not a source.
5. `_fill` and `_gaussian_match` are deterministic given their inputs.
6. `_frame_fwhm` returns a **median** over detected sources — a discrete
   selection, so it *can* convert a last-bit input change into a finite jump.
7. No `sep` pixstack overflow/retry occurred (checked in `iris.log`), so the
   adaptive-threshold path is not involved.

## The wrong diagnosis (do not repeat)

It was concluded that astroalign's 1e-13 jitter propagates through registration
into `_frame_fwhm`, whose median flips discretely, changing the PSF-matching
kernel width and hence the whole difference image.

**That chain is plausible but was never demonstrated, and the arithmetic does
not support it.** A direct test perturbed the science image by `1e-7` — six
orders of magnitude *larger* than the real jitter — and produced only 633 vs 636
detections. A far larger perturbation gives a far smaller effect than the real
824 vs 893. Something else is producing the large effect.

Critically, **the FWHM was never observed to differ between the two original
runs.** That was inferred backwards from the symptom.

## A proposed fix, deliberately NOT in the tree

A candidate fix was written and then **reverted on purpose**, so this diagnostic
bisects unmodified production code. An unproven fix in the tree would change the
very behaviour being measured.

The change was two lines in `transient_search/difference.py`: quantise
`_frame_fwhm`'s return value to a 0.05 px grid.

```python
_FWHM_QUANTUM_PX = 0.05        # module level
...
q = round(val / _FWHM_QUANTUM_PX) * _FWHM_QUANTUM_PX
return q if q > 0 else default # in place of: return val
```

It does what it claims in isolation — a `1e-12` perturbation no longer moves the
returned value — and is defensible on its own terms, since seeing is not knowable
to better than ~0.1 px and selecting a convolution kernel from a full-precision
float is spurious precision. But it did **not** stabilise the difference image in
the proxy test (635 vs 634 detections with it enabled), and it is **not**
established as the fix.

Re-apply it only if the bisection shows `FWHM value=` as the first divergence.
`scripts/diag_transient_determinism.py` prints `quantum=ABSENT` when the constant
is missing, so it works either way.

## The experiment to run

`scripts/diag_transient_determinism.py` fingerprints every pipeline stage —
each registered frame, the combined science and template images, every
`_frame_fwhm` value, the scale/background fit, the difference image, and every
detection call — so two runs can be diffed and the **first** stage that
disagrees identified. Bisect; do not reason backwards.

```bash
cd ~/Documents/srt && source .venv/bin/activate
python scripts/diag_transient_determinism.py ngc5907 R > /tmp/run1.txt
python scripts/diag_transient_determinism.py ngc5907 R > /tmp/run2.txt
diff /tmp/run1.txt /tmp/run2.txt | head -40
```

Run at least 3 pairs — the effect may be intermittent, and two identical runs
prove nothing on their own.

### Reading the diff

- **First divergence at a `REG frame[...]` line** → registration is the origin.
  Check whether the *set* of accepted frames also changed (`n_accepted`); a
  frame passing QA in one run and failing in another would change the median
  combine wholesale and would easily explain 13%. This is the leading
  hypothesis and was never tested.
- **`REG` identical but `IN science`/`IN template` differ** → the combine is at
  fault, not registration.
- **Inputs identical, `FWHM value=` differs** → the original hypothesis is
  correct after all; the quantisation fix stands and its comment can be kept.
- **Inputs and FWHM identical, `SCALE`/`DIFF` differ** → look inside
  `_gaussian_match` / scipy, or non-associative float reduction.
- **Everything identical up to `DIFF`, only `DET` differs** → non-determinism
  inside `sep` itself; every residual count is then approximate and thresholds
  need revisiting.

### Then

Once the origin is known, re-run the acceptance test to confirm the floor is
stable across runs:

```bash
python transient_search/test_injection.py ngc5907 R    # exits non-zero on failure
```

Two consecutive runs must give the same 50%-recovery flux. Only then is a
limiting magnitude quotable, via the photometric zero-point.

## Also worth checking

`stacking/stacker.py` shares `_register_frames`, so if registration is the
origin the convergence curves published on the lab site inherit the same
instability.
