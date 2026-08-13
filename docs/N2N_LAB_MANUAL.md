# Noise2Noise denoiser — lab manual

The experimental record for the N2N denoiser: what was tried, what it measured,
and what is actually true right now. `N2N_SPARK_RUNBOOK.md` is the *procedure*
— how to run the chain. This is the *evidence* — why it is in the state it is.

Conventions borrowed from `lab_report/README.md`: every number is traceable to a
frame, method is recorded and not just conclusions, and **negative results are
kept**. Most of this document is negative results. That is the point.

---

## Current state — 2026-08-13

**The denoiser does not work, and must not be used for anything.** Not science,
not display.

Measured on held-out m92, R 300s, frame
`2026-06-15_22-56-52_R_773_300.00s_0051.fits`, with the checkpoint trained
2026-08-12 (epoch 64 of 130):

| quantity | raw | denoised | verdict |
| --- | --- | --- | --- |
| background RMS | 10.007 | 2.388 | 4.19x lower — the one genuine improvement |
| flux, brightest bin (n=24) | — | **0.0023** | 99.8% of flux destroyed |
| flux, mid bin (n=57) | — | **0.0117** | 98.8% destroyed |
| flux, faint bin (n=138) | — | **0.5468** | 45% destroyed |

The gate's failure threshold is 0.97. This is not a marginal miss.

Visually the full frame looks *better* — smooth background, cluster legible. A
300 px crop on the brightest star shows it has been erased outright and replaced
with a mottled patch. There is also a visible checkerboard texture from the
tiled-inference overlap blend, and in faint regions sharp point-like artefacts
survive while real PSFs are removed, which is backwards.

**Do not judge this denoiser by a downsampled full-frame view.** That view has
looked plausible at every single stage of this investigation, including when the
network was emitting a constant.

---

## Instrument and data

Per `lab_report/README.md`: PlaneWave CDK17, QHY600M, 0.26 "/px. Typical FWHM
~6 px. All work below is R filter, 300 s subs, 204 frames across 6 DSOs
(abell2151 54, abell2218 9, m13 20, m92 56, ngc5033 19, ngc5907 46).

Runs on the Spark (`spark-3129`) only — `subs_dir` is defined for that host
alone and the observatory PC has no torch. Frames at
`/home/taylor/Desktop/Targets`, ~50 GB per denoised set.

The data is **heavily noise dominated**, which governs everything else. On
abell2151 in asinh units: total std 0.841, noise std 0.782, signal std 0.311.

---

## Chronology

### Starting point (2026-08-11, commit 91b43b9)

Three bugs fixed but never executed anywhere — no torch on the machine they were
written on. Per-DSO registration restored, train/inference normalisation
re-unified, BatchNorm removed (correctly: this is a regression where the absolute
output level carries the photometry).

### 1. Training diverged at epoch 2

First real execution. Train loss 0.20 -> **24834**, val 0.19 -> 1438, no
recovery. Because the checkpoint is selected on best val, the saved model was
the *epoch-1* weights — an untrained network that would have passed through
denoise and evaluate looking like a result.

Removing BatchNorm exposed a heavy-tailed gradient it had been hiding. Measured
over 250 batches at lr=1e-3, no clipping: median grad norm 0.2, p99 3.1, **max
45.8** (221x median), 8% of batches above 1.0.

Fix: `clip_grad_norm_(max_norm=1.0)`. After it, 130 epochs ran monotonically.

**Refuted along the way:** the obvious explanation — bright star in patch drives
huge gradient — is false. Correlation between patch peak and grad norm measured
**-0.001**, and the top-5% grad-norm batches had slightly *lower* peaks. The
trigger remains unidentified. Clipping bounds it regardless.

### 2. The stable run was also worthless

130 epochs, best val 0.2231 at epoch 45, no divergence, clean checkpoint. It
looked like success and was reported as such. It was not.

Denoising produced 204 files of the right size whose stars measured as *negative*
dips. Direct test of the model:

```
INPUT  patch : min=-0.407  max=837.212  std=1.7886
OUTPUT patch : min= 0.136  max=  0.464  std=0.0198
corr(input, output) = -0.0158
```

The network ignored its input and emitted a constant. Its val loss of 0.2230 was
the **constant-predictor baseline of 0.2263** — which is why the loss curve
looked converged.

Signals that had been misread as healthy: the loss plateau from epoch 5, and
`best` freezing at epoch 45 and never moving.

### 3. Is N2N even applicable to sky-dominated frames?

Asked honestly, answered quantitatively. On m92 in linear units:

```
L1 of PERFECT predictor (clean image) = 0.1276
L1 of CONSTANT predictor (median)     = 0.2263
room between them                     = 43.6% of baseline
share of L1 mass from source pixels   = 45.26%
```

**N2N is applicable.** There is ample signal; the model captured ~1.5% of it.
This also refuted a hypothesis stated earlier in the investigation — that L1
gives the network no incentive to model sources because stars are ~0.004% of
pixels. On m92, 1.05% of pixels exceed 10 sigma and they carry 45% of the L1
mass.

### 4. normalise() was not producing what it claimed

Docstring said `~[0,1]`; measured max was **837** (m92) and 631 (ngc5907),
because 99% of pixels are sky so `p99-p1` spans the sky *noise*.

A linear scale cannot fix this — the dynamic range is ~4000:1:

| scale | max | sky std |
| --- | --- | --- |
| p99 (then-current) | 837 | 2.08 |
| p99.99 | 12 | 0.030 |
| p100 | 1.00 | 0.0025 |

Either stars overflow or sky underflows. Replaced with `asinh(x / sky_sigma)`:
max ~9.4, sky noise 0.50, exactly invertible (round-trip 4e-7), and no longer
field-dependent (m92 9.4 vs ngc5907 8.7, against 837 vs 631).

**The caveat was stated at the time and has since come due:** `sinh` grows
exponentially, so network error at the bright end is amplified on inversion.

### 5. Root cause: registration was destroying alignment

The real reason for the collapse, found by measuring against phase correlation
as ground truth:

```
                    BEFORE   AFTER (old code)
m92       frame1     0.10  ->  2.84
          frame3     0.10  ->  5.44
          frame4     8.61  ->  0.67   <- the one real dither, fixed
abell2151 frame1     0.00  ->  5.58
          frame4     0.00  -> 14.02
```

Frames arrive aligned to 0.0-0.1 px; the old estimator shifted them apart by
3-14 px, i.e. **1-2.3 FWHM**. N2N requires two noisy views of the *same* scene.
Given pairs that far apart, the local mean is the correct answer to the question
being asked, and the collapse was the optimiser succeeding.

Mechanism: `_find_stars` decimated by 2 (centroids +-1-2 px), then
nearest-neighbour matched two 50-star lists at 200 px tolerance. In a dense
field that pairs *different* stars; the median of mismatched deltas is a
spurious shift, applied unconditionally with nothing checking it helped.

Replaced with phase cross-correlation, star matching demoted to fallback, shifts
under 0.5 px not applied at all (the order-3 spline low-pass filters the frame,
correlating the noise N2N assumes is independent).

On the full 204-frame set: **113 frames arrive already aligned, 85 genuinely
dithered, 0 unregistered.** So the old code was actively corrupting ~57% of the
training set.

Verified per DSO — m13 is genuinely dithered (6.4 to 39.7 px, monotonic drift
within 2026-06-19) and is correctly shifted; abell2151/abell2218/ngc5907 arrive
aligned and are correctly left alone.

### 6. Retrain with all fixes

130 epochs, lr 3e-4, best val 0.7016 at **epoch 64**. Collapse check on held-out
targets, scored against the measured ceiling `sqrt(corr(frame_i, frame_j))`:

| target | corr | ceiling | % of ceiling |
| --- | --- | --- | --- |
| m92 | 0.3423 | 0.6400 | 53.5% |
| m13 | 0.2953 | 0.6175 | 47.8% |

**The collapse is fixed** (from -0.0158). The model is mediocre, not broken.

### 7. Source-biased patch sampling

Uniform cropping spends most patches on pure sky. Biased sampling toward source
content, mixed 70/30 with uniform so sky is still seen.

First implementation used a block **max** and did nothing measurable (0.010% vs
0.011% source pixels) — over a 32x32 block the max of pure sky already sits near
3.2 sigma by extreme-value statistics, so noise dominates the score. Counting
pixels above a sky-relative threshold works: 2.4-2.9x more source content per
patch.

A/B on held-out m92, 30 epochs, one seed each:

```
uniform      0.3861  (60.3% of ceiling)
source-bias  0.4020  (62.8% of ceiling)   +4.1%
```

**Kept but not proven.** One seed; +4% is within plausible seed noise.

### 8. Denoising output measured — the current failure

Section "Current state" above. Bright-end flux destroyed by the `sinh`
inversion amplifying bright-end error, exactly as flagged in step 4.

Net effect of asinh: **relocated the damage rather than removing it.** The
linear model erased everything uniformly; this one erases bright sources while
partially preserving faint ones. Both fail the gate.

---

## Standing conclusions

1. **Denoised frames are not a science product and not a display product.**
   Calibrated linear frames remain the science product for transit work, the
   colour-magnitude diagram, and the transient search.
2. **Loss is not evidence.** A constant predictor drives it down. Judge a
   checkpoint by `corr(input, output)` against the measured ceiling, and by
   aperture flux ratios — never by the curve or a downsampled preview.
3. **The ceiling is not 1.0.** These frames are noise dominated; a perfect
   denoiser reaches ~0.64 on m92, ~0.37 on abell2151. An early version of the
   collapse check used a naive >0.9 threshold and failed a working model.
4. **Every failure here produced plausible output.** A diverged run wrote a
   checkpoint; a collapsed net wrote 204 correctly-sized FITS; destructive
   registration reported success on every frame. Assume the next one will too.

## Open questions

- **Bright-end inversion.** The live problem. Candidates: predict a *residual*
  rather than the image, so the network never has to reproduce a large absolute
  value; or clamp the asinh scale so bright pixels stay in the linear regime.
- **Epoch count.** `best` has landed at epoch 45, then 64, of 130 on successive
  runs, and a 30-epoch subset run beat the full 130-epoch run on the same
  held-out target (60.3% vs 53.5%). The back half looks actively harmful. Not a
  clean comparison — different training sets.
- **Tiling artefact.** Visible checkerboard from the overlap blend, not yet
  investigated.
- **Source-bias**, unconfirmed across seeds.
- **The gradient tail trigger** from step 1, still unidentified.

## Cost, measured on the Spark

| step | wall |
| --- | --- |
| scan + registration + dataset build | ~20-25 min |
| train, 130 epochs | ~75 min (35 s/epoch) |
| denoise 204 frames | ~50 min (14.6 s/frame) |
| evaluate, 56-frame target | ~5-6 h (FWHM-bound) |

Peak RSS during training ~99 GB of 121 GB. Do not run a stack or transit search
alongside it.
