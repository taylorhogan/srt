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

### 9. Clamping the asinh scale — helps 4x, does not reach the gate

The amplification `sinh` applies to network error on inversion is `cosh(t)`, so
a larger asinh scale (bright end nearer asinh's linear regime) should reduce it.
Measured on m92, the scale only ever trades one end against the other:

| mult | max t | sky noise in t | amplification at max t |
| --- | --- | --- | --- |
| 1 | 9.43 | 0.5007 | 6235x |
| 10 | 7.13 | 0.0540 | 624x |
| 100 | 4.83 | 0.0054 | 62x |

Reaching the linear regime (`t < 0.5`, amplification < 1.13) needs a scale ~2x
the peak signal, which puts sky noise at ~8e-5 — unresolvable. So the analytic
answer is that scale alone cannot fix it. Tested anyway, because a larger scale
might shrink the network's error faster than 62x amplification hurts.

Three 25-epoch models, same seed and frames, each denoising the same 2048^2 crop
of held-out m92 (62 sources, raw rms 11.072), scored on aperture flux:

| mult | rms | brightest | mid | faint |
| --- | --- | --- | --- | --- |
| raw | 11.072 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 2.768 | 0.0426 | 0.0240 | 1.2470 |
| **10** | 3.012 | **0.1735** | **0.1055** | **1.0068** |
| 50 | 2.572 | 0.0339 | 0.0325 | 0.9785 |

**Not monotonic.** mult=10 is an optimum; mult=50 regresses to worse than
mult=1 on the bright bins. Too small a scale and `cosh` destroys the bright end;
too large and sky noise falls to ~0.011 in t units, the network cannot resolve
what it is denoising, and its relative error grows faster than the reduced
amplification saves.

Worth 4x on bright and mid, and it fixes a second failure mode visible at
mult=1: the faint bin reads **1.2470**, i.e. faint sources come out 25%
*brighter* than raw — the denoiser adding flux, not just removing it. At mult=10
that is 1.0068.

But 0.17 against a required 0.97 is not a path to passing. Scored on flux rather
than loss or correlation deliberately: both of those looked acceptable at mult=1
while the photometry was being destroyed.

`ASINH_SIGMA_MULT` is kept in `nn/denoiser.py` as a single module constant, since
training and inference must agree on it and every other way of expressing that
has drifted at least once (c617047, c61cd26).

### 10. Measurement variance — two earlier A/B results retracted

`N2NDataset` seeded its RNG with `np.random.default_rng()` — unseeded, so patch
selection came from OS entropy and `torch.manual_seed()` never touched it. Two
runs of an identical configuration differed **5.8x** on brightest-bin flux
(0.0426 vs 0.2490).

Retracted: the mult=10 "optimum, worth 4x" of step 9, and the +4.1% for
source-biased sampling in step 7. Both effects are inside this noise floor.

Surviving, because they are far outside it or involve no training at all: the
collapse (-0.0158) vs a working net; the registration bug (measured by phase
correlation directly); and bright-end destruction, which lands between 0.002 and
0.43 across every configuration and seed against a required 0.97.

A second bug in the same place: with `num_workers=2` the dataset is copied to
each worker *with its RNG state*, and `__getitem__` ignores the sample index, so
every worker generated the identical patch sequence — each epoch was
num_workers copies of the same data. Every run in this investigation, including
both 130-epoch ones, saw half the intended patch diversity. Fixed with a
per-worker reseed keyed on pid.

`N2NDataset(seed=...)` is now available and experiments must pass one.

### 11. Linear-space correction — the first thing that worked

The correction has to be applied in **linear** units, not asinh. Predicting
`asinh(sinh(t) + g)` puts the correction in units of sky sigma *after* the
compression, so the ADU error is `sigma * dg` with no exponential factor. The
same correction magnitude of 3 moves a 4949-sigma star by 0.06% while moving the
sky by a full 3 sigma — the asymmetry the task needs.

This is what step 8's "asinh residual" got wrong: adding the correction *before*
the sinh leaves `cosh(t)` sensitivity fully intact, and it measured worse than
direct (0.0645 vs 0.2490).

Two seeds, 25 epochs, trained on abell2151/ngc5907/ngc5033 (40 frames),
evaluated on held-out m92:

| config | rms | brightest | mid | faint |
| --- | --- | --- | --- | --- |
| raw | 11.072 | 1.0000 | 1.0000 | 1.0000 |
| direct seed0 | 2.711 | 0.1425 | 0.0209 | 2.5547 |
| **linear seed0** | 2.702 | **0.9168** | **0.8220** | **0.9874** |
| direct seed1 | 2.840 | 0.4320 | 0.3646 | 8.0481 |
| **linear seed1** | 2.697 | **0.9171** | **0.8486** | **0.9873** |

Brightest 6.4x better, mid 39x better, faint back from 2.55 (sources emerging
155% too bright) to 0.987. **Noise reduction is undiminished** — rms 2.70
against raw 11.07, 4.1x, matching the direct arms — so this is not preserving
flux by declining to denoise.

The seed agreement is the strongest evidence for the mechanism: linear arms
match to 0.03% on brightest while direct arms swing 3x. Predicted by the theory
— under direct prediction every bright pixel is reconstructed through a
cosh(t)~6000 gain, so seed-to-seed differences are amplified into wild flux
errors; under linear correction the bright value passes through deterministically
and only a bounded correction is learned.

**Not a passed gate.** Mid sits at 0.82-0.85 against the required 0.97. Flux is
still being lost, at 15% rather than 99.8%. This is 25 epochs on 40 frames,
evaluated on a dense globular by a model trained on three sparse extragalactic
fields; a full run should do better, which is an expectation and not a result.

### 12. Split-half stacks — the distribution gap closes, the faint end fails

Built (`nn/stacks.py`, `scripts/n2n_train_stacks.py`): register each DSO's subs,
partition into two **disjoint** halves, combine each. The pair is two noisy views
of the same scene with a stack's noise character, so inference on the full stack
is in distribution. Disjointness is load-bearing — overlapping halves share noise
realisations and the network could cut its loss by reproducing that shared noise.

204 frames -> 12 half-stacks over 6 DSOs (depths 4 to 28 per half; the
`min_frames_per_dso=6` floor let abell2218 through at 4 and should be raised).
Held out m13 + m92, trained on 4 pairs, best val at epoch 56 of 130.

Denoising the **full** stack, flux against the raw full stack:

| dso | role | rms raw | rms den | brightest | mid | faint | very faint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ngc5907 | trained on | 3.2205 | 1.9932 | 0.6547 (n=4) | 0.9933 | 0.9852 | 0.8906 |
| m92 | held out | 1.5246 | 0.2041 | 0.9646 | 0.9570 | 0.9918 | 0.7278 |

Against the sub-based model on the same held-out m92 (0.7734 / 0.5712 / 0.9854):
brightest 0.77 -> 0.96, mid 0.57 -> 0.96.

**The generalisation gap is gone.** The held-out target now scores *better* on
the bright bins than the trained-on one, which says the sub-based run's train/val
chasm was the sub-vs-stack distribution mismatch, not overfitting.

**The failure moved to the faint end.** `very faint` is 0.7278 on m92 (n=4948)
and 0.8906 on ngc5907 (n=7511) — a 27% flux loss on the faintest sources,
measured on thousands of them.

This is the failure the runbook predicted at the outset and that nothing until
now was clean enough to expose: **L1's optimum is the conditional median**, and
for a source near the detection limit the posterior mass sits at background, so
the median sits at background too. Every earlier configuration failed so badly at
the bright end that faint-end suppression was invisible underneath it.

Also suspicious: 7.47x further noise reduction on a stack already 56 frames deep
(rms 1.52 -> 0.20). That is aggressive smoothing, and the erased faint sources
are its visible cost.

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

- **Bright-end inversion — largely solved** by the linear-space correction
  (step 11), `UNet(residual="linear")`, now the default. Brightest 0.917, mid
  0.82-0.85, faint 0.987, reproducible across seeds. Not yet at the gate's 0.97.
- **Next:** a full 204-frame run with `residual="linear"`, then the real
  `n2n_evaluate.py` gate. The subset result is a ranking of approaches, not a
  substitute for it.
- **Mid bin** is now the worst bin (0.82-0.85) where it used to be the best-hidden
  failure. Unclear yet whether more data fixes it or whether it needs its own
  explanation.
- **Epoch count.** `best` has landed at epoch 45, then 64, of 130 on successive
  runs, and a 30-epoch subset run beat the full 130-epoch run on the same
  held-out target (60.3% vs 53.5%). The back half looks actively harmful. Not a
  clean comparison — different training sets.
- **Tiling artefact.** Visible checkerboard from the overlap blend, not yet
  investigated.
- **Source-bias**, unconfirmed across seeds.
- **The gradient tail trigger** from step 1, still unidentified.

## Design alternative: denoise stacks, not subs

Raised 2026-08-13 and not yet tried. The current chain denoises every sub and
then stacks. The alternative is to leave subs alone and denoise the **stacked**
image, training on **split-half stacks** — split a target's subs in two, stack
each half independently, and use that pair as the N2N training pair.

Why it is better posed than what is built:

- **The pair is legitimate.** Two half-stacks are two noisy views of the same
  scene, which is exactly what N2N requires, and they carry the noise character
  of a stack rather than of a sub. Inference on the full stack is then
  in-distribution.
- **It removes the failure mode the whole gate is built around.** Per-frame
  denoising puts every frame through the same network with the same learned
  prior, so any error is identical across frames; independent noise averages
  down as sqrt(N) but a shared bias survives stacking untouched. Denoising once,
  after stacking, cannot launder a bias into an apparently-converging curve.
- **It is ~200x cheaper at inference.** 330 tiles instead of 204 x 330, and the
  stack is the artefact that actually gets published.
- **It matches what the denoiser is permitted to be.** Denoised output is a
  display product; the display product *is* the stack.

What it gives up: the "buy frames" claim — that denoising subs reaches a given
SNR from fewer of them. That claim only means anything if the denoiser runs
before stacking. It is also the claim least likely to survive the gate, for the
shared-bias reason above.

What it costs: far less training data. One pair per target per filter instead of
thousands of patch pairs across 204 frames, though full-resolution stacks yield
many patches each and recover some of that.

Note the current model must **not** simply be pointed at a stack. It is trained
on single-frame noise (sky sigma ~10 ADU); a 56-frame stack is ~1.3 ADU and,
after registration and interpolation, spatially correlated rather than per-pixel
independent. `normalise()` divides by the frame's own sky sigma so the
amplitude would be handled, but the correlation structure would not, and that is
what the learned prior is tuned to.

## Cost, measured on the Spark

| step | wall |
| --- | --- |
| scan + registration + dataset build | ~20-25 min |
| train, 130 epochs | ~75 min (35 s/epoch) |
| denoise 204 frames | ~50 min (14.6 s/frame) |
| evaluate, 56-frame target | ~5-6 h (FWHM-bound) |

Peak RSS during training ~99 GB of 121 GB. Do not run a stack or transit search
alongside it.
