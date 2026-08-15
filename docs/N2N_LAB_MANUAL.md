# Noise2Noise denoiser — lab manual

The experimental record for the N2N denoiser: what was tried, what it measured,
and what is actually true right now. `N2N_SPARK_RUNBOOK.md` is the *procedure*
— how to run the chain. This is the *evidence* — why it is in the state it is.

Conventions borrowed from `lab_report/README.md`: every number is traceable to a
frame, method is recorded and not just conclusions, and **negative results are
kept**. Most of this document is negative results. That is the point.

---

## Current state — 2026-08-14

**Usable as a display product. Not a science product. Specifically not for the
transient search.**

Best configuration, and the one to reproduce:

```bash
python scripts/n2n_train_stacks.py R 300 --seed 0     # ~30 min on the Spark
# L1 · 60 epochs · split-half stacks · residual="linear" · lr 3e-4
# -> local/models/n2n_stack_R_300s.pt   (kept as ..._s2.pt)
```

It is applied **once to a finished stack**, not to every sub.

Two measurements, because they answer different questions and a model can pass
one while failing the other. Both on held-out m92, R 300 s, 56 subs.

**Existing sources** — aperture flux, denoised stack ÷ raw stack:

| bin | n | median SNR | ratio | gate 0.97 |
| --- | --- | --- | --- | --- |
| brightest | 13 | 81,672 | 0.9817 | pass |
| mid | 87 | 12,858 | 0.9534 | fail, marginally |
| faint | 236 | 1,933 | 0.9365 | fail |
| very faint | 4,947 | 22 | 0.7480 | fail |

**A newly appearing source** — injected PSFs, incremental response:

| aperture SNR | recovered |
| --- | --- |
| 2.9 | 0.66 |
| 5.7 | **1.07** |
| 14.3 | **1.13** |
| 22.8 | 1.08 |
| 114.2 | 0.94 |

Background RMS 1.52 -> 0.23, a 6.5x reduction.

**Read those together.** Flux of *existing* sources is preserved to 2-6% down to
SNR ~2000 and then degrades to -25% at the detection limit. The response to an
*added* source is non-monotonic and crosses unity — it under-recovers the
faintest by a third and **over**-recovers mid-SNR sources by 8-13%. A transient
search measures exactly that second quantity, which is why the denoiser is barred
from it: a source that appears would be measured 13% too bright at some
brightnesses and 34% too faint at others, with no single correction.

The 80/20 framing that matters: **80% of real detected sources sit below SNR
114**, in the range the injection curve covers. The reassuring bright-bin numbers
describe the other 20%.

### What is fixed, and what is not

Fixed, and each of these was a real bug that produced entirely plausible output:
destructive registration (the root cause), `normalise()` overflowing its stated
range, the correction being applied in asinh instead of linear space, an unseeded
RNG, and duplicated DataLoader worker streams. Bright-star flux went from
**0.0023 to 0.98** over 2026-08-12/13.

Not fixed: the faint end, and the over-recovery at mid SNR. Neither has an
identified mechanism.

**Do not judge this denoiser by a downsampled full-frame view.** That view has
looked plausible at every stage of this investigation, including when the network
was emitting a constant uncorrelated with its input.

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

### 13. L2 refutes the faint-end explanation; 60 epochs beats 130

Two results from one A/B on split-half stacks, same seed and split, scored on
held-out m92's full stack.

**Epochs.** L1 at 60 epochs beats the 130-epoch run on every bin:

| bin | 130 ep | 60 ep | gate |
| --- | --- | --- | --- |
| brightest | 0.9646 | **0.9735** | pass |
| mid | 0.9570 | **0.9735** | pass |
| faint | 0.9918 | **0.9946** | pass |
| very faint | 0.7278 | **0.8104** | fail |

Three of four bins pass. Consistent with `best` landing at epoch 45, 64, 67 and
56 across four runs — the back half of a 130-epoch run has never helped and here
measurably hurts. **Set epochs to ~60.**

**L2 is worse everywhere, and that refutes the standing explanation.**

| loss | rms | brightest | mid | faint | very faint |
| --- | --- | --- | --- | --- | --- |
| L1 | 0.2396 | 0.9735 | 0.9735 | 0.9946 | 0.8104 |
| L2 | 0.1878 | 0.9010 | 0.8516 | 0.8647 | 0.7227 |

The faint-end suppression has been attributed throughout — in the runbook, from
before this investigation — to L1 optimising the conditional *median*, which for
a source near the detection limit sits at background. If that were the mechanism,
L2's conditional mean should have relieved it. It made it worse, and dragged
three passing bins below the gate with it.

What the rms column shows instead: L2 smooths *harder* (0.1878 vs 0.2396),
because squared error punishes large residuals and pushes toward smoother output.
**Flux loss tracks smoothing aggressiveness, not the median-vs-mean property of
the loss.** The L1-median story is not supported by measurement and should not be
repeated.

The likelier explanation is less convenient: a source at 1-3 sigma in the stack
is nearly indistinguishable from noise pixel by pixel, and any denoiser tuned to
remove structure at the noise scale suppresses it. N2N can in principle separate
them — the source is consistent across both halves, the noise is not — but with
4 training pairs it does not learn to. That makes it a data/capacity limit, not
a loss-function bug, and more frames or more targets is the lever.

**Best configuration found: L1, ~60 epochs, split-half stacks, residual=linear.**

### 14. Injection-recovery: the suppression is pervasive, not faint-end

"very faint = 0.81" was a bin median. Injecting 810 synthetic PSFs (FWHM 6 px)
across nine amplitudes into the held-out m92 full stack turns it into a curve.

Scored as the *incremental* response `D(S+I) - D(S)` against the true injected
signal in the same aperture — denoising is non-linear, so that difference is the
response to the added source alone, with host, sky and any underlying star
cancelled by construction. Measuring flux at injection sites directly would
confound all three.

Arm 2 is a **coincidence filter**, the non-learned baseline this investigation
lacked: keep a pixel only where both independent half-stacks exceed
`k*sigma_half`, else set to background. Note the test is `min(A,B)`, not
`|A-B|` — a real source contributes equally to both halves so it *cancels* in
the difference, leaving pure noise; thresholding the difference selects on noise
alone and carries no information about whether a source is present.

| aperture SNR | N2N | coinc k=1.0 | k=1.5 | k=2.0 |
| --- | --- | --- | --- | --- |
| 2.85 | **0.53** | 0.21 | 0.07 | 0.00 |
| 5.71 | **0.55** | 0.35 | 0.14 | 0.06 |
| 8.56 | **0.57** | 0.47 | 0.29 | 0.13 |
| 14.27 | 0.60 | **0.69** | 0.54 | 0.39 |
| 22.83 | 0.66 | **0.83** | 0.74 | 0.64 |
| 57.09 | 0.77 | **0.93** | 0.90 | 0.86 |
| 114.17 | 0.84 | **0.97** | 0.96 | 0.94 |

**Neither reaches the gate in this range.** The network never does at all; the
coincidence filter only at SNR 114, where the raw stack was already fine.

They **cross over at SNR ~10**: the network wins below it, the filter above.
That is the hard-decision penalty — at low per-pixel SNR each half independently
falls below threshold by chance and the AND discards real sources. At k=1.0 the
filter keeps only **7.56% of pixels**; the rest are zeroed. The network's bias is
also flatter (0.53-0.84 over a 40x SNR span, against 0.21-0.97), and a flatter
response is easier to characterise and calibrate than one that couples strongly
to brightness.

**Reconciled with the photometry bins**, which appeared to disagree (brightest
0.9735 against 0.84 at SNR 114). They measure almost disjoint regimes:

| bin | n | median aperture SNR |
| --- | --- | --- |
| brightest | 13 | 81,672 |
| mid | 87 | 12,858 |
| faint | 236 | 1,933 |
| very faint | 4,947 | **22.2** |

Only `very faint` overlaps the injection range, and there the two agree (0.66
injected at SNR 22.8 against 0.73 measured). So both are valid: recovery rises
smoothly with SNR and reaches 0.97 only around SNR 1e3-1e4. **80% of real
detected sources sit inside the injected range**, where recovery is 0.53-0.84 —
the reassuring bins described the other 20%.

Caveat: this used an unseeded draw that checkpointed at **epoch 7** with val
0.7152, worse than the 0.6957 of the best measured configuration, so the network
arm is pessimistic by an unknown amount. Which is why:

### 15. Training is now reproducible

`n2n_train_stacks.py --seed N`, plumbed through `trainer.train(seed=)` to both
datasets, the half-stack partition, and torch. With `cudnn.deterministic` two
runs of the same seed are now bitwise identical (verified: same weights, same
val loss); without the cuDNN flags a seeded run still drifted ~4e-5 from
algorithm selection and atomics.

Before this, a run could not reproduce its own result, and the good draw above
was unrecoverable.

### 16. Quarter splits — more pairs, worse model. Depth match beats pair count.

Data thinness looked like the binding constraint (4 pairs, val bottoming out at
epoch 7), so `build_split_stacks` gained an adaptive split count: `frames // 6`
clamped to [2, max_splits]. Adaptive rather than fixed so deep targets give more
pairs without pushing shallow ones below usable depth.

At max_splits=4 that is 18 stacks and **24 pairs** against 10 and 5 — training
pairs go from 3 to 15. Both arms seeded 0, same DSOs, same held-out m92,
everything else identical.

| | splits=2 | splits=4 | delta |
| --- | --- | --- | --- |
| brightest | 0.9817 | 0.9901 | +0.008 |
| mid | 0.9534 | 0.9761 | +0.023 |
| faint | 0.9365 | 0.9305 | -0.006 |
| very faint | 0.7480 | 0.7776 | +0.030 |
| inj SNR 2.9 | 0.6645 | 0.5027 | **-0.162** |
| inj SNR 5.7 | 1.0690 | 0.5552 | **-0.514** |
| inj SNR 22.8 | 1.0844 | 0.6995 | **-0.385** |
| inj SNR 114 | 0.9407 | 0.7453 | **-0.195** |

**Real-source photometry barely moved; the incremental response collapsed.**
Quarters suppress added flux by 25-45% where the baseline sits near unity.

Best explanation: quarter-stacks are sqrt(2) noisier than the full stack the
model is applied to, so they teach shrinkage calibrated for noisier data, which
over-shrinks at inference. **Matching training depth to inference depth beat
multiplying pairs 5x.** `max_splits` is back to 2; the parameter stays so the
experiment can be repeated.

Two cautions. The baseline's **over-recovery** — 1.07-1.13 at mid SNR, i.e. it
*adds* 8-13% flux to an injected source, non-monotonically — is itself a bias,
is harder to calibrate than steady suppression, and is NOT explained by the
depth story. And these are single seeds with checkpoints at different epochs
(47 vs 27), so magnitudes are soft.

Verified before believing any of it: the difference image's pedestal in blank
regions is +0.00002 ADU, contributing 0.00 ADU to an 8 px aperture, so the
above-unity recovery is real and not an aperture artefact.

**Structural finding, and the one worth carrying forward:** real-source
photometry and injected-source response are *different measurements*, and a
model can look fine on the first while being badly biased on the second. The
transient search measures a new source appearing — that is the injection
response, not the bins.

### Note on the holdout change

Dropping abell2218 (9 frames, below the new depth floor) left 5 DSOs, and the
seeded split then holds out **m92 only** — so **m13 is now in training**. Every
earlier result held out both globulars, meaning the model had never seen a dense
star field. Results from step 16 onward are therefore not comparable to earlier
ones; m13 entering training is a competing explanation for any improvement on
m92.

### 17. Pooled L+R+G+B — also worse, and the obvious explanation is wrong

The argument for pooling: data thinness limits this, L+R+G+B at 300 s is 807
frames against R's 204, `normalise()` removes the sky-brightness differences
before the network sees anything, and what remains (PSF, noise correlation, star
profiles) comes from optics and sensor and is shared. Unlike quartering it adds
*scenes* at the same stack depth, so it avoids what sank step 16.

Built with `build_split_stacks_streaming` — 807 frames is ~200 GB against 121 GB
of RAM, so groups are loaded, registered, stacked and freed one at a time.
Group key is `(dso, filter)`, never bare dso: an L stack and an R stack of one
target are different scenes photometrically, and pairing them is the same class
of error as the misregistration bug. 36 stacks, 18 pairs, 18 groups. m92 excluded
**in every filter** via `--exclude`, because a group-level holdout would have
held out `m92|R` while training on `m92|L` and `m92|B` — the same field.

Against the R-only baseline on held-out m92, identical stack, sources and
injection sites:

| | R-only | pooled | delta |
| --- | --- | --- | --- |
| brightest | 0.9817 | 0.9808 | -0.001 |
| mid | 0.9534 | 0.9424 | -0.011 |
| faint | 0.9365 | 0.8757 | **-0.061** |
| very faint | 0.7480 | 0.7060 | **-0.042** |
| inj SNR 5.7 | 1.069 | 0.832 | **-0.237** |
| inj SNR 114 | 0.941 | 0.722 | **-0.219** |
| rms after | 0.2332 | 0.1410 | smooths far harder |

**Worse on every measurement.** 5x the training pairs, at matched depth, and it
lost.

**The mechanism is unidentified, and the obvious candidate is refuted.** The
natural explanation is that chromatic focus gives each filter a different PSF, so
one shared prior cannot fit them all. Measured from `frame_stats.json`, median
FWHM is L 2.507", R 2.343", G 2.463", B 2.559" — a 2-9% spread, well inside each
filter's own 16-84 range of roughly 2.0-2.9". Broadband PSF differences cannot
account for it. (Narrowband does differ: Ha 1.92", O-III 1.96".)

What can be said: the pooled model smooths much harder, and in every experiment
here heavier smoothing has come with worse flux recovery. That is a correlation
across L2, quarters and pooling, not a demonstrated cause.

Keep per-filter models. `scripts/n2n_train_pooled.py` stays for re-running it.

### Standing tally of what has been tried against the 2026-08-13 baseline

L1 / 60 epochs / split-half stacks / residual=linear has now survived three
attempts to improve it — L2 (step 13), quarter splits (step 16) and pooled LRGB
(step 17). All three measured worse. Two of the three also refuted the mechanism
proposed for them beforehand.

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

Resolved and closed: the bright-end inversion (step 11, linear-space
correction), the epoch count (step 13, now 60), reproducibility (step 15), and
whether N2N suits this data at all (step 3, yes).

Still open, roughly in order of what would be worth knowing:

- **The faint end, 0.75 at the detection limit.** Three attempts to move it have
  failed — L2 (13), quarter splits (16), pooled LRGB (17). No identified
  mechanism. The L1-median story that stood for months was tested and is wrong.
- **Over-recovery at mid SNR.** The current model *adds* 8-13% flux to an
  injected source at SNR 5-23, non-monotonically, crossing unity. Unexplained,
  and it is the property that most directly disqualifies the denoiser from the
  transient search. Verified not to be an aperture or pedestal artefact
  (step 16).
- **Why heavier smoothing tracks worse recovery.** L2, quarters and pooling all
  smoothed harder and all recovered less flux (rms 0.14-0.19 against the
  baseline's 0.23-0.24). Consistent across three independent interventions, but
  correlation only — never tested as a cause, and smoothing strength has never
  been varied directly as the single variable. This is the most promising
  untested lead.
- **What actually limits this.** Data thinness was the standing hypothesis and
  it now looks wrong: two separate 5x increases in training pairs (quarters,
  pooling) both made it worse. Whatever the constraint is, it is not simply pair
  count.
- **Tiling artefact.** Checkerboard from the overlap blend, seen on the sub-based
  model, never checked on the stack model.
- **Source-bias** (step 7), unconfirmed across seeds and inside the noise floor.
- **The gradient-tail trigger** from step 1, still unidentified.
- **Validation leakage in the pooled path.** The holdout is group-level
  (`dso|filter`), so it can hold out `abell2151|L` while training on
  `abell2151|B`. It does not affect any reported m92 number — those used
  `--exclude` — but it makes pooled val optimistic.

## Design note: why stacks and not subs

Raised 2026-08-13, **implemented and adopted the same day** (step 12,
`nn/stacks.py`). This section records the reasoning; the measurements are in the
chronology. The original chain denoised every sub and then stacked; the current
one leaves subs alone and denoises the **stacked** image, training on split-half
stacks.

Why it is better posed than what came before:

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

What it appeared to cost: far less training data — one pair per target per
filter instead of thousands of patch pairs across 204 frames. That worry has
since been tested twice and did not hold up. Quartering the stacks (16) and
pooling four filters (17) each multiplied training pairs ~5x and each measured
*worse*, so pair count is not what limits this. Full-resolution stacks also
yield many patches per pair, which recovers more than the pair count suggests.

**The sub-based and stack-based models are not interchangeable**, which is why
they have separate checkpoint names (`n2n_R_300s.pt` vs `n2n_stack_R_300s.pt`).
A sub-trained model is fitted to single-frame noise, sky sigma ~10 ADU; a
56-frame stack is ~1.5 ADU and, after registration and interpolation, spatially
correlated rather than per-pixel independent. `normalise()` would rescale the
amplitude but not the correlation structure, and the correlation structure is
what the learned prior is tuned to. Pointing one at the other's data is exactly
the class of silent mismatch this pipeline has hit twice.

## Cost, measured on the Spark

Current path — **stacks**, R 300 s, 204 frames:

| step | wall |
| --- | --- |
| scan + register + build half-stacks | ~10 min |
| train, 60 epochs | ~31 min (31 s/epoch) |
| denoise one full stack | ~35 s |
| injection test (`n2n_injection_test.py`) | ~8 min |
| model comparison (`n2n_compare_models.py`) | ~5 min |

Pooled L+R+G+B is ~50 min to build (807 frames) and streams group-by-group;
loading it whole would need ~200 GB against 121 GB of RAM.

Superseded — the **sub-based** path, kept for reference:

| step | wall |
| --- | --- |
| scan + registration + dataset build | ~20-25 min |
| train, 130 epochs | ~75 min (35 s/epoch) |
| denoise 204 frames | ~50 min (14.6 s/frame) |
| `n2n_evaluate.py`, 56-frame target | ~5-6 h (FWHM-bound) |

That 5-6 h gate is FWHM-bound and measures two things, only one of which still
applies: its convergence curve tests whether denoising *bought frames*, which is
meaningless once the denoiser runs after stacking. The photometry half is the
live gate and is what `n2n_compare_models.py` measures in ~5 min.

Peak RSS ~99 GB of 121 GB on the sub-based path, ~64 GB while building stacks.
Do not run a stack or transit search alongside either.
