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

> **The L2 half is superseded by step 18.** This A/B also ran on misaligned
> pairs. L2 is now the default in both runner scripts and was part of the run
> that took Ha from 0.016 to 1.11 flux retention. The 60-epoch result stands.

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

> **Superseded by step 19.** This ran on training pairs that were 8-36 px
> misaligned (step 18). Re-measured on the fixed pipeline, pooling *wins*. The
> table below records what was measured at the time and is left intact; the
> conclusion drawn from it is wrong.

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

### 18. The training pairs were never aligned — 0.016 to 1.11 flux

Found 2026-08-17, and it is the largest single defect this pipeline has had.

`build_split_stacks` calls `stack_paths` once per split, and `stacker.stack`
picks its own reference frame unless given one. So **the two halves of every N2N
training pair were registered onto different grids**. Measured on two stacks of
sh2-92 built by that path: **(-8, +8) px** and **(+36, 0) px**, with the offset
drifting from (20,-28) to (28,-32) across the field — rotation, not a pure shift.
`stacker.stack` has taken `shared_reference` all along and says in its docstring
what it is for; `color_process` passes it, `nn/stacks.py` did not.

**How it surfaced.** Not from the pictures — from an impossible number. The
collapse check scored O-III at **285% of its ceiling**, and nothing can correlate
with its input better than a perfect denoiser does. The ceiling is
`sqrt(corr(stack_a, stack_b))`, and it was being taken on those same unaligned
stacks, so misalignment was read as noise and pushed the ceiling *down*. Both
things were broken by one cause, and only the impossibility gave it away.

**Why deleting stars is the loss-optimal response.** N2N assumes input and target
differ only in noise. Offset them and the target holds background where the input
holds a star, so erasing it lowers the loss. The network cannot compensate by
shifting: astroalign fits rotation, making the error position-dependent, and
convolution is translation-invariant by construction.

**Why the stars came out black.** `residual="linear"` predicts
`asinh(sinh(t) + g)`, so erasing a bright star needs `g ~= -sinh(t)` — thousands
of sky sigma. Overshoot by a percent and the pixel lands well below background.
That is why the **brightest** quintile was worst, at **-0.033** flux retained,
while fainter ones stayed positive. No other hypothesis explained the inversion.

Measured on sh2-92 (bubble-trained, depth-matched, same seed):

| | broken | fixed | ideal |
| --- | --- | --- | --- |
| training pair offset | 8-36 px | 0 px | 0 |
| Ha sources surviving | 13% | 104% | 100% |
| Ha flux retained | 0.016 | 1.11 | 1.00 |
| Ha corr vs ceiling | 30% | 76% | 100% |
| Ha std ratio | 0.21 | 0.52 | 0.48 |
| O-III sources | 64% | 100% | 100% |
| O-III flux | 0.62 | 1.05 | 1.00 |
| O-III std ratio | 0.46 | 0.478 | 0.476 |

O-III was quietly damaged too — 64% of its sources, and nothing in the picture
said so. Before the fix the two filters diverged wildly (Ha 30% of ceiling,
O-III 77%); after it they agree. The gulf was never filter-specific, it was how
far apart that run's two stacks happened to land, with one training pair and
nothing to average against it.

**Fixed** by `nn.stacks.shared_reference_for` — one reference per `(dso, filter)`,
sharpest frame from `frame_stats.json` when cached, middle frame otherwise —
threaded into every `stack_paths` call for the group. `scripts/n2n_holdout_run.py`
then measures the residual offset and refuses to train past `--max-offset`
(default 2 px), so it cannot recur silently.

**Confound.** The loss moved L1 -> L2 in the same run, so the credit is not split.
The registration fix is the one with a measured mechanism; a `--loss l1` rerun
would separate them.

**Two metrics were wrong as well**, both fixed in `collapse_check` /
`source_survival`:

- The ceiling must be taken on *aligned* stacks. Unaligned it read misalignment
  as noise: on identical smoke-test data it rose **0.1926 -> 0.3182** once
  aligned, and Ha's real ceiling went 0.3119 -> 0.4510.
- Source survival needs **one absolute threshold taken from the raw sky**. Per
  image, the denoised frame's own 5 sigma sits far lower in ADU because its sky is
  8-30x quieter, so it "finds" *more* sources than the raw frame — 203% and 382%
  measured — with fluxes inflated to match. That measures the threshold, not the
  network.

### 19. Pooling re-tested post-fix — it wins, overturning step 17

`scripts/n2n_pool_ab.py`, 2026-08-19. One model trained on bubble Ha + O-III
(2 groups, 2 pairs) against the per-filter models from the same seed and the same
`split_depths`, so both arms stack exactly the same frames. The **test stacks are
not rebuilt** — they are loaded from the per-filter run, so both arms are scored
on byte-identical data. Pairs are still formed strictly within `(dso, filter)`;
pooling shares weights, never pairs.

| | per-filter | pooled | ideal |
| --- | --- | --- | --- |
| Ha corr / ceiling | 76% | **84%** | 100% |
| Ha sources | 104% | **99%** | 100% |
| Ha flux retained | 1.109 | **1.055** | 1.000 |
| Ha std ratio | 0.516 | **0.498** | 0.476 |
| O-III corr / ceiling | 75% | **82%** | 100% |
| O-III sources | 100% | 102% | 100% |
| O-III flux retained | 1.048 | 1.070 | 1.000 |
| O-III std ratio | 0.478 | 0.486 | 0.476 |

Ha improves on all four. O-III gains 7 points of correlation and drifts ~2% on
the rest. The headline is correlation, +8 and +7 points on identical test data —
the largest gain since the fix itself — and Ha's flux excess **halving from 11%
to 5.5%**.

`pairs_per_epoch` stayed at 2000 across both arms, so the pooled model got half
the sampling per pair and won anyway.

Note this contradicts the standing note in `scripts/n2n_train_pooled.py` that
narrowband should not be pooled because Ha/O-III sky is far darker and shifts the
regime from sky-dominated toward read/dark-dominated. Ha and O-III pool fine.
Whether narrowband pools with *broadband* is still untested, and is now the
obvious next experiment: it would give far more than 2 pairs.

### 20. The LRGB ladder — pooling wins (its "stops paying" half superseded by 22)

**Read step 22 before using the arm-3 numbers here.** This step held
`pairs_per_epoch` at 2000 across arms of 1, 4 and 14 groups, so the 14-group arm
got 143 patch draws per pair against the 4-group arm's 500. Its conclusion that
pooling "stops paying" at 4 groups was a budget artefact and does not hold; the
finding that pooling across filters wins does. Everything measured below is
still true of what it measured.

`scripts/n2n_lrgb_ladder.py`, 2026-08-19. Step 19's open question was what
limits this pipeline, with data thinness leading. This is the experiment it
named, on broadband instead of narrowband: **18 pairable (dso, filter) groups at
300 s, 778 frames**, with ngc5907 held out entirely and all four filters scored.

Three arms differing in exactly one thing — which groups are in the training
pool. Same seed, same L2 loss, same 60 epochs, same `pairs_per_epoch` of 2000,
and every arm scored on **byte-identical test stacks**, because the stacks are
built once to `local/n2n_ladder/stacks/` and the arms only choose subsets.

| arm | groups | corr/ceiling | spread | mean abs flux error | mean std/ideal |
| --- | --- | --- | --- | --- | --- |
| per-filter (abell2151, one filter each) | 1 | 47-106% | 59 | 3.33% | 1.017 |
| **pooled-filters (abell2151, L+R+G+B)** | **4** | **63-69%** | **6** | **0.52%** | **0.982** |
| pooled-scenes (all but ngc5907) | 14 | 36-54% | 19 | 0.50% | 0.991 |

Per filter, coverage-masked (see below):

| filt | arm | %ceil | sources | flux | quintiles faint->bright |
| --- | --- | --- | --- | --- | --- |
| L | per-filter | 63% | 98% | 1.0072 | 1.008 1.023 1.019 1.005 1.000 |
| L | pooled-filt | **69%** | 96% | 0.9938 | 0.953 0.981 0.997 0.997 0.999 |
| L | pooled-scenes | 54% | 99% | 1.0077 | 0.996 1.000 1.006 1.018 1.005 |
| R | per-filter | 62% | 102% | 1.0098 | 1.018 1.032 1.024 1.001 0.996 |
| R | pooled-filt | **63%** | 95% | 0.9936 | 0.945 0.981 0.998 0.998 1.001 |
| R | pooled-scenes | 42% | 97% | 1.0024 | 0.972 0.976 1.000 1.014 1.005 |
| G | per-filter | 47% | 99% | **1.1080** | 1.039 1.083 1.106 1.117 1.168 |
| G | pooled-filt | **63%** | 96% | 0.9956 | 0.960 0.987 0.999 0.997 0.999 |
| G | pooled-scenes | 37% | 98% | 1.0053 | 0.994 0.988 1.011 1.018 1.004 |
| B | per-filter | **106%** | 93% | 1.0082 | 0.927 0.987 1.008 1.010 1.013 |
| B | pooled-filt | 65% | 95% | 0.9962 | 0.963 0.993 1.000 0.998 0.999 |
| B | pooled-scenes | 36% | 98% | 1.0046 | 0.987 0.990 1.015 1.016 1.004 |

**Pooling across filters wins, and it wins by removing variance rather than by
raising a score.** Four single-pair models behave four different ways — G carries
an 11% flux excess rising with brightness (1.039 -> 1.168, the halo defect of
step 18), B barely denoises at all — while one pooled model behaves the same way
on all four filters and lands photometry within 0.5% everywhere. It did that
while getting **4x less patch sampling per pair**, which is the strongest form
the result could take.

**Pooling across scenes as well makes it worse.** 14 groups is worse than 4 on
correlation for every filter. The mechanism is not what it first looked like:
`std/ideal` is ~0.99, so arm 3 removes almost exactly the right *amount* of
variance, and its point-source photometry is the best of the three (97-99%
sources, flux within 0.5%). A first reading of "right variance, wrong
correlation" as *inventing texture* was checked at 1:1 and is **wrong** — arm 3
is visibly smoother than arm 2, not more textured. It over-smooths extended
structure, which a 3-px aperture does not measure.

**The confound, which this run cannot remove.** `pairs_per_epoch` was held at
2000 for every arm, so patch draws per pair went 2000 -> 500 -> 143. What is
measured is scene count traded against per-pair sampling, not scene count alone.
Arm 2 beating arm 1 at 4x less sampling is real; arm 3 losing at 3.5x less again
is not attributable. **The experiment that settles it is arm 3 at `--pairs
7000`**, matching arm 2's per-pair sampling, one ~2 h run.

#### The stacks are full-frame and the uncovered part is a flat constant

Found while checking why G looked catastrophic. `stacker.stack` crops to its
high-coverage region but **skips that crop when `shared_reference` is passed** —
correctly, since cropping per group would put groups back on different grids and
undo step 18 — and fills residual NaN with the global sky median instead. So
every stack keeps uncovered pixels as a flat constant, and the two halves of a
pair have *different* uncovered footprints because they hold different dithered
frames.

On ngc5907 G that is 0.28% of stack A and 0.47% of stack B, all in one band at
y < 1000, and `sep` finds ~460 detections in it — **48% of that stack's 952
sources**. Consequences, both of which produced confident wrong numbers:

- Source survival read **57%** for a model that had destroyed nothing. Masked to
  the covered region it is **99%**.
- Repeatability between the two independent test stacks read **51%**, against
  97-99% for L, R and B. Masked, it is **99%** — G agrees as well as any filter.

`scripts/n2n_ladder_rescore.py` recomputes every metric with those pixels
excluded, from the saved arrays, so nothing needs retraining. **`rescored.json`
is authoritative for this run**; the per-arm `results.json` predates the fix.

The models were still *trained* on patches that can contain fill —
`N2NDataset` samples patch origins with no knowledge of coverage. That is the
real defect and it is **not** fixed.

#### Calibration: what the headline metric reads at both ends

A 2-epoch checkpoint on ngc5907 G, which is still the identity because
`residual="linear"` zero-initialises the head, read **324% of ceiling**, std
ratio 0.9984, flux 1.0000 in all five quintiles. So the scale is approached from
above and is not monotonic:

    ~0%    collapsed - constant output
    100%   perfect denoiser
    324%   identity - did nothing (this target, this depth)

A number rising past 100% means the model stopped denoising, not that it
improved. B per-filter at 106% is that: near-identity, and its 93% source count
is the worst of the twelve.

#### Also settled

- **Every one of the 18 training pairs aligned to (0,0) px**, across 6 targets
  and 4 filters. Step 18's fix was previously evidenced only on bubble/sh2-92.
- **The tiling artefact is present on the stack model**, closing that open
  question. Visible at 1:1 in the winning model as a diagonal weave, absent from
  a 6x-downsampled preview. Residual power peaks at 64 px and 512 px, which are
  exactly `denoise_frame`'s `overlap` and `tile_size` — indicative of the
  overlap blend rather than transposed-convolution checkerboard, though a
  two-tile-offset difference is the test that would prove it.
- **The quality gate cuts hard and unevenly**, so nominal depth overstates by
  25-46%: abell2151 rejects 12% of R, 24% of G, 33% of L and 46% of B. The
  manifest records `accepted` per stack.
- **The FWHM cache has never worked on this machine.** `frame_stats.json` is
  written by the observatory PC with Windows path keys, and
  `_load_precomputed_fwhm_stars` matches on `os.path.abspath`, so every lookup
  misses silently. Costs 65% of stack wall time, and means
  `shared_reference_for` has always used the middle frame rather than the
  sharpest. Not fixed — the cached values disagree with the stacker's own
  (FWHM ~11% high, star counts ~1.5x low) and the quality gate cuts on both.
  See the runbook.

### 21. The denoiser is not colour-safe on low-surface-brightness structure

2026-08-20, found by looking at the LRGB render rather than at a number. The
composite shows a green halo around ngc5907 that the raw does not. It is real,
it is the model's, and **no metric in the ladder could have caught it** — every
one of them is aperture photometry on point sources, and this is a ratio error
between channels on extended emission.

Measured on the depth-matched render channels (all four on one shared reference,
so the regions are the same pixels), gradient-removed exactly as `compose` does,
in an annulus following the edge-on disk:

| region | L | R | G | B |
| --- | --- | --- | --- | --- |
| halo, raw (ADU) | 5.080 | 1.681 | 1.920 | 1.609 |
| halo, denoised | 3.877 | 0.602 | 0.987 | 0.657 |
| **fraction kept** | **76.3%** | **35.8%** | **51.4%** | **40.8%** |
| far sky control, raw | -0.175 | -0.098 | -0.080 | -0.119 |
| far sky control, denoised | -0.040 | -0.046 | -0.038 | -0.076 |

The control matters: far from the galaxy both raw and denoised sit at zero, so
the halo signal is real extended emission and not a residual gradient.

Colour ratios against L in the halo move by a third to a half:

    R/L  0.331 -> 0.155   (-53%)
    G/L  0.378 -> 0.255   (-33%)
    B/L  0.317 -> 0.169   (-47%)

**G is retained 1.44x as well as R, and that ratio is the green halo.**

The galaxy *core* is untouched — G/L 0.300 -> 0.295, R/L 0.324 -> 0.317, B/L
0.228 -> 0.223, all within 1-2%. So the defect is confined to low surface
brightness. Point sources are likewise fine: a bright star's radial profile is
preserved to 0.1% at the core and 0.8% at r=15 px.

Retention does not track SNR cleanly. Halo signal in units of each channel's own
sky sigma is L 1.30, R 0.60, G 0.52, B 0.50, so L's high retention fits, but
among R/G/B the ordering does not — R has the *best* SNR of the three and the
*worst* retention. R's stack was also the shallowest (11 of 22 frames survived
the gate against 19 for G and B), which is the obvious suspect and is untested.

Consequences:

1. **The denoiser cannot be the colour path as it stands.** LRGB channels are
   essentially never equal depth, and a depth- or noise-dependent retention on
   faint extended flux turns straight into a colour cast.
2. This is the same phenomenon as step 20's arm-3 result read through a
   different instrument — 14 groups over-smoothed extended structure while
   scoring 97-99% on sources. Extended low-SB emission is what this model gives
   up, and point-source metrics are blind to it by construction.
3. **A gate on extended structure is missing.** Everything in the current gate
   (`source_survival`, aperture flux, corr against ceiling) is either
   point-source photometry or a whole-frame average. The measurement above —
   region medians against a far-sky control, per channel — is cheap and should
   become part of it.

#### Two display traps found rendering that comparison

Neither is the model's fault and both produced a picture that looked like a
verdict on it.

1. **`compose`'s black point is a percentile of each channel** (`BLACK_PCT = 65`).
   Denoising shrinks the sky's spread ~2.5x, so the same percentile lands at a
   different ADU on the denoised channel than the raw one — and by a different
   amount per channel, since each has its own noise. Structure that sat below
   black in the raw rises above it, gets asinh-amplified, and the per-channel
   differences become colour. The first raw/denoised composite came out in
   rainbow blotches for this reason alone. This is the same error class as the
   per-image detection threshold `source_survival` was fixed for: a relative
   threshold on two images of different noise measures the noise.

2. **`compose`'s default white point saturates this field completely.** It is
   p99 of max(R,G,B), which on ngc5907 is 12 ADU against a 32,000 ADU star, so
   the galaxy and every star clip in *both* images. The raw then looks *more*
   detailed, because its noise dithers the clipping edge while the denoised
   output crosses it cleanly — and the "square stars" that suggests are JPEG
   blocking at extreme contrast. Measured star profiles are identical to 0.1%
   at the core and 0.8% at r=15 px. `--white-pct 99.95` (229 ADU) renders both
   properly.

The rule that survives both: **one stretch, computed from the raw, applied
unchanged to both frames** — which `save_pair_pngs` already documents and
`compose` does not do.

#### Inference depth changes how hard the model smooths

Measured on the same channels at two depths, sky noise raw -> denoised:

| depth | L | R | G | B |
| --- | --- | --- | --- | --- |
| matched (~11-19 frames) | ~3x | ~3x | ~3x | ~3x |
| full (57/35/38/33) | **17x** | **20x** | **24x** | **26x** |

Galaxy integrated signal survives both (1.005-1.022), and this is *not* what
made the full-depth render look glassy — that was the stretch above. But the
ratio itself is real: `normalise()` divides out each frame's own sky sigma, so
the network sees unit-sigma noise either way, and it still removes 6-8x more of
it from a deeper stack. The learned prior is keyed to something normalisation
does not equalise — most likely the spatial correlation structure that
registration and interpolation leave behind, which differs with stack depth.
Untested, and the cleanest probe available for what this model actually keys on.

#### Two more, for the trap list

- **A correlation taken on linear data measures the bright stars, not the
  denoising.** The first version of `n2n_ladder_rescore.py` omitted the asinh
  normalisation that `collapse_check` applies, and every arm and filter came
  back at corr 0.9996-0.9998 against a "ceiling" of 0.9987-0.9993 — 100% of
  ceiling on all twelve rows. The dynamic range here is ~4000:1, so Pearson r on
  raw ADU is set almost entirely by whether a handful of 66,000 ADU stars line
  up, and any two stacks of the same field agree on those. asinh is linear where
  the sky noise lives, which is the part being judged. Uniform, plausible, and
  meaningless — it is the reason `normalise()` exists, rediscovered.

- **`scripts/n2n_train_pooled.py` never passed `loss=`**, so it kept training L1
  after step 18 made L2 the default everywhere else in the chain, silently, for
  three days. Fixed 2026-08-20 (defaults to L2, `--loss` to override). No
  measurement in this manual came from it, but any pooled checkpoint built with
  that script between 2026-08-17 and 2026-08-20 is L1 and should be rebuilt.

### 22. Scene count re-tested at matched sampling — 14 groups ties 4

2026-08-20, `--arm pooled-scenes --pairs 7000 --suffix _p7000`. Step 20 held
`pairs_per_epoch` at 2000 for every arm, so patch draws *per pair* went
2000 -> 500 -> 143 as groups went 1 -> 4 -> 14, and its "pooling stops paying at
4 groups" could not be told apart from "the budget got too thin". This run gives
14 groups the same 500 draws per pair that 4 groups got.

| arm | groups | draws/pair | corr/ceiling | spread | mean abs flux err | mean faint quintile |
| --- | --- | --- | --- | --- | --- | --- |
| per-filter | 1 | 2000 | 47-106% | 59 | 3.33% | 0.998 |
| pooled-filters | 4 | 500 | 63-69% | 6 | **0.52%** | 0.955 |
| pooled-scenes | 14 | 143 | 36-54% | 19 | 0.50% | 0.987 |
| pooled-scenes_p7000 | 14 | **500** | **63-70%** | 8 | 0.80% | 0.947 |

Head to head at 500 draws/pair:

| filt | 4 groups | 14 groups | 4 grp flux | 14 grp flux |
| --- | --- | --- | --- | --- |
| L | 69% | 70% | 0.9938 | 0.9964 |
| R | 63% | 64% | 0.9936 | 0.9803 |
| G | 63% | 63% | 0.9956 | 0.9915 |
| B | 65% | 66% | 0.9962 | 1.0001 |

**Both halves of step 20's conclusion move, in opposite directions.**

1. "Pooling stops paying at 14 groups" was **wrong** — it was the fixed budget.
   Starved at 143 draws/pair the same 14 groups scored 36-54%; fed at 500 they
   score 63-70%. G alone goes 37% -> 63%.
2. "More scenes keep helping" is **also unsupported.** 14 groups matches 4 on
   correlation (+1 point or level on every filter, inside run-to-run variance)
   and is slightly *worse* on photometry — mean flux error 0.80% against 0.52%,
   faint quintile 0.947 against 0.955.

**The whole measurable gain is the 1 -> 4 step**, i.e. pooling across the filters
of one target. Going to 14 groups costs 3.5x the training time (110 min against
~31) and buys nothing. Data thinness, as "not enough scenes", is dead as the
leading explanation for what limits this pipeline.

The epoch curves say the same thing from the other side. Best epoch tracks total
patch exposure per pair, not group count:

    per-filter      1 grp   2000/pair   epoch 13-16
    pooled-filters  4 grp    500/pair   epoch 26
    pooled-scenes  14 grp    143/pair   epoch 36    <- starved, still improving late
    pooled-scenes  14 grp    500/pair   epoch 14    <- fed, then overfits

At 7000 pairs the run overfits from epoch 14 onward — val rises monotonically
0.5966 -> 0.6052 while train falls 0.5318 -> 0.5062. Four times the scenes did
not delay overfitting; it arrived *sooner* than with 4 groups. Whatever limits
this model, more distinct fields is not it.

`std_ratio` on L reached 0.4020 against an ideal of 0.4022, the closest any
configuration has come.

### 23. ic1396 in HOO — the denoiser deletes the nebula

2026-08-20, and the first run where a prediction from `n2n_extended_check.py`
was written down before the picture existed and then matched it exactly.

ic1396 at 300 s: 88 frames (Ha 50, O-III 38) across 2026-08-18/19, stacked onto
**one shared Ha reference** so the two channels land on the same pixels, 33 and
32 surviving the quality gate. Denoised, then composed R<-Ha, G/B<-O-III.

**The nebula is gone.** The raw shows faint Ha filling the whole frame threaded
with dark dust lanes; the denoised output is a few isolated bright knots on
black, and the dust structure vanishes with the nebulosity it was silhouetted
against.

This was predictable, and predicted, from surface brightness alone. ic1396's
nebulosity sits at **1-5 sigma** — p99 is 3.5 ADU above a 1.2 ADU sky, p99.9 is
40 ADU — and the retention curve says that band is where almost everything is
lost:

| smoothed SB (sky sigma) | Ha kept | O-III kept |
| --- | --- | --- |
| 1..2 | 0.250 | 0.255 |
| 2..4 | 0.314 | 0.294 |
| 4..8 | 0.380 | 0.384 |
| 8..16 | 0.530 | 0.460 |
| 16..32 | 0.614 | 0.691 |
| >32 | 0.992 | 1.008 |

Only above ~32 sigma is signal safe, and those are exactly the knots that
survived in the image. A galaxy loses a halo to this; a nebula loses its subject.

**Which model, settled by measurement rather than by which scored better.** The
ladder winner (`pooled-filters`, trained on abell2151 broadband) was beaten by
`pooledNB` (trained on bubble Ha + O-III) in **every bin on both filters** —
0.380 vs 0.309 at 4-8 sigma, 0.992 vs 0.946 above 32. Domain match beat the
better benchmark score, which is what `scripts/n2n_train_pooled.py` has argued
all along about narrowband sky being a different noise regime. Neither model had
seen ic1396.

The denoised frame looks *cleaner* and is worse. Anyone judging by how smooth
the sky came out would call it an improvement — the same failure the 2026-08-15
run made when it reported quality as `sep.Background().globalrms`.

#### The gate that was missing

`scripts/n2n_extended_check.py`. It bins by **smoothed** surface brightness
rather than per-pixel value — at 1-2 sigma a pixel is mostly noise, so per-pixel
binning puts signal and noise in the same bin and the ratio collapses toward
zero for any working denoiser, which says nothing — and it reports the
**cross-channel spread** per bin, because a model that eats 30% from every
channel dims the image while one that eats 36% from R and 51% from G changes its
colour. Everything else in the gate is point-source photometry or a whole-frame
average, and both are blind to this.

Run it on any model before trusting it on extended targets.

### Standing tally of what has been tried against the 2026-08-13 baseline

**This tally was invalidated on 2026-08-17 and is kept for the record.** It read:
L1 / 60 epochs / split-half stacks / residual=linear had survived three attempts
to improve it — L2 (13), quarter splits (16), pooled LRGB (17), all three worse.

All three ran on training pairs registered to different references and sitting
8-36 px apart (step 18). Two have since been re-measured on the fixed pipeline
and **both reversed**: L2 is now the default, and pooling beats per-filter models
(step 19). Quarter splits has not been re-run; its stated mechanism — shallower
stacks teach over-aggressive shrinkage — is independent of alignment, so that
verdict is the one most likely to survive, but it is now the only one resting on
unaudited ground.

The general lesson is worse than any single reversal: misregistration did not
merely degrade the models, it inverted the conclusions of the experiments used to
steer the design. Every A/B run before 2026-08-17 is suspect.

---

## Standing conclusions

1. **Denoised frames are not a science product, and are a display product only
   for point-source-dominated fields.** Calibrated linear frames remain the
   science product for transit work, the colour-magnitude diagram, and the
   transient search. Since step 18 point sources survive intact and their
   photometry is within a few percent — but steps 21 and 23 rule out anything
   whose subject is extended and faint. Low-surface-brightness flux is retained
   at 25-40% below ~8 sigma, unequally per channel, which costs a galaxy its
   halo colour and costs a nebula its nebula. **Never judge a denoised frame by
   how smooth the sky came out**; run `scripts/n2n_extended_check.py` first.
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

- **The faint end.** The three attempts that "failed" — L2 (13), quarter splits
  (16), pooled LRGB (17) — all ran on misaligned pairs (18). Two have reversed on
  re-test. Re-measure before treating any of it as known.
- **Over-recovery at mid SNR.** Survived the fix and is now the leading defect.
  Post-fix on sh2-92 the model *adds* 11% (Ha) and 5% (O-III) of real source
  flux, flat across brightness; pooling halves Ha's to 5.5%. Re-verified
  2026-08-17 against three benign explanations, all excluded: background pedestal
  (0.038 ADU, would give +1 ADU against +42 measured); PSF sharpening (the ratio
  *rises* with aperture radius, 1.106 at r=2 to 1.179 at r=20, and the denoised
  PSF is slightly *broader*); and background-fitting differences (one shared
  model gives the same ratio). The network adds a real halo around sources.
- **Why heavier smoothing tracks worse recovery.** The correlation held but the
  sign flipped: post-fix, pooling smooths *less* (std ratio 0.516 -> 0.498, toward
  ideal) and recovers *more*. A correlation that survives an intervention
  reversing both terms is better evidence of a real mechanism than the original
  three-way agreement was. Still never varied as the single variable.
- **What actually limits this.** Data thinness is back as the leading hypothesis.
  It was dismissed because two increases in pair count measured worse — but both
  were confounded by misalignment, and pooling reversed on re-test (19). Pair
  count is not sufficient on its own (quarters also raised it, at the cost of
  depth), but at *matched depth* more pairs now helps.

  **Answered by steps 20 and 22, and the answer is: not scene count.** Pooling
  the four filters of one target (1 -> 4 groups) is a large win — photometry
  error 3.33% -> 0.52%, cross-filter spread 59 -> 6 points. Going on to 14
  groups, at matched per-pair sampling, changes correlation by +1 point or less
  and makes photometry slightly worse, at 3.5x the training cost. The step-20
  appearance that 14 groups was much worse was a fixed `pairs_per_epoch`
  starving them at 143 draws per pair. So "not enough scenes" is no longer a
  live explanation, and what actually limits this is **still unidentified** —
  step 21's finding, that faint extended structure is what the model gives up,
  is the most promising place to look next.
- **Tiling artefact — CONFIRMED on the stack model** (step 20). Visible at 1:1 in
  the winning pooled model as a diagonal weave in blank sky, invisible in a 6x
  downsampled preview. Residual power peaks at 64 px and 512 px, exactly
  `denoise_frame`'s `overlap` and `tile_size`, which points at the overlap blend
  rather than transposed-convolution checkerboard. Proving it needs the same
  region denoised at two tile offsets and differenced.
- **Training patches can contain uncovered fill.** `stacker.stack` skips its
  coverage crop under `shared_reference` and fills uncovered pixels with the sky
  median; `N2NDataset` samples patch origins with no knowledge of coverage, so
  some fraction of every model's training patches are part flat constant. Step
  20 fixed only the *measurement* side of this. Unknown how much it matters —
  0.3-0.8% of pixels, but concentrated at one edge.
- **Why B behaves differently.** Alone among the filters it is near-identity when
  trained per-filter (106% of ceiling, 93% sources) and it is the group the
  quality gate cuts hardest (46% rejected on abell2151, leaving a 6-frame pair).
  Whether that is depth, passband, or the draw is untested.
- **Why faint extended retention differs per channel** (step 21). R keeps 36% and
  L keeps 76% of the same halo. Stack depth is the leading suspect (R survived
  the gate at 11 frames against 19) and separating it from passband needs one
  render with equal-depth channels.
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
