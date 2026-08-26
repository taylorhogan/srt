# Noise2Noise denoiser — lab manual

The experimental record for the N2N denoiser: what was tried, what it measured,
and what is actually true right now. `N2N_SPARK_RUNBOOK.md` is the *procedure*
— how to run the chain. This is the *evidence* — why it is in the state it is.

Conventions borrowed from `lab_report/README.md`: every number is traceable to a
frame, method is recorded and not just conclusions, and **negative results are
kept**. Most of this document is negative results. That is the point.

---

## Current state — 2026-08-23

**If you read one thing, read this.** The denoiser works and its limits are
measured.

- **Use it** on point-source-dominated fields and on bright nebulae. Point
  sources survive to 0.1%, sky noise drops ~55x, photometry is within 1-2% when
  trained deep.
- **Do not use it** where the subject is faint extended emission. It is
  band-limited: transfer ~0.98 above 512 px, ~0.72 at 128-256, ~0.33 at 4-8
  (step 29). ic1396 loses its nebulosity; NGC 6888's bright filaments survive.
  The discriminator is surface brightness, not object class — measure with
  `scripts/n2n_sb_profile.py` (step 23, and the routine-path section).
- **It is not colour-safe.** Channels are retained unequally, so SHO and LRGB
  composites shift hue. Worst measured: NGC 6888 S-II 0.556 against Ha 0.887
  (steps 21, 30-31).
- **Production models**: `n2n_pooledNB_300s.pt` (narrowband),
  `n2n_ladder_pooled-filters_300s.pt` (broadband). Six attempts to beat the
  narrowband one failed (step 31).
- **Never report an A/B without a seed control.** Floor: ~0.02 retention at
  1-2 sigma, ~0.012 at 4-8, ~0.03 on flux (step 28).

Levers that measurably work: **integration depth** (deeper stacks fix the
photometric excess and are the only thing that moves S-II), and **patch size**
(512, step 30). Levers that measurably do not: training target, scene count,
pooling scheme, channel count, source bias.

## Superseded — current state as of 2026-08-14

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

   **Done 2026-08-26**: `extended_retention()` in `scripts/n2n_holdout_run.py`.
   Three bands (core >10 sigma, halo 0.5-3 sigma, far sky <0.1 sigma) taken from
   a 64px-smoothed copy of the raw, point sources masked, far sky reported as an
   absolute level rather than a ratio of two near-zero numbers. Selection is on
   the *smoothed* frame on purpose: picking pixels by their own noisy value and
   reading the denoised frame there shows shrinkage even for a perfect denoiser,
   because a pixel at +1 sigma is mostly noise. Re-measured on this same ngc5907
   depth-matched set through the automatic regions: L 88 / R 50 / G 79 / B 51,
   core 91-101%, far sky ~0 in both — a **38.6 point** channel spread against the
   40.5 measured here by hand, from a region the operator never drew. The runner
   warns below 70% halo retention, and the summary flags a cross-filter spread
   over 15 points.

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

**Qualified by step 29.** Injection with ground truth shows the model is
band-limited rather than blind: large-scale emission survives at 96-98% and fine
texture at 30-50%. The picture and the retention number below stand; the
mechanism stated here does not.

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

### 24. Training depth — falsifies the fix it was aimed at, solves a different one

**Its retention numbers are retracted by step 28** (the +0.011/+0.036 gains sit
inside the seed-to-seed floor). The photometry finding stands at ~2x the floor.

2026-08-20, `scripts/n2n_depth_ab.py --train sh2-92 --test ic1396 --caps 22,0`.

**The prediction, written before the run:** the model over-smooths because its
shrinkage is calibrated to the noise level it trained on — ~3x sky-noise removal
at matched depth against 17-26x when the input is cleaner than anything it saw.
Every narrowband model had learned on 22-29 frame half-stacks; sh2-92 allows 51
and 72. So a deeper-trained model should hold low-surface-brightness structure
better and stop deleting nebulae.

Depth is the only variable: same target, same two filters, same seed, same
pooled architecture and L2 loss, same frame ordering. Both arms scored on the
**byte-identical ic1396 stacks** step 23 used, so pooledNB's row is directly
comparable. Post-gate depths: Ha 17,20 -> 44,42 and O-III 19,17 -> 63,55.

| model | trained on | depth | Ha 4-8σ | O-III 4-8σ | Ha flux | O-III flux | Ha src | O-III src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooledNB | bubble | 22/29 | **0.380** | **0.384** | 1.0619 | 1.0704 | 101% | 102% |
| cap22 | sh2-92 | 18/18 | 0.329 | 0.281 | 1.0593 | 1.0860 | 102% | 105% |
| full | sh2-92 | 43/59 | 0.340 | 0.317 | **0.9924** | **0.9930** | 96% | 97% |

**1. The prediction is falsified.** Deeper training moves 4-8 sigma retention by
+0.011 (Ha) and +0.036 (O-III). ic1396 is still destroyed — 34% and 32% of its
flux at the surface brightness it actually has. Depth is **not** the mechanism
behind the extended-structure loss, so the cause sits in the spatial prior
itself rather than in a miscalibrated noise level. That closes off the cheapest
hypothesis and points the next work at the loss function or the architecture.

**2. It solves the leading defect it was not aimed at.** The point-source flux
excess — over-recovery, open since step 18 and the standing "leading defect"
after the registration fix — **disappears with training depth**: 1.059 -> 0.992
on Ha and 1.086 -> 0.993 on O-III, with source counts falling from 102%/105% to
96%/97%. Both *shallow* models carry the excess (bubble-trained 1.062/1.070,
sh2-92-trained 1.059/1.086) and the deep one does not, which is two independent
confirmations from different training targets. **The halo the network was
thought to add around sources is an artefact of training on stacks shallower
than the frame it is applied to.**

**3. Training target beats depth for extended retention.** Bubble-trained at
22/29 holds more faint flux than sh2-92-trained at 43/59 on both filters
(0.380/0.384 against 0.340/0.317), on a quarter of the data. Why one nebula
generalises to ic1396 better than another is unexplained and is now the more
interesting question than depth.

The obvious next run is cheap: **pool bubble and sh2-92 at their natural
depths**, combining the target that generalises with the depth that fixes
photometry, and score it on the same ic1396 arrays.

### 25. Pooling shallow with deep gives you the shallow model's bias

2026-08-20, `n2n_depth_ab.py --train bubble,sh2-92 --caps 0`. Step 24 left two
findings pulling in opposite directions: bubble generalises to ic1396 better,
and depth fixes the photometric excess. Bubble maxes out at 22/29 frames, so
this pooled all four groups — bubble at 22/29, sh2-92 at 51/72 — hoping to get
both. Four narrowband models now exist, all scored on byte-identical ic1396
stacks.

**Ha**

| model | 1-2σ | 2-4σ | 4-8σ | 8-16σ | 16-32σ | >32σ | src | flux |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooledNB (bubble 22/29) | 0.250 | 0.314 | **0.380** | **0.530** | **0.614** | 0.992 | 101% | 1.0619 |
| cap22 (sh2-92 18/18) | 0.243 | 0.293 | 0.329 | 0.496 | 0.596 | 0.922 | 102% | 1.0593 |
| deep (sh2-92 43/59) | **0.270** | 0.305 | 0.340 | 0.482 | 0.565 | 0.916 | 96% | **0.9924** |
| pooled (both 22-72) | 0.248 | 0.289 | 0.340 | 0.493 | 0.580 | **1.007** | 103% | 1.0736 |

**O-III**

| model | 1-2σ | 2-4σ | 4-8σ | 8-16σ | 16-32σ | >32σ | src | flux |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooledNB (bubble 22/29) | 0.255 | **0.294** | **0.384** | **0.460** | **0.691** | 1.008 | 102% | 1.0704 |
| cap22 (sh2-92 18/18) | 0.222 | 0.239 | 0.281 | 0.345 | 0.536 | 0.797 | 105% | 1.0860 |
| deep (sh2-92 43/59) | **0.262** | 0.278 | 0.317 | 0.380 | 0.584 | 0.878 | 97% | **0.9930** |
| pooled (both 22-72) | 0.243 | 0.263 | 0.342 | 0.419 | 0.664 | **1.028** | 104% | 1.0904 |

**The shallowest group in the pool sets the photometric bias.** Every model
containing a group under ~30 frames reads 1.06-1.09 flux; the one deep-only
model reads 0.992-0.993. The pooled model contains the deep groups *and* the
shallow ones and comes out at **1.074/1.090 — worse than either parent**. This
is not an average: half the training pairs being shallow is enough to teach the
whole model the shallow model's over-recovery.

Mechanically that is what you would expect if the excess is learned boost.
Trained on noisy stacks the network learns to lift features out of noise;
applied to a cleaner frame that lift over-recovers. Mixing depths does not
average the lift, because the shallow pairs are the ones that reward it.

**Design rule: never pool groups whose depths differ by more than ~2x if the
output is meant to be photometric.** Equivalently, the useful depth of a pooled
training set is its *minimum*, not its mean — which also puts a caution on the
broadband ladder's `pooled-scenes` arms, whose groups spanned 6 to 31 frames
(steps 20 and 22). Their photometry was measured as fine, but they were never
compared against a deep-only broadband model, so the same effect could be
sitting in them unnoticed.

**On extended structure, pooling averaged rather than took the better parent.**
Ha stayed at 0.340 against bubble's 0.380; O-III reached 0.342, best among the
sh2-92-containing models and still under bubble's 0.384. Nothing tried has
exceeded ~0.38 at 4-8 sigma, so **ic1396 remains unrenderable by any of these
four models** and the extended-structure problem is untouched by anything in
steps 24 or 25.

The bright end is the one place pooling won: >32 sigma goes to 1.007/1.028
against deep-only's 0.916/0.878, so bright extended flux survives — now slightly
over-recovered rather than under.

#### Where this leaves the two goals

They conflict on current data:

- **Photometry** wants deep-only training, no group under ~40 frames. Only
  sh2-92 qualifies.
- **Extended structure** wants bubble, which cannot be deep — it has 59 Ha and
  77 O-III frames in total.

Two ways out, neither tested: **more bubble integration**, so the target that
generalises can also be trained deep; or an answer to **why bubble generalises
to ic1396 better than sh2-92 does** despite a quarter of the data. The second is
the more valuable question — it is about training-set composition, which no
hyperparameter sweep in this manual has touched.

### 26. Distribution match — proposed here, FALSIFIED in step 27

**Read step 27 before using anything below.** The distribution-match explanation
offered here was tested on a third narrowband field and did not survive. What
stands from this step is the *observation* — bubble-trained models beat
sh2-92-trained ones on ic1396, and it is the target rather than pooling — plus
the surface-brightness axis itself, which remains a useful way to characterise a
field. The proposed *mechanism* is wrong.

2026-08-20. Step 25 left the live question: why does bubble generalise to ic1396
better than sh2-92 does, on a quarter of the data and at half the stack depth?

**First, the effect is the target, not pooling.** Running step 18's two
*per-filter* bubble models on the same ic1396 arrays, no pooling involved:

| model | trained on | Ha 4-8σ | O-III 4-8σ |
| --- | --- | --- | --- |
| bubble per-filter | bubble | **0.397** | 0.342 |
| pooledNB | bubble | 0.380 | **0.384** |
| deep | sh2-92 | 0.340 | 0.317 |
| cap22 | sh2-92 | 0.329 | 0.281 |
| pooled | both | 0.340 | 0.342 |

Every bubble-trained model beats every sh2-92-trained model, pooled or not.
Bubble-trained spans 0.342-0.397; sh2-92-trained spans 0.281-0.340. The
separation is clean and pooling is orthogonal to it.

**The hypothesis that failed.** The obvious explanation was that bubble shows the
model more low-surface-brightness structure, so it learns that faint extended
emission is real. It is exactly backwards: sh2-92 has **more** of its area at
1-8 sigma (6.87% Ha, 7.66% O-III) than bubble (2.62%, 3.09%) and generalises
worse.

**What does predict it: how closely the training field's surface-brightness
distribution matches the target's.** Percent of frame area per smoothed-SB bin,
in units of each stack's own sky sigma:

| stack | <0σ | 0-1σ | 1-8σ |
| --- | --- | --- | --- |
| **ic1396 Ha (target)** | **41.51** | **55.45** | **2.46** |
| bubble Ha (22 fr) | 41.69 | 55.25 | 2.62 |
| sh2-92 Ha (22 fr) | 29.79 | 65.39 | 4.33 |
| sh2-92 Ha (51 fr) | 24.42 | 67.85 | 6.87 |
| **ic1396 O-III (target)** | **45.51** | **51.52** | **2.26** |
| bubble O-III (29 fr) | 45.04 | 51.26 | 3.09 |
| sh2-92 O-III (22 fr) | 36.60 | 58.49 | 4.39 |

Bubble tracks ic1396 to within 0.2 percentage points on the two bins that hold
97% of the frame. sh2-92 is 12 points off in the `<0` bin even when depth-matched
— capping it to 22 frames moves 1-8 sigma from 6.87% to 4.33%, so depth explains
part of the gap but not the part that matters. **The difference is intrinsic to
the fields.**

The mechanism this suggests: the network learns, from the distribution it is
shown, where "signal" stops and "noise" begins. Trained on a field whose
emission mostly sits above 1 sigma, it learns to treat sub-sigma structure as
noise — and then deletes exactly that on a field like ic1396, which is 55%
sub-sigma. Bubble, being 55% sub-sigma itself, teaches the opposite.

**Status: suggestive, not established.** This is two training targets and one
test target — a correlation across n=2, with a plausible mechanism. The
falsifiable version, which needs a third narrowband field to run cleanly:
**a bubble-trained model should LOSE to an sh2-92-trained model when the test
target's distribution resembles sh2-92's.** If that fails, distribution match is
not the mechanism and something else about bubble is doing the work.

If it holds, it is the most actionable finding in this manual, because it is
about **training-set composition** — an axis no experiment here has ever varied
deliberately. Every sweep so far has moved depth, loss, pair count, or pooling.
It would mean choosing training targets by the surface-brightness profile of
what you intend to denoise, and it costs nothing but selection.

### 27. Distribution match falsified — bubble is simply the better teacher

2026-08-21. Step 26 proposed that generalisation tracks how closely the training
field's surface-brightness distribution matches the target's, and stated the
falsifiable form: **a bubble-trained model should LOSE to an sh2-92-trained one
on a field whose distribution resembles sh2-92's.**

The archive drive supplied the third field this needed. `/media/taylor/cdk17`
holds five narrowband targets from 2024 that were not in `Targets/`.

**Corrected counts (2026-08-21).** The first inventory of this drive scanned
both `/media/taylor/cdk17/FromCDK17` and `/media/taylor/cdk17`, and the second
contains the first, so everything under `FromCDK17` was counted twice. Unique
files at 300 s:

| object | Ha | O-III | S-II |
| --- | --- | --- | --- |
| IC 405 | 87 | 37 | 40 |
| NGC 6888 | 83 | 61 | 50 |
| NGC 7635 | 27 | 40 | 32 |

**`bubble` in `Targets/` is NOT NGC 7635.** Corrected 2026-08-23 after the two
were treated as the same object earlier in this manual. `Targets/bubble` is at
RA 20h15m22s, l=75.6 b=+1.7 — **PN G75.5+1.7, the Soap Bubble Nebula**, in
Cygnus. The archive's NGC 7635 is at 23h20m46s, l=112.2 — the **Bubble Nebula**,
in Cassiopeia. They are 47 degrees apart and share nothing but a nickname.

Everywhere this manual says bubble "beats sh2-92 as a training target" (steps
24-27), the target is the **Soap Bubble**, from 2026 data with 59 Ha and 77
O-III and no S-II. NGC 7635 is 2024 archive data, thinner, and is the only
source of S-II.
| PK 205+14.1 | 19 | 10 | 10 |
| IC 1318 | 15 | 10 | 10 |

NGC 6888 was reported as 166/122/100 and is **not** deeper in Ha than sh2-92
(137); IC 405 is the deepest archive Ha at 87. IC 405 sits outside `FromCDK17`
and was never doubled, so its figures were always right.

NGC 6888's O-III channel lands on the sh2-92 side of the axis, which is what the
test required (stacked uncalibrated, 22 frames — see the caveat below):

| field | <0σ | 0-1σ | >1σ |
| --- | --- | --- | --- |
| ngc6888 O-III | **34.5** | 59.2 | 6.27 |
| sh2-92 O-III | 36.6 | 58.5 | 4.91 |
| bubble O-III | 45.0 | 51.3 | 3.70 |
| ic1396 O-III | 45.5 | 51.5 | 2.97 |

**The prediction failed.** Extended-flux retention at 4-8 sigma:

| model | trained on | ic1396 O-III | ngc6888 O-III |
| --- | --- | --- | --- |
| bubble per-filter | bubble | 0.342 | 0.388 |
| pooledNB | bubble | **0.384** | **0.459** |
| cap22 | sh2-92 | 0.281 | 0.342 |
| deep | sh2-92 | 0.317 | 0.379 |

Bubble-trained wins on **both** fields, regardless of which one the test field
resembles. Distribution match is not the mechanism.

**What is ruled out for bubble's advantage:** data volume (bubble has a
quarter), stack depth (bubble is shallower — 22/29 against 51/72), pooling
(bubble's *per-filter* models win too), and now distribution match.

**What replicated on a field neither model had seen:** pooling Ha with O-III
beats per-filter, 0.459 against 0.388. That is step 19 confirmed on new data.

**The remaining lead, unproven.** At matched depth bubble's stacks carry 2-7x
more total structure power in sky-sigma units than sh2-92's, and half the
fine-scale (2-8 px) power — the band dominated by noise. Both peak at the same
8-32 px scale, so it is not a difference in the *scale* of structure but in how
much structure there is relative to noise. Whether that is seeing, guiding, sky,
or the object is untested, and no further hypothesis is being built on a single
measurement after this one.

**Caveats on this run.** The NGC 6888 stacks are **uncalibrated** — no epoch-
matched flats exist for 2024, though 2024-09 darks and bias at -20C do. Absolute
numbers are therefore not comparable to the calibrated ic1396 column; the
model-to-model comparison within the ngc6888 column is fair, because all four
models saw the identical array. Registration plus sigma clipping rejects most
fixed-pattern artefacts on a dithered set, which is what makes the shape
measurement usable at all, but nothing here should be trained on.

**Trap recorded: the 2024 archive labels O-III as `O2`.** `index_frames` matches
FILTER literally and `color_process._ALIASES` lists only OIII/O3/O-III/OXYGEN3,
so all 122 NGC 6888 O-III frames are invisible to every existing path — an HOO
process would quietly render a one-channel image. `Sii` is fine (uppercases into
the S-II aliases). Anything reading the archive must handle it.

### 28. The noise floor, and a re-audit of everything measured on 2026-08-20/21

2026-08-21. Six hypotheses were argued that day from differences of 0.01-0.07 in
extended retention, with no error bars and no decision rule. This step measures
what a repeat of an identical run produces, and re-reads every claim against it.

#### The instrument was wrong twice before it was right

1. **No uncertainty at all.** Every retention number reported before this step is
   a point estimate with nothing attached.
2. **Bootstrapping over pixels fakes precision.** A retention bin can hold 10^6
   pixels and still be poorly determined, because they sit in a few spatially
   correlated patches. On ic1396 O-III only **54 tiles** of 256 px carry enough
   4-8 sigma emission, and **140** carry 1-2 sigma. The honest n is tens, not
   millions.
3. **Unpaired intervals are the wrong test.** Unpaired 95% CIs came out at
   +/-0.05 and declared every comparison indistinguishable. Both models see the
   same field, so tile-to-tile variation in how much structure a tile holds is
   common to both and must be cancelled. The paired test on identical data
   resolves 0.02 cleanly.

`scripts/n2n_compare_paired.py` is the resulting instrument: per-tile retention,
paired differences, block bootstrap over tiles.

#### The noise floor

Three trainings of one configuration (sh2-92 deep, L2, 60 epochs, same cached
stacks, `--train-seed 0/1/2` so only patch sampling and weight init differ):

| | 1-2σ | 4-8σ | Ha flux | O-III flux |
| --- | --- | --- | --- | --- |
| seed 0 | 0.270 | 0.328 | 0.9924 | 0.9930 |
| seed 1 | 0.261 | 0.321 | 1.0147 | 1.0227 |
| seed 2 | 0.235 | 0.307 | 1.0022 | 1.0059 |
| **paired vs seed 0** | **up to -0.021** | **up to -0.012** | | |

**Minimum detectable effect: ~0.02 at 1-2 sigma, ~0.012 at 4-8 sigma, ~0.03 on
aperture flux.** Anything smaller is a seed.

#### Re-audit

| claim | effect | vs floor | verdict |
| --- | --- | --- | --- |
| Denoiser destroys faint extended structure | 0.25-0.36 against ideal 1.0 | ~30x | **stands** |
| bubble beats sh2-92 at 4-8σ | +0.044 | 3.7x | **stands** |
| Pooling Ha+O-III beats per-filter | +0.042 and +0.071, two fields | 3-6x | **stands** |
| Deep training removes the flux excess | 1.06 -> 1.01 | 2x | **marginal, replicated** |
| `source_bias` 0.7 helps | +0.018 to +0.025 | 1.0-1.2x | **not established** |
| Deeper training improves retention | +0.011 to +0.036 | within floor | **retracted** |
| Any target effect at 1-2σ | ~0.004 | floor is 5x larger | **retracted** |

Two corrections to what steps 24-27 assert:

- **Step 24's retention claim is retracted.** Its photometry claim (depth removes
  the flux excess) survives at 2x the floor and is supported by three shallow
  models from two targets against three deep seeds, but it is not the clean
  result it was written as.
- **The `source_bias` result claimed on 2026-08-21 is withdrawn.** It was called
  confirmed, closing step 7's open question. Its effect is the same size as the
  seed-2 difference. Step 7 stays open.

**bubble beating sh2-92 survives at 3.7x the floor and is robust to
`source_bias`** — +0.044 at 0.7 and +0.047 at 0.0 — so whatever bubble does, it
does not act through patch sampling. It remains unexplained, with data volume,
stack depth, pooling, seeing (sh2-92 is *sharper*, 1.59" against 1.90"),
distribution match, star density, and now patch sampling all excluded.

#### Standing rule

**No A/B on extended structure is reportable without a seed control.** One
configuration trained at two or more seeds, compared with the same paired test,
establishes the floor for that dataset; only effects clearly above it count. The
cost is one extra training run per experiment, against a day spent chasing six
explanations for differences that were mostly noise.

### 29. The denoiser is band-limited, not blind — measured transfer function

2026-08-21. Step 23 concluded the denoiser "deletes the nebula" from a retention
number and a picture. Injection with known ground truth says something more
specific, and more useful.

**Injection is the only measurement here with ground truth.** Everything else
compares denoised against raw, which cannot separate "the model failed" from
"the information was not there". Adding a known structure and differencing two
denoiser runs — `denoise(base + truth) - denoise(base)` — cancels the underlying
field and both background models, leaving the model's response to a signal whose
shape is known exactly.

#### The transfer function

`scripts/n2n_fractal_injection.py`, injecting a band-limited Gaussian random
field with P(k) ~ k^-3 at 1 sigma RMS into ic1396 O-III, denoised with pooledNB:

| scale (px/cycle) | T(k) |
| --- | --- |
| 1024-4096 | 0.984 |
| 512-1024 | 0.962 |
| 256-512 | 0.875 |
| 128-256 | 0.720 |
| 64-128 | 0.615 |
| 32-64 | 0.545 |
| 8-16 | 0.495 |
| 4-8 | 0.323 |

**Half-power point around 100-150 px.** The model preserves large-scale emission
almost perfectly and progressively discards finer structure.

This is corroborated by an independent experiment. `n2n_extended_injection.py`
injects Gaussian blobs and the transfer function predicts its results
quantitatively: sigma=100 blob (dominant scale ~400 px) predicted ~0.9, measured
0.92; sigma=30 (~120 px) predicted ~0.72, measured 0.79; sigma=10 (~40 px)
predicted ~0.55, measured 0.63. Two independent injections agreeing to a few
percent is the strongest internal consistency any measurement in this manual has
shown.

#### What this rules in and out

- **Not an information limit.** A 0.5 sigma-peak, 100 px structure carries
  integrated SNR ~89 and is recovered at 0.924. The photons are there and the
  model uses them. The "more integration is the only fix" reading, offered
  earlier the same day, is wrong.
- **Not a broken metric.** `n2n_extended_check`'s retention, applied to the
  *injected* field where truth is known, reads 0.93-0.96. It is unbiased on
  large-scale-dominated input.
- **Not background subtraction.** Raw and denoised background models differ by
  0.0126 ADU, 1.2% of sky sigma; sharing one model changes retention the wrong
  way.
- **Point sources are unaffected**: radial profile preserved to 0.1% at the core
  and 0.8% at r=15 px.

#### The gap that remains

Real ic1396 emission reads 0.384 where the transfer function alone predicts
better. An attempt to close it by decomposing the real field into spatial
octaves gave an implausible 10.7 sigma RMS at 32-64 px against a field whose p99
is 3.4 sigma — bright star halos leaking past a mask covering only 1.6% of
pixels. **That decomposition is not to be trusted and the gap is open.** Either
real nebular morphology differs from a power-law field in a way that matters, or
the model's response to *pre-existing* structure differs from its response to an
*added* perturbation — injection measures the marginal response, retention the
absolute one, and for a nonlinear operator those need not agree.

#### Consequence

Denoised output is band-limited, and the band it discards is exactly the
resolution-limit detail that makes a nebula look like a nebula (seeing here is
FWHM ~7 px). That is an architecture property, not a photon-count problem: a
4-level U-Net on 256 px patches has a receptive field of the same order as the
measured half-power point, which is the first thing to check. Widening the
receptive field, or weighting the loss toward fine scales, are the levers this
finding points at — none of which is a hyperparameter already swept.

Step 23's "deletes the nebula" is too strong and is qualified here: large-scale
emission survives at 96-98%, fine texture at 30-50%.

### 32. The Wiener bound: the fine scales were never the defect — the mid scales are

2026-08-23, `scripts/n2n_wiener_bound.py`. Phase 0a of the improvement plan.
Step 29 measured the transfer function falling to ~0.33 at 4-8 px and the work
since has treated that fine-scale loss as the thing to fix. Nobody had checked
what an *optimal* estimator would do. For a scene with signal spectrum S(k)
under noise N(k), the linear-MMSE filter passes

    T*(k) = S(k) / (S(k) + N(k))

and an additive perturbation — exactly what the fractal injection adds — is
passed with that gain.

**Both spectra come from a half-stack pair with no model in the loop**: the
cross-spectrum Re(FA·conj(FB)) gives S(k) because independent noise cancels in
the cross term, and P[A-B]/2 gives N(k) because the signal cancels — the
correlation-ceiling trick, taken per frequency band. Consistency check: S+N
reproduces the measured power of half A to 1-9% in every band. Point sources
are masked (radius capped at 25 px so the nebula itself survives the mask —
the uncapped version masked the filaments and ran 20 minutes before that was
caught) and any residual star wings inflate S(k), i.e. bias the verdict toward
"headroom", never toward "optimal". Measured T(k) is taken by injecting into
the same half-stack the bound is computed for, so bound and measurement see the
same noise.

ngc6888, both channels, pooledNB and the p512 model:

| band (px) | O-III S/N | O-III T* | O-III p512 | Ha S/N | Ha T* | Ha p512 |
| --- | --- | --- | --- | --- | --- | --- |
| 256-512 | 870 | 0.999 | 0.931 | 200 | 0.995 | 0.931 |
| 128-256 | 781 | 0.999 | 0.871 | 174 | 0.994 | 0.871 |
| 64-128 | 83 | 0.988 | 0.859 | 248 | 0.996 | 0.849 |
| 32-64 | 107 | 0.991 | 0.788 | 157 | 0.994 | 0.742 |
| 16-32 | 17.5 | 0.946 | 0.727 | 67 | 0.985 | 0.718 |
| 8-16 | 2.7 | 0.731 | **0.738** | 9.7 | 0.906 | 0.748 |
| 4-8 | 0.36 | 0.263 | **0.388** | 0.83 | 0.452 | 0.397 |

**Three findings, one per region:**

1. **The 4-8 px band is closed.** S/N there is 0.36-0.83, the bound is
   0.26-0.45, and the measured 0.36-0.40 sits at or *above* it — a nonlinear,
   spatially adaptive net beating a global linear filter, which it is allowed
   to do. The "notorious 0.33" of steps 29-30 is optimal estimation, not a
   defect. Nothing but photons wins there, and the manual stops chasing it.
2. **8-16 px is at the bound for p512** on O-III (0.738 vs 0.731) and has
   modest headroom on Ha (+0.16), tracking that channel's higher S/N.
3. **16-256 px is the actual defect.** Those bands are signal-dominated —
   S/N 17 to 870 — so the bound sits at 0.95-0.999, and the model passes
   0.72-0.93. Headroom **+0.07 to +0.27**, largest at 16-64 px, on both
   channels. The model is optimal at the hard scales and is discarding
   strongly-detected signal at the easy ones — the signature of an
   over-regularised prior, and precisely what a mid-band-weighted loss should
   attack.

**A mechanism for colour non-safety falls out for free.** T*(k) scales with
channel S/N, so an optimal estimator retains more of a brighter channel. Part
of the Ha >> S-II retention gap (steps 21, 31) is therefore the physics of the
bound, not model error — consistent with the self-trained upper bound (31)
failing to close it.

Caveats: one field, two channels, one input depth (a 23/31-frame half-stack);
the bound is the *global linear* MMSE, which a spatially adaptive model may
locally exceed (and does, below 10 px). None of these move the mid-band
verdict: even a tenfold star-wing inflation of S(k) leaves S/N >= 10 and the
bound above 0.9 in every 16-256 px band.

**Consequence for the improvement plan:** Phase 2 proceeds, retargeted. The
objective is the 16-256 px bands; the per-band headroom above is the loss
weighting; 4-16 px is closed and any future claim of improving it must first
beat this bound.

### 33. Phase 1 measured: the tiling weave dies, the epoch cut does not survive

2026-08-23, the improvement plan's cheap-wins phase. One intervention promoted,
one retracted — the retraction being a correction to step 31.

#### Crop stitching replaces the Tukey blend (promoted; now the default)

`denoise_frame` gained `stitch="crop"`: each tile runs at full size but only its
central `tile_size - 2*overlap` region is kept, so every output pixel is
computed exactly once, from a tile in which it has >= 64 px of real context.
The frame is reflection-padded so borders get context too. No blending means no
two-tile disagreement to average — which step 20 identified as the weave.

Measured by translation-consistency (`scripts/n2n_tile_artifact.py`: denoise
twice with the grid shifted 192 px — a multiple of neither stitching period —
and difference the interiors; any difference is pure tile-placement
dependence), on ic1396 O-III with pooledNB:

|  | blend | crop |
| --- | --- | --- |
| placement-dependence rms | 0.0043 sigma | **0.0013 sigma** (0.31x) |
| p99.9 | 0.033 sigma | **0.011 sigma** (0.35x) |
| fractal-injection recovery | 0.872 | 0.871 |

Recovery identical to three decimals — a pure stitching win. The blend's
artefact power sat exactly where step 20 said: 54% in the 32-96 px bands (the
64 px overlap) and 25% at 256-768 px (the tile scale). The weave was visible
despite its tiny rms because it is coherent and periodic; 3x down should put it
under the sky grain. Cost: ~36% more tiles, ~13 s per frame. `"crop"` is now
the default; `"blend"` is kept for comparison.

#### Epochs cannot be cut to 20 (retracted; corrects step 31)

Step 31 observed both logged loss curves plateauing by epoch 7-12 and suggested
`epochs: 60` should be ~20. Tested properly — one 20-epoch run (cosine
`T_max=20`, so a genuinely rescheduled run, not a truncation) on the sh2-92
deep p512 config, against the three existing 60-epoch seeds:

| metric | 60-epoch seeds (3) | 20-epoch |
| --- | --- | --- |
| best val | 0.57713-0.57909 | 0.57942 |
| retention Ha 4-8 sigma | 0.327-0.363 | 0.336 |
| retention O-III 4-8 sigma | 0.302-0.336 | 0.314 |
| **T(k) 128-256 px** | **0.848-0.888** | **0.826** |
| **T(k) 64-128 px** | **0.839-0.859** | **0.777** |
| T(k) 32-64 px | 0.696-0.804 | 0.702 |

Val loss and retention are indistinguishable — and the mid-band transfer, the
exact band step 32 identified as the real defect, is ~0.05-0.06 **below the
seed range**, three times the seeds' spread at 64-128 px. One seed, so recorded
with that caveat, but the size and the location leave little room: **the L2
plateau hides the transfer function still improving between epochs 20 and 60.**

Two consequences. `epochs` stays at 60, at least for Phase-2 work whose whole
objective is mid-band T(k). And "loss is not evidence" (standing conclusion 2)
gains a sharper form: **even a flat loss can conceal a metric that is still
moving.** Judged on the loss curve alone, the epoch cut would have shipped —
which is precisely why the improvement plan required judging it on retention
and transfer instead.

### 34. Headroom-weighted loss refuted — and a one-seed story corrected mid-flight

2026-08-23, Phase 2a. Step 32 measured +0.07-0.27 of claimable mid-band signal
and the obvious lever was to spend the loss there: `nn/losses.py`, a
Laplacian-pyramid L2 on the residual with octave weights proportional to the
measured headroom (peak 1.7x at 16-64 px, mean-normalised so lr and grad-clip
transfer). Two seeds on the sh2-92 deep p512 config, judged by T(k) against the
three plain-L2 baselines, everything measured in one run under crop stitching.

| band (px) | baseline, 3 seeds | msl2, 2 seeds | |
| --- | --- | --- | --- |
| 256-512 | 0.920-0.941 | 0.903-0.943 | overlap |
| 128-256 | 0.852-0.892 | 0.788-0.883 | overlap |
| 64-128 | 0.846-0.865 | 0.718-0.849 | overlap |
| 32-64 | 0.702-0.811 | 0.633-0.729 | overlap |
| 8-16 | 0.602-0.661 | 0.486-0.600 | **msl2 worse, separated** |
| 4-8 | 0.329-0.337 | 0.320-0.345 | overlap |

**Refuted, on three counts**: no band where msl2's worst seed beats the
baseline's best; one band separated worse; and a seed-to-seed spread of ~0.13
at 64-128 px against the baseline's 0.02 — the weighting destabilises training
without buying anything. Photometry also wobbled (flux 0.995 and 1.016 across
the two seeds, straddling the floor).

**A correction to the interim read, recorded because it is the protocol's whole
point.** Seed 0 alone showed deficits of 0.08-0.13 in the mid band and was
reported as "worse across the board, 5-10x the seed spread" — implicitly using
the *baseline's* spread. Seed 1 landed 0.13 higher in those same bands: the
stark picture was substantially msl2's own, much larger, variance. One seed of
the new arm told a false story in both magnitude and mechanism. The two-seed
rule (step 28) is what caught it.

Why a diagonal band weighting was never going to shift the optimum is clear in
hindsight: for any fixed positive weights the minimiser of a weighted L2
against a noisy target is still the conditional mean, band by band. The
weighting can only reallocate finite capacity — and the measurement says the
plain-L2 model was not capacity-starved in the mid band, because paying more
for mid-band error changed nothing except stability. **The mid-band headroom is
therefore not reachable through the loss**, which narrows the remaining
explanation to architecture: the receptive field. Phase 2b (a fifth U-Net
level, ~2x the receptive field, 10.97M against 7.76M parameters) is training as
this is written, two seeds, same protocol.

Also in this phase, the confound hygiene that made the comparison valid at all:
baselines re-measured under crop stitching before judging msl2 against them
(crop shifts T(k) by <=0.006 per band vs blend), since crop became the default
mid-experiment.

### 35. The fifth level wins at one amplitude — and the amplitude was the blind spot

2026-08-24. Phase 2b, and the deferred Phase 0c that turned out to govern it.

**2b at the standard amplitude: the first separated win of the campaign.** A
fifth U-Net level (features 32,64,128,256,256, 10.97M params, ~2x receptive
field; checkpoints now carry `features` and every loader rebuilds from them).
Two seeds against the three 4-level baselines, fractal injection at 1 sigma RMS:

| band (px) | baseline, 3 seeds | 5-level, 2 seeds | |
| --- | --- | --- | --- |
| 64-128 | 0.846-0.865 | **0.897-0.900** | better, separated |
| 32-64 | 0.702-0.811 | **0.859-0.860** | better, separated |
| 8-16 | 0.602-0.661 | 0.470-0.658 | overlap, l5 spread 0.19 |

The mid-band gain is tight across seeds (0.001-0.003 spread) — systematic, not
luck — and claims about half the gap to the Wiener bound. The apparent 8-16 px
regression from seed 0 alone did not survive seed 1 (one-seed trap, again, in
the opposite direction from msl2's). Photometry clean (flux 0.9996-1.0120).

**But retention on real ic1396 emission did not move** (Ha 0.314-0.316 against
baselines 0.327-0.363 at 4-8 sigma — at the floor, slightly the wrong side).
The transfer gain refused to appear in the display metric, which is the step-29
marginal-vs-absolute gap demanding to be measured. That is Phase 0c, listed in
the plan's analysis phase and **deferred — a sequencing error, recorded as
such**, because it turns out to govern everything Phase 2 did:

**0c: the transfer function is strongly amplitude-dependent.** Injecting the
same field at three amplitudes, 64-128 px band:

| injected RMS | base_s0 | l5_s0 |
| --- | --- | --- |
| 0.25 sigma | 0.621 | **0.540** |
| 1.0 sigma | 0.846 | **0.897** |
| 2.0 sigma | 0.863 | 0.829 |

Swings of +-0.25 per band, and opposite trends in different bands (base 8-16
goes 0.896 -> 0.661 -> 0.580 as amplitude rises while its mid bands go up). So:

1. **T(k) at a single amplitude is an operating-point measurement, not a filter
   property.** Every transfer number in steps 29-35 carries an implicit "at
   1 sigma RMS" that was never stated until now.
2. **The fifth level's win is amplitude-local.** At 0.25 sigma — where faint
   real nebulosity lives — it is *worse* than the baseline in the bands it was
   built to improve. That is why retention never moved: the gain exists at an
   amplitude real faint emission does not have.
3. The step-29 gap (real emission at 0.384 where injection predicted better) is
   explained in kind: the injection was probing a more favourable operating
   point than the scene occupies.

**Phase 2 closes with no promotion.** Both levers ran: the loss cannot reach
the headroom (34), and the architecture reaches it only at one amplitude (35).
The receptive-field mechanism is proven — the mid-band ceiling does move with
it — but a model that helps bright structure and hurts faint structure in the
same bands fails the plan's no-regression gate outright.

What a future attempt needs, stated for whoever picks this up: the objective
must be an **amplitude-swept** transfer surface T(k, A), not a slice of it; and
the Wiener comparison of step 32 should be redone per amplitude, since the
bound comparison inherits the same caveat. Until then production stays as it
is, and the honest levers remain the ones steps 24-31 established: integration
depth and photons.

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

### Colour safety is worst where channel depth is unequal (NGC 6888, SHO)

2026-08-21, noticed by eye in the first SHO render — "the S-II nebulosity is
somewhat removed" — then measured. Retention per channel on NGC 6888 with
pooledNB:

| SB bin | S-II | Ha | O-III | spread |
| --- | --- | --- | --- | --- |
| 2-4σ | 0.340 | 0.523 | 0.422 | 0.184 |
| 4-8σ | 0.439 | 0.664 | 0.492 | 0.225 |
| 8-16σ | 0.556 | **0.887** | 0.606 | **0.331** |
| 16-32σ | 0.695 | 0.915 | 0.751 | 0.219 |
| >32σ | 1.009 | 1.021 | 1.037 | 0.028 |

S-II is the worst-retained channel in every bin below 32 sigma and Ha the best,
so in SHO the *red* channel loses a third more than the *green* — a visible
green shift. Largest cross-channel spread recorded here; step 21's ngc5907 case
was 0.15.

**It is depth and brightness, not the filter.** S-II is disadvantaged twice:

| | frames kept | sky sigma | nebula signal | in sigma |
| --- | --- | --- | --- | --- |
| Ha | 74 of 83 | 0.734 | 4.20 ADU | 5.7 |
| S-II | 39 of 50 | 0.948 | 2.62 ADU | 2.8 |

Half Ha's structure SNR, and retention tracks structure SNR (step 29). Bringing
S-II from 50 frames to ~85 would put its noise near Ha's and close most of the
gap — the fix is integration time, not the model.

**Consequence for SHO:** do not read ionisation structure off a denoised
composite. The S-II/Ha ratio is altered by up to 40% in the mid-brightness
range. The raw composite is unaffected, which is the reason the routine path
writes both.

### 31. Five narrowband variants, one conclusion: the training set is not the limit

2026-08-22/23. Five models were trained to try to improve narrowband denoising,
all scored on the same held-out NGC 6888 stacks. Recording them together
because the individually-null results are the point, and each is cheap to
repeat by accident.

| model | trained on | patch | ch | Ha | O-III | S-II | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `n2n_pooledNB_300s` (production) | Soap Bubble | 256 | 1 | 0.887 | **0.606** | **0.556** | **0.331** |
| `n2n_sc_sho7635` | NGC 7635 | 512 | 1 | 0.893 | 0.513 | 0.506 | 0.387 |
| `n2n_mc3_sho7635` | NGC 7635 | 512 | 3 | 0.812 | 0.649 | 0.392 | 0.419 |
| `n2n_nb_soap_7635` | both | 512 | 1 | 0.941 | 0.579 | 0.510 | 0.431 |
| `n2n_pooledNB_p512` | Soap Bubble | 512 | 1 | 0.890 | 0.585 | 0.534 | 0.356 |
| `n2n_self6888_p512` | **NGC 6888 itself** | 512 | 1 | **0.954** | 0.602 | 0.572 | 0.382 |

(retention at 8-16 sigma; spread is the colour-safety number)

**Nothing beat production on the channels that matter.** Every variant improved
Ha and left O-III and S-II where they were, so every one made cross-channel
spread *worse*. `pooledNB`, trained at patch 256 on two groups and never having
seen an S-II frame, is still the best narrowband model available.

#### The upper bound settles the S-II question

`n2n_self6888_p512` was trained on NGC 6888's own split stacks and then applied
to NGC 6888 — not a held-out test, deliberately. Self-training removes
generalisation error entirely, so its retention is approximately **the best this
architecture can do on this data**, and it bounds what any training-set choice
could achieve.

It reaches Ha **0.954** and leaves O-III at 0.602 and S-II at 0.572 — within
noise of production. So the S-II deficit is **not generalisation error**. No
training target, no depth, no patch size and no channel count closes it. It is
structure-SNR limited, exactly as the convergence curves implied (NGC 6888 S-II
converges to 15.3% against Ha's 10.4%), and **integration time is the only
remaining lever**.

#### The models are nearly interchangeable

Measured on the ngc6888 Ha stack, in units of its 0.734 ADU sky noise:

    model vs model      rms 0.307 - 0.585 sigma
    model vs raw        rms 0.896 - 1.114 sigma

Every model moves the frame ~1 sigma from the raw and differs from its siblings
by a third to a half of that, in scattered directions rather than coherently.
That is why the rendered images are visually indistinguishable — noticed by eye
first, then confirmed. **Model choice makes ~10% of the decision; the shared
band limit makes the other 90%.**

Practical consequence: stop training narrowband variants hoping for a visible
gain. Depth and integration are the levers.

#### 60 epochs is roughly 3x more than needed

The first two runs ever to log per-epoch curves (see below) both plateau early.
On `self6888_p512`, val is within 0.002 of its best **by epoch 7**; epochs 7-60
changed val by +0.003 and cost ~106 minutes of the 2-hour run. On
`pooledNB_p512` the plateau is by epoch ~12.

This also retires a claim made repeatedly in this manual. "Best epoch" has been
quoted as a signal — 13, 15, 19, 23, 25, 26, 36 — and step 22 read a story into
it ("the pooled model keeps learning while single-pair models peak early,
evidence for thinness"). Those are **tie-breaks on a flat noisy plateau**, and
the differences are smaller than the seed floor. Do not cite best-epoch as
evidence.

`epochs: 60` should probably be ~20, which would make every future A/B three
times cheaper. Not changed yet: it deserves one run at 20 against one at 60,
compared on retention rather than on loss.

#### Nothing logged curves until 2026-08-23

Only `nn.trainer.train` ever wrote TensorBoard scalars. Every script written
during the ladder, depth, multichannel and patch-size work used its own training
loop and logged nothing but every-tenth-epoch console lines — so a week of runs
left six sampled points each and no curves. `n2n_pooled_nb.py` now writes
per-epoch `loss/train`, `loss/val`, `loss_best` and `lr` to
`local/runs/{tag}_{exptime}s`; the other trainers still do not. View with
`tensorboard --logdir local/runs`.

The plateau finding above was invisible for a week purely because of this.

## The routine path: producing a night's images

`python scripts/n2n_lrgb_render.py routine --dso <name> --recipe <LRGB|HOO|SHO>`

Stacks each filter onto one shared reference, writes a mono JPEG per channel,
denoises, writes those channels again, and composes both a raw and a denoised
colour image — eight files for a three-filter recipe, into
`<image_dir>/Iris/<dso>/`. No training and no held-out scoring: this is the path
for a target that already has a suitable model.

The model is chosen by **domain**, narrowband or broadband, from the filters the
recipe needs (step 23: the narrowband-trained model beat the broadband one in
every bin on both filters when applied to narrowband). `--model` overrides it.
A missing model degrades to raw products rather than failing.

**One stretch, taken from the raw, is applied to every output** — both composites
and all channel JPEGs. `compose` would instead pick black per channel by
percentile, which lands at a different ADU on a denoised channel and
manufactures a colour shift out of nothing. Both composites are always written,
so the pair can be compared rather than the denoised one quietly replacing the
raw.

### When the denoised version is the better product

Not by object class. **By surface brightness.** NGC 6888 is a nebula and its
denoised SHO is arguably the better image — the filaments are bright, well above
the noise, and survive; the background comes out cleaner and the faint outer
patches become *more* visible. ic1396 is also a nebula and its denoised render
is ruined, because its emission sits at 1-5 sigma, in the band the model
discards (step 29).

The discriminator is measurable before rendering, with
`scripts/n2n_sb_profile.py`: the fraction of frame area above 1 sigma. ic1396
O-III is 2.26%, NGC 6888 Ha 5.78%. Structure that clears a few sigma survives;
structure at 1-2 sigma does not, whatever it is attached to.

So: **run both, look at both, publish the one that is actually better.** That is
why the routine path writes the pair rather than choosing.

### 30. Training patch size — the band limit is partly a training artefact

2026-08-22, prompted by the question of why the training patch (256) and the
inference tile (512) differ at all. Neither constant records a reason, and the
mismatch turns out to matter.

The receptive field of this 4-level U-Net with two 3x3 convs per block is
~140 px. So the share of a patch whose pixels see their full surroundings is

    256 px patch:  21%
    512 px patch:  53%

which means the network was trained mostly on edge-starved pixels and applied
mostly to well-contexted ones — a train/inference distribution gap nobody chose,
falling out of two independently picked constants. The same shape of mistake as
the shallow-training flux excess of step 24.

Tested by training sh2-92 deep at patch 512, **three seeds per arm**, and
measuring the transfer function with `n2n_fractal_injection.py`:

| band (px/cycle) | patch 256, 3 seeds | patch 512, 3 seeds | |
| --- | --- | --- | --- |
| 1024-4096 | 0.991-0.993 | 0.990-0.993 | overlap |
| 512-1024 | 0.976-0.983 | 0.975-0.981 | overlap |
| 256-512 | 0.915-0.939 | 0.920-0.942 | overlap |
| **128-256** | **0.785-0.845** | **0.848-0.888** | **separated** |
| **64-128** | **0.690-0.777** | **0.839-0.859** | **separated** |
| **32-64** | **0.605-0.672** | **0.696-0.804** | **separated** |
| 16-32 | 0.581-0.700 | 0.619-0.667 | overlap |
| 8-16 | 0.508-0.692 | 0.593-0.642 | overlap |
| 4-8 | 0.308-0.364 | 0.325-0.335 | overlap |

**Three adjacent bands separate with no overlap between the arms' ranges**, and
they are exactly the bands near the receptive field. Everything coarser was
already ~0.99 and cannot improve; everything finer is noise-limited, where
context cannot help. Global amplitude is unchanged (0.918-0.931 against
0.908-0.928). That pattern is what the hypothesis predicts and is hard to get
from a lucky seed.

**The half-power point moves from ~100 px to ~40 px.** `patch_size` is now 512 in
`configs/config_public.py`.

What it costs: ~4x the training wall time at unchanged `pairs_per_epoch` — 31 min
becomes ~2 h — because a 512 patch is four times the pixels. `batch_size` 8 fits
without OOM on the GB10.

What it does **not** fix: 4-8 px is still 0.33, so ic1396's faintest structure
still does not survive. Better, not solved.

**Consequence for existing checkpoints.** Every model in `local/models/` was
trained at 256, including the two in production use (`n2n_pooledNB_300s.pt`,
`n2n_ladder_pooled-filters_300s.pt`). They are still valid — patch size is a
training-time choice and the network is fully convolutional — but they no longer
match the config default, and any A/B against a freshly trained model now
carries patch size as a second variable. Retraining both at 512 is the obvious
follow-up and has not been done.

## Choosing training data

Guidance, distilled from steps 18-29. Everything here is measured; the section
exists because the evidence is otherwise spread across a dozen chronology
entries and the wrong criteria are more intuitive than the right ones.

### Select on these

**1. Depth: aim for >=40 frames per half-stack, and never pool groups whose
depths differ by more than ~2x.**

The only criterion with a mechanism behind it. Shrinkage is calibrated to the
noise level the model trained on, so a shallow-trained model over-recovers when
applied to anything cleaner — the "halo around sources" that was this manual's
leading defect for a week is exactly that (step 24). Deep-only training moved
aperture flux from 1.06-1.09 to 0.992-0.993.

Mixing depths does not average, it takes the worse (step 25): bubble at 22/29
pooled with sh2-92 at 51/72 gave 1.074/1.090 — **worse than either parent**. The
useful depth of a pooled training set is its *minimum*.

Budget for the quality gate, which rejects 12-46% depending on filter and
target. Nominal 51 became 44 and 42 in practice; nominal 12 became 6 and 7.

**2. Match the domain: narrowband models for narrowband, broadband for
broadband.**

On narrowband ic1396 the narrowband-trained model beat the broadband-trained one
in every bin on both filters (step 23), 0.380/0.384 against 0.340/0.317. Two
models. Whether one model over all seven filters would work is untested, and
`n2n_train_pooled.py`'s argument that narrowband sky shifts the noise regime
from sky-dominated toward read/dark-dominated still stands unrefuted.

**3. Pool the filters of a target; about four groups is enough.**

Pooling beats per-filter on four held-out targets across both domains (steps 19,
20, 23, 27). Going from 4 groups to 14 bought nothing once per-pair sampling was
matched (step 22). Pairs are always formed within `(dso, filter)` — pooling
shares *weights*, never pairs, because an Ha stack and an O-III stack of one
target are two different scenes.

### Do not select on these

Each sounds reasonable and each was measured to be irrelevant or backwards:

| criterion | what the measurement said |
| --- | --- |
| data volume | bubble has a quarter of sh2-92's frames and trains a better model |
| seeing | sh2-92 is *sharper* — 1.59" against 1.90" — and trains worse |
| surface-brightness similarity to the target | proposed in step 26, falsified in step 27 on a third field |
| amount of faint structure in the training field | sh2-92 has 2.6x more area at 1-8 sigma and teaches worse |
| star density | excluded — the effect survives with `source_bias` off |
| scene count | no gain from 4 groups to 14 at matched sampling |

### The gap

**Bubble beats sh2-92 by +0.044 at 4-8 sigma (3.7x the seed floor) and nothing
explains it** after excluding all six above plus stack depth and patch sampling.
There is a real "this field teaches better" axis that cannot currently be named,
which means training data cannot be chosen by inspection.

### So measure it — about an hour per candidate

1. Build split stacks and **assert every training pair is (0,0) px**. A
   misaligned pair teaches the network to delete point sources (step 18).
2. Train pooled across that target's filters, L2, deep.
3. Score on a **held-out** target with `scripts/n2n_extended_check.py`, paired
   against the current best with `scripts/n2n_compare_paired.py`.
4. Require the difference to clear the **seed floor** — ~0.02 retention at 1-2
   sigma, ~0.012 at 4-8, ~0.03 on flux. One replicate at a different
   `--train-seed` establishes it for a new dataset (step 28).

Step 4 is not optional. Six experiments on 2026-08-20/21 were argued from
differences that turned out to be inside the floor.

## Standing conclusions

0. **The training set is not the limit; integration is.** Six narrowband models
   across four training targets, two patch sizes and one to three channels land
   within 0.3-0.6 sigma of each other on the same frame, against ~1.0 sigma of
   total effect — visually indistinguishable (step 31). A model trained on the
   target itself, which bounds what any training choice can reach, leaves S-II
   and O-III exactly where production has them. Spend nights, not epochs.
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
2b. **A difference is not a result until it clears the seed floor.** Measured
   2026-08-21 on ic1396 O-III: ~0.02 retention at 1-2 sigma, ~0.012 at 4-8, and
   ~0.03 on aperture flux, from three trainings of one configuration. Effective
   n is independent *structures* (54 tiles at 4-8 sigma), not pixels. Use
   `scripts/n2n_compare_paired.py`, and run a seed control in every A/B.
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
- **Over-recovery at mid SNR — LARGELY EXPLAINED by step 24.** It tracks
  training depth: models trained on 22-29 frame half-stacks read 1.06-1.09 on
  held-out ic1396, and the same architecture trained at 43-59 frames reads
  0.992-0.993. Deep-train before treating any residual excess as real.
  Original entry follows.
- **Over-recovery at mid SNR.** Survived the fix and was long the leading defect.
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
- **Source-bias** (step 7), still unconfirmed. A 2026-08-21 result claimed it
  helped (+0.018 to +0.025) and was withdrawn in step 28: the effect is the same
  size as the seed-to-seed spread. Needs several seeds per arm, not one.
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
