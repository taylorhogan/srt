# Learned deconvolution plan — 2026-10-19

The second of the three directions laid out when the star heuristics in
`stacking/deblur.py` stopped converging (2026-10-18): **train a network to
undo the field's own blur, using the Noise2Noise training setup**, so that
clipped cores, spikes and denoiser-eroded profiles are handled because the
model has seen them, not because someone wrote a rule for each. The first
direction — RL on the starless image with the stars rendered back
(`stacking/starless.py`) — shipped as the default on 2026-10-19 and is the
baseline this plan has to beat. It is good on ngc6946 and ngc7320 and known
to be weak where a bright star sits on a smooth galaxy (NGC 7317), and every
new field has so far found the next special case.

This plan follows the rules the denoiser work paid for
(`N2N_IMPROVEMENT_PLAN.md`, `N2N_LAB_MANUAL.md` steps 28–38): the control is
built before the experiment, every judgement has ground truth or a paired
baseline, nothing is judged by loss, best epoch or a preview, and the analysis
that decides whether training is worth doing runs first.

## The idea, and the one assumption it rests on

We have no sharp reference images, so the training signal is manufactured
from the data itself. Take a real stack `X`, which carries one PSF `K` of blur
(`X = K * S` for the unknown sharp scene `S`). Blur it once more with the
field's own measured kernel: `Y = K * X + n`. Train the network to recover `X`
from `Y`. At inference, feed it `X` and it removes one kernel's worth of blur,
giving an estimate of `S`.

The assumption is that "remove one `K` from `K*K*S`" and "remove one `K` from
`K*S`" are the same operation. For a Gaussian kernel it is exact: the transfer
learned is `exp(+k²σ²/2)`, which applied to `K*S` returns `S`. For the real
kernel — a Moffat-like core, a wing, spikes — it is approximately true for the
core and untested for the rest. **Phase 0a tests it on synthetic scenes where
`S` is known before anything is trained on real data.** If it fails there, the
method is replaced by the teacher–student fallback at the end of this
document, which buys speed and smoothness but not quality.

Three design choices follow from how this pipeline has failed before:

1. **Non-blind, kernel-conditioned.** A blind network has to decide whether a
   smooth galaxy envelope is blur or structure, and a scene class missing from
   training will be read as blur, exactly as the denoiser read M33's arms as
   noise (step 37). So the network is *told* the kernel. The cheapest faithful
   encoding is a second input plane `K * image`: the difference between the
   two planes shows the network what one kernel of blur does to this
   particular image, locally, with no hand-built descriptor. `UNet(in_ch=2)`
   already exists. At inference the plane is `K * X` with the measured `K`.
   A kernel narrower than measured gives a milder correction, which is the
   strength knob, free.
2. **Saturated cores are left as imaged.** A plateau has no "one kernel
   sharper" answer. The loss is masked within `deblur.flat_tops` + `SAT_DILATE`
   of every plateau, and inference writes the input back inside those discs,
   the same guard `deblur.py` already applies.
3. **Noise is restored after blurring.** `K*X` has smoothed noise; the real `X`
   does not. A network trained on smooth noise sees the real stack's noise as
   detail and sharpens it. Noise is added to `Y` to bring its fine-band power
   back to `X`'s measured sky sigma (`deblur.sky_sigma`), so the only
   difference between training input and inference input is the blur.

The head parametrisation is the open hardware-level question. N2N's
`residual="linear"` exists to keep bright pixels fixed, which is the opposite
of what a star needs here (its core gains flux, its wing loses it). Three
candidates — the asinh residual, a multiplicative log-gain in linear units
(`out = x · exp(h)`, RL's own form), and the linear residual — are compared on
synthetic data in Phase 0b before any real training.

## What "better than starless" means, measurably

Every metric is computed identically for the learned model, for
`method=starless` and for `method=rl`, on the same held-out stacks, paired per
tile (`scripts/n2n_compare_paired.py` pattern). Effects must clear the
two-seed floor measured in Phase 0c.

1. **Recovery of known truth** on re-blurred held-out stacks: residual RMS per
   octave against `X` (this is the training objective on unseen fields, so it
   is a ceiling, not a verdict).
2. **Injection on real stacks** (`n2n_injection_test.py` method): a sharp
   template (Sérsic knots, a point source) convolved with the measured `K` is
   added to the real stack; the incremental response `D(X+I) − D(X)` is
   compared with the pre-convolution template. The one real-data measurement
   with ground truth.
3. **Stars, on the real stack.** Per brightness bin, 20 bins from 50σ to
   clipped: output FWHM (target: uniform across bins, near the kernel's
   self-deconvolved core, 5.2 px from 8.0 on ngc6946); aperture flux
   conservation r ≤ 12 px (within 2%); ring depth, the minimum in the 6–20 px
   annulus relative to local sky (never below sky − 1σ); residual spike
   power beyond 10 px. These are the four artefacts the hand-written path
   fought, made numbers.
4. **Split-half correlation of the added detail.** Deblur the two independent
   half-stacks the ladder already has (`*_train0/1`, `*_val0/1`). High-pass
   the difference output − input at 2–16 px and correlate between halves.
   Real recovered detail correlates; invented detail does not. This is the
   hallucination guard and the same test that justified 40 RL iterations.
5. **Acceptance by eye** on the two approved crops: ngc6946 HALRGB (centre
   3031, 4849; 3200×2400) and ngc7320 LRGB (3080, 4620; 2000×1500), against
   `*_default.jpg` on the Desktop. Both targets are held out of training for
   that reason.

## Phase 0 — analysis and feasibility (~1 day, small GPU runs). Decides whether Phase 2 exists.

**0a. Does one-kernel removal compose on the real kernel?** Build fully
synthetic fields: Sérsic galaxies, point stars over five decades of flux,
convolved with ngc6946's *measured* kernel including spikes
(`deblur.measure_psf`), clipped at the measured saturation level, Poisson
noise at the stack's level. Train a small model 10 epochs on `(K*X, X)` pairs
and apply it to `X`. Compare `f(X)` with `S` on metric 1 and 3, and against
RL40 on the same `X`. Pass: star FWHM within 10% of the Gaussian prediction
(FWHM/√2) with no ring deeper than 1σ, galaxy residual per octave no worse
than RL's. Fail: the method is dropped for the fallback below. One afternoon;
it would be the cheapest kill this project has had.

**0b. Head parametrisation** on the same synthetic set: asinh residual vs
log-gain vs linear residual, one seed each, scored on star flux and FWHM
error. The winner is fixed for the rest of the plan.

**0c. Noise floor.** Two seeds of the 0b winner. Record the spread of every
metric in the table above; nothing later is claimed inside it.

**0d. Baseline table.** `method=starless` and `method=rl` run on the held-out
ladder stacks (ngc5907 L/R/G/B, ngc7320, ngc6946, both raw and denoised) with
metrics 3 and 4 computed. This is the bar, and it is useful on its own: the
starless pipeline has so far been judged by eye only.

## Phase 1 — data (~1 day, mostly unattended)

- **Stacks.** Reuse `local/n2n_ladder/stacks/` (11 targets at 300 s, 2 train +
  2 val half-stacks per group, seed 0, 30 GB) and add ngc6946, ngc1961, m33 and
  ngc7380 through the same `nn/stacks.build_split_stacks` path so every
  training pixel is a byte-identical half-stack. Hold out ngc5907 (the
  ladder's quantitative test), ngc6946 and ngc7320 (the acceptance crops).
  Keep bubble and ic1396 in: nebulae teach "smooth and broader than the
  kernel stays", the failure mode this design most needs examples against.
- **Two domains.** The morning render deblurs the *denoised* stack; `process`
  deblurs the *raw* one. Denoise every training stack with the broadband
  default (`n2n_ladder_groups_galaxies_300s.pt`, narrowband with its own) and
  store both, so one model can be trained on either or both.
- **Per-stack sidecar** (`<stack>.psf.npz`): the measured kernel, spikes,
  saturation level, flat-top mask and sky sigma from `deblur.measure_psf`,
  `flat_tops`, `sky_sigma`. Measured once, so every arm trains on the same
  kernels. Stacks that give no kernel are dropped and listed.
- **Degradation on the fly** (new `nn/deblur_data.py`, a `Dataset` beside
  `N2NDataset`): per patch, blur with the stack's kernel, optionally scaled
  0.8–1.25× (the conditioning plane uses the same kernel, so the network
  learns "remove the kernel you are shown"), restore the noise, build the
  two-plane input, build the loss mask (plateaus, coverage fill). Patches
  512 px: a bright star's wing and arms reach 100 px (`ARM_RADIUS`) and the
  p512 arm of the ladder showed the band limit is partly a patch-size artefact
  (step 30). Same asinh normalisation as the denoiser (`denoiser.normalise`),
  so the two pipelines cannot drift apart.

Disk: about +25 GB (denoised copies and new targets). RAM: do not run beside
a stack or a transit search, as for the denoiser.

## Phase 2 — training (2–3 days wall, GPU mostly unattended)

Arms, two seeds each, 60 epochs, 512 px patches, 2000 pairs per epoch,
`MultiScaleL2` with the two coarsest octaves weighted up (the low-pass of the
residual is the flux-conservation term):

| arm | training domain | question |
| --- | --- | --- |
| A | denoised stacks only | the morning-render case on its own |
| B | raw + denoised pooled | does one model serve `process` and the render |

Estimated ~1–2 h per run at 512 px (N2N measured 31 s/epoch at 256 px and
~4× the pixels per patch); four runs in a day. If A and B tie within the
floor, B ships, since it covers both callers.

Scored on ngc5907 and the two acceptance targets, metrics 1–4, paired against
the Phase 0d baseline. **Gate to Phase 3:** star metrics at least as good as
starless (FWHM uniformity, flux within 2%, no ring, no spike residue), split-
half correlation of added detail at least RL's, and no detail correlated with
a pure-noise half. One iteration is allowed if the gate fails (patch size,
loss weights, kernel-scale range); a second failure is recorded in the lab
manual and the project stops there.

## Phase 3 — integration (~1 day)

- `stacking/learned_deblur.py`: load the checkpoint, measure `K`, build
  `(X, K*X)`, tiled inference 1024 px tiles with 128 px overlap and the crop
  stitch (the wings of bright stars need more context than the denoiser's 64),
  write the input back inside plateau discs, restore the sky.
  `scripts/n2n_tile_artifact.py`'s two-offset test checks tiling near bright
  stars before anything is promoted.
- `deblur.deblur_channels(..., method="learned")` beside `starless` and `rl`;
  `color_process` option `method=learned`. The reference channel only, as
  starless does, unless the recipe shows every plane (HOO/SHO).
- **Spark only.** Like the denoiser, this needs torch; the observatory PC's
  `process` has none. `method=learned` on the PC refuses with a clear message
  and starless stays the PC default for good. The learned model can only ever
  be the morning render's default.
- Trial week: the morning render writes a fourth composite tagged `learned`
  beside raw, denoised and deblurred, so every new field is a side-by-side
  with no extra effort. The Pushover goes out as four.

## Phase 4 — acceptance and promotion (~half a day)

Both crops compared by eye against `*_default.jpg`; the paired table from
Phase 2 re-run on whatever the trial week rendered. If promoted, the render's
`deblurred` tag switches to the learned model, starless stays as
`method=starless`, `CLAUDE.md`'s stacking paragraph and the lab manual get the
measurements, memory `project_deblur_denoised_psf` is updated, version bumped.

## Fallback: teacher–student

If Phase 0a shows one-kernel removal does not compose on the real kernel, the
honest alternative is to train the network on `(stack, starless-pipeline
output)` pairs. It can only reproduce the teacher — NGC 7317's rings included —
so it buys a 10 s forward pass instead of 5–10 min and a smoother function
across fields, not better pictures. Worth doing only if the render time
becomes the problem; it is not today.

## Risks, named

- **Hallucination** is the structural risk of any learned deconvolver.
  Guards: L2 (the conditional mean, not an adversarial sharpener), the
  split-half test, injection. A model that fails the split-half test is not
  promoted whatever the crops look like.
- **Composition on the real kernel** (Phase 0a) and **head parametrisation**
  (0b) are the two unknowns that could end the project early. Both are
  resolved in the first day, on synthetic data, before any real training.
- **Training/inference mismatch** was the denoiser's recurring failure
  (normalisation drift, depth mismatch, step 16). Here the degradation is
  applied to the same stacks the model is used on, in the same
  normalisation, with the noise restored; the kernel-scale augmentation is
  the only deliberate gap and the conditioning plane closes it.
- **Scope.** Narrowband fields, crowded fields and raw stacks are untested
  for starless too; this plan includes bubble/ic1396 and the raw domain but
  the acceptance is still the two galaxy crops. A nebula acceptance target
  should be chosen before Phase 4 if HOO/SHO renders are to use the model.

## Timeline

| phase | wall | GPU |
| --- | --- | --- |
| 0 feasibility and baselines | 1 day | ~2 h |
| 1 data | 1 day | denoising ~1 h |
| 2 training and scoring | 2–3 days | ~8 h |
| 3 integration and trial week | 1 day + 7 nights | per-render ~1 min |
| 4 acceptance | half a day | — |

About a week and a half of calendar time, two to three days of attention.

---

## Phase 0 results — 2026-10-19

Run on the Spark in one day with `scripts/deconv_phase0.py` (synthetic truth,
the four heads, the seed floor) and `scripts/deconv_phase0_real.py` (the
baseline table on real half-stacks). Everything under `local/deconv_phase0/`.
**Verdict: Phase 2 exists, with one architectural condition** (0e below).

### 0a. One-kernel removal composes on the real kernel — for extended structure

Synthetic fields: ngc6946's measured kernel (profile FWHM 7.47 px, 4% of the
flux beyond 24 px, spikes), saturation 66 031 ADU, sky 497, sigma 3.2; Sérsic
galaxies with nuclei, knots and dust lanes; 1500 stars with
dN/dlog(amp) ~ amp^-0.6 over 2–20 000 ADU plus 6 at 20 000–300 000 (half
clipped). Six training fields, two test fields, 2048 px. Twenty epochs of
2000 256-px pairs, four runs sharing the GPU at ~100 s/epoch.

The kernel convolved with itself has FWHM 12.38 px, not the Gaussian 10.56,
so one-kernel removal predicts a 4.50 px star, not 5.28. Measured, against
the known sharp scene, mean over both test fields:

| galaxy residual rms (sigma) | 2–4 px | 4–8 | 8–16 | 16–32 | 32–64 | total |
| --- | --- | --- | --- | --- | --- | --- |
| input (one kernel of blur) | 1.41 | 1.58 | 1.30 | 0.81 | 0.38 | 5.10 |
| RL40, true kernel | 1.06 | 0.66 | 0.25 | 0.21 | 0.18 | 3.00 |
| starless (the default) | 1.11 | 0.75 | 0.31 | 0.23 | 0.18 | 3.14 |
| learned, asinh head, 3 seeds | 1.08 ±0.01 | 0.70 ±0.02 | 0.20 ±0.01 | 0.08 ±0.01 | 0.17 | 2.87 ±0.03 |
| learned, scatter head (0e) | 1.07 | 0.69 | 0.20 | 0.10 | 0.09 | 2.86 |

Every learned head beats RL40 from 8 px up and matches it at 2–8 px, after a
20-epoch run on six synthetic fields. The composition assumption holds for
everything that is not a point source. (The 1–2 px band is the noise floor,
~1.4–1.8 sigma for everything.)

### 0b. The three pointwise heads all fail on stars; the fourth passes

Stars, per amplitude bin above sky, median over isolated stars (counts are
thin: 2–8 stars per field above 1000 ADU, so read the trend across bins):

| 1000–10 000 ADU | FWHM px | flux r≤12 / truth | ring (sigma) |
| --- | --- | --- | --- |
| input | 7.09 | 0.84 (the wing holds the rest) | +0.9 |
| RL40 | 2.86 | 1.01 | −1.21 (the moat) |
| starless | 3.62 | 1.01 | −0.21 |
| asinh head | 4.40 ±0.16 (3 seeds) | **1.56 ±0.17** | |
| loggain head | 4.24 | **1.56** | |
| linear head | 6.16 | 1.05 | |
| asinh + chi² flux term | 4.23 | **1.62** | |
| **scatter head + chi²** | 4.41 | **0.93** | +2.2 (no ring) |

The asinh and loggain heads sharpen to the predicted width and **create
flux**: 1.4–1.8× the truth at every bin above 100 ADU, seed-dependent (the
seed floor on this number is ±0.2, the largest of any metric). The linear
head conserves flux and barely sharpens a bright star, as the plan predicted.
A Poisson-weighted chi² term in linear units did not move the flux at all
(1.62 vs 1.56 ±0.17), so this is not loss weighting. The mechanism: the flux
that must move into the core sits in the wing, below the noise for all but
the brightest stars, so a pointwise head cannot measure it and learns a
*core gain* instead — the gain that is right for K*K (training) is wrong for
K (inference). That is the composition failure, and it is specific to point
sources.

**0e. A flux-conserving head fixes it.** The network emits, per pixel, a
15×15 window of softmax weights, and each input pixel's flux is scattered
over its window by those weights (`F.fold`): flux is conserved by
construction, nothing can go negative, nothing can ring. Trained once (seed
0, with the chi² term): stars at 4.1–5.2 px, flux 0.92–0.94 of the truth
(the input's 0.84 plus what the window could reach of the wing), no negative
ring at any bin above 100 ADU, galaxy residual at the asinh floor. Its limit
is the window: flux can move 7 px per pass, so a bright star keeps a faint
halo at the window's edge. Phase 1's architecture is therefore the scatter
head with reach to the wing (a second, coarser scatter level, or a 2-pass
scheme), not any of the three heads the plan listed.

Also measured: all three learned heads suppress noise as well as sharpen,
since the L2 optimum is the conditional mean — the outputs are visibly
smoother than RL's. Not a defect, but it means the learned deblur partly
overlaps the denoiser, and the order of the two should be re-measured in
Phase 2.

### 0c. The seed floor (asinh head, seeds 0/1/2, half-range)

| | floor |
| --- | --- |
| galaxy residual, total | ±0.03 sigma |
| galaxy residual, 4–8 px band | ±0.02 |
| star FWHM, 1000–10 000 ADU | ±0.16 px |
| star flux ratio, 1000–10 000 ADU | ±0.17 |
| ring depth | ±0.3–1.5 sigma (thin bins) |

Galaxy metrics are stable; star metrics are not, at these star counts. Phase
1 needs more test fields (or a fixed star grid) before any star claim finer
than 0.2 px / 20% flux.

### 0d. The bar: starless and RL40 on real half-stacks

ngc5907 and ngc7320, L, the ladder's two independent half-stacks, 3000 px
round the galaxy, raw and denoised (both measured in the raw half's sigma).
The first numbers the starless default has had.

| stars 1000–10 000 sigma | FWHM | flux out:in r≤12 | ring |
| --- | --- | --- | --- |
| ngc5907 raw, RL40 | 3.25 | 1.18 | −1.43 |
| ngc5907 raw, starless | 3.93 | 1.16 | +0.11 |
| ngc7320 raw, RL40 | 3.08 | 1.20 | −1.42 |
| ngc7320 raw, starless | 3.82 | 1.20 | −2.13 |
| ngc7320 denoised, RL40 | 3.07 | 1.20 | −0.76 |
| ngc7320 denoised, starless | 3.87 | 1.20 | −3.48 |

A flux ratio of ~1.19 is what a perfect deconvolution gives here (1/0.84, the
wing brought inside the aperture), so both conserve flux on real stars. The
starless path's rings are not moats but the **NGC 7317 case**: on the
denoised ngc7320 half-stack the 100–1000 sigma bin bottoms at −164 sigma
(raw sigma), a hole beside the bright star on NGC 7317's envelope, worse on
the half-stack than on the full stack where it showed as coloured rings.
That is the one known failure of the default and it is now a number.

Split-half correlation (Spearman, band-passed added detail 1–8 px, galaxy
pixels with every star disc cut):

| | rho added (RL40 / starless) | rho of the input's own detail |
| --- | --- | --- |
| ngc5907 raw | 0.64 / 0.65 | 0.40 |
| ngc7320 raw | 0.94 / 0.94 | 0.74 |
| ngc5907 denoised | 0.57 / 0.44 | 0.79 |
| ngc7320 denoised | 0.90 / 0.88 | 0.97 |

On raw stacks what the deconvolution adds agrees between halves *better*
than the input's own fine structure does: the added detail is real. On
denoised stacks it agrees *less* well than the input's, which the denoiser
has already made smooth and consistent; the deconvolution of a denoised
stack adds more half-specific structure. Not a verdict on the current
stack → denoise → deblur order, but the first measurement that questions it,
and Phase 2's injection test on real stacks is where it gets settled.

### What changes in the plan

- Phase 1 builds the scatter head with reach to the wing, not the asinh head.
- The synthetic star catalogue stays as a *test* instrument alongside the
  real half-stacks: it is the only place star flux has a truth.
- Phase 2's gate adds: flux ratio within 0.05 of the starless path on real
  stars, and rho_added on denoised stacks no lower than the starless path's.
- The teacher–student fallback is not needed.

---

## Phase 1 built — 2026-10-19

What exists, where, and how it is run. Phase 2 (training arms and scoring)
starts from here.

- **`nn/deblur_data.py`** — `prepare_stack` writes a background-subtracted
  stack (`<name>_sub.npy`, memory-mapped at training time) and its sidecar
  (`<name>.psf.npz`: kernel, spikes, FWHM, saturation, sky sigma, plateau
  and coverage-fill masks, measured once on the stack the model will see).
  `DeblurDataset` draws source-biased 512 px patches and builds the pair on
  the fly: the patch re-blurred by its own kernel scaled U(0.8, 1.25), noise
  restored, the kernel plane beside it, the target the patch itself, a mask
  that drops plateaus (+6 px) and fill. Patches are read with a 160 px margin
  so the two convolutions see real context. Normalisation is
  `denoiser.normalise`'s, the same rule inference uses.
- **`nn/deblur_model.py`** — the scatter head on the N2N U-Net: 449 softmax
  weights per pixel over a dense 15×15 window and a dilation-3 15×15 window
  (reach 21 px, the wing), flux moved by `F.fold`. Identity at init.
  `deblur_frame` is the tiled inference (1024 px tiles, 128 px overlap, crop
  stitch, planes built on the whole frame, plateau guard, sep background off
  and on). Checkpoints carry the asinh scale and refuse a mismatch.
- **`scripts/deconv_train.py`** — `prepare` walks the ladder manifest and
  writes both domains for every usable group's two training halves (~25 s
  and 0.5 GB per stack; the denoised domain through the broadband or
  narrowband N2N model as the render does it). Holdouts ngc5907, ngc6946,
  ngc7320 are prepared and flagged, never trained on. `train --name --seed
  --domain raw|den|both` fits the model, epoch selection on m13 + abell2218
  (whole targets), loss L2 in asinh space + λ·chi² in linear units.
  `smoke` runs one short epoch end to end. `list` prints the pool.
- **`tests/test_deblur_learned.py`** — flux conservation and identity of the
  head, kernel scaling, sidecar round-trip and pair construction (shapes,
  blur ordering, mask), tiled inference equal to a single tile.
- **Extra half-stacks** for ngc6946, ngc1961 and ngc7380 are being built
  through the ladder's own `stacks` stage (`--dsos`), so they join the
  manifest like every other group; ngc6946 is a holdout, the other two train.

Two things found while building it:

1. The first smoke run's loss was 18.8 at identity, and the cause was not an
   unmasked plateau (abell2151 B has none, raw or denoised) but the chi²
   term's scale on real fields: a 10 000-sigma star blurred once more is
   6 000 sigma off at its core, and Poisson-weighted that is thousands per
   pixel. Measured at identity on 24 real 512 px patches: asinh L2 1.29,
   chi² 5.7 on average with a tail to 51, where Phase 0's synthetic fields
   had few such stars. λ defaults to 0.2 so the two terms are comparable at
   init; Phase 2 should sweep it once (0.05, 0.2, 1).
   The denoised sidecar also ORs in the raw half's plateau mask as a
   precaution, since the raw half is where a ceiling is certain to be seen.
2. The epoch-selection scenes are a globular cluster and a rich cluster, so
   the val loss is star-heavy. Fine for picking an epoch, not a quality
   number; the quality numbers are Phase 2's.

Run order for Phase 2: wait for `prepare` (all groups) and the extra stacks;
`train --name scatter_both_s0 --seed 0 --domain both`, then seeds 1 and 2;
score with `deconv_phase0.py`'s synthetic catalogue (star flux truth) and
`deconv_phase0_real.py`'s half-stack metrics against the starless rows.

## Phase 2, first arm — 2026-10-20

`scatter_both_s0`: scatter head, both domains, seed 0, 60 epochs of 2000
pairs on 104 stacks (9 targets; 168 prepared in all after the extra
ngc1961 / ngc6946 / ngc7380 halves), 12 validation stacks, λ 0.2. 6 h on
the Spark (416 s/epoch). Train loss 4.52 → 0.58, settled by epoch 30; val
0.69 at the best epoch (50) and bouncing 0.77–1.12 after, a factor 1.5
between neighbouring epochs — the star-heavy validation scenes let a few
bright stars swing the χ² term. Scored with `deconv_train.py score`
(`learned_s0` rows in both harnesses).

**Synthetic (truth S, ngc6946 kernel).** Star flux 0.93–0.97 of truth across
the four unsaturated bins (Phase 0 scatter 0.92–0.94; starless 1.01; every
pointwise head 1.4–1.8), FWHM 3.7–4.7 px against the one-kernel prediction
4.50 (starless 3.6, RL40 2.6–3.7), no rings (deepest −0.8 σ, the input's
own −0.4). Galaxy residual 2.58 σ total — the best row in the table (RL40
3.00, starless 3.14, previous best learned 2.73) — and the only method
that improves every octave including 32 px and above (0.074 / 0.047 vs
starless 0.180 / 0.145). The saturated bin is the input, by the guard.

**Real half-stacks (ngc5907, ngc7320 L; holdouts).**

| target/domain | method | FWHM px | flux out:in | deepest ring σ | ρ_added | ρ_input | added rms σ |
|---|---|---|---|---|---|---|---|
| ngc5907/raw | starless | 3.9–4.0 | 1.16–1.20 | −0.54 | 0.654 | 0.403 | 0.76 |
| | learned_s0 | 4.3–4.4 | 1.15 | −0.83 | 0.282 | | 0.72 |
| ngc5907/den | starless | 4.2–4.3 | 1.22–1.25 | −2.49 | 0.437 | 0.794 | 1.74 |
| | learned_s0 | 4.1–4.3 | 1.14–1.15 | −0.03 | 0.474 | | 0.82 |
| ngc7320/raw | starless | 3.8–3.9 | 1.20 | −16.7 | 0.937 | 0.742 | 2.29 |
| | learned_s0 | 4.2–4.4 | 1.15 | −0.86 | 0.656 | | 1.82 |
| ngc7320/den | starless | 3.9 | 1.20–1.21 | −163.9 | 0.884 | 0.969 | 2.44 |
| | learned_s0 | 4.1–4.2 | 1.13–1.16 | −0.19 | 0.879 | | 1.92 |

(flux out:in is the r ≤ 12 aperture after over before; the synthetic truth
for this kernel sits at 1/0.84 = 1.19, so starless 1.20 is on it and learned
1.15 is the synthetic 0.95 again — the wing is collected to 95 %.)

- **The weak case is gone.** NGC 7317's star: starless −164 σ, RL40 −4.9 σ,
  learned −0.19 σ on the denoised half, −0.86 on the raw. By eye the star
  is a tight disc with the galaxy envelope and the spikes running through
  it unbroken; starless has rings and a moat, RL40 a moat with the wing in
  fragments. No star in any bin on any half rings deeper than −0.9 σ.
- **Flux gate (within 0.05 of starless):** 1.145 vs 1.200, just outside,
  in the under-collecting direction the synthetic predicts. The head's
  reach (21 px) or the kernel-scale range is the lever; the pointwise heads'
  failure mode (creating flux) does not appear.
- **Sharpness:** learned stars are 0.3–0.5 px broader than starless on real
  halves (4.1–4.4 vs 3.8–4.3) and 0.8–1.2 px broader than RL40.
- **Split-half:** on the denoised halves the gate holds (ρ_added 0.474 ≥
  0.437 on ngc5907; 0.879 vs 0.884 on ngc7320, a tie) with half the added
  rms — the same agreement from less change. On the raw halves ρ_added is
  well below starless's (0.28 vs 0.65; 0.66 vs 0.94): what the net adds to a
  raw half agrees less with what it adds to the other. The net was trained
  with the noise restored on re-blurred stacks, so on a raw input it may be
  sharpening noise peaks; the denoised rows say that goes away once the
  noise does, which is the order the pipeline uses. Seeds 1 and 2 (running)
  say whether 0.28 is the model or the seed; the λ sweep is the second
  check.
- Plateau guard did nothing on the real crops (brightest pixel 52.5 k of a
  63.9 k ceiling): the −164 σ starless hole was never a saturation effect.

Next: seeds 1 and 2 (`scatter_both_s1/2`, ~6 h each, queued), λ sweep
(0.05, 1.0) at seed 0, then the reach question if the flux stays at 0.95.

### Seeds 1 and 2 — 2026-10-20

Same recipe, seeds 1 and 2 (seed 2 is the first run with the loader
workers keyed on worker index rather than pid, so it is the first
recoverable one; the sampling distribution is unchanged). Best epochs 49
and 59. The seed floor, three runs:

| metric | s0 | s1 | s2 | starless |
|---|---|---|---|---|
| synthetic star flux (100–64k bins) | 0.96–0.97 | 0.97 | 0.97 | 1.01 |
| synthetic FWHM px, 1000–10000 bin | 3.85 | 3.55 | 3.73 | 3.62 |
| galaxy residual σ total | 2.58 | 2.56 | 2.48 | 3.14 |
| real FWHM px (median of bins) | 4.1–4.4 | 3.8–4.1 | 3.9–4.5 | 3.8–4.3 |
| real flux out:in | 1.15 | 1.15–1.16 | 1.15–1.16 | 1.20 |
| NGC 7317 ring σ (den) | −0.19 | −0.38 | −0.30 | −163.9 |
| ρ_added ngc5907 raw / den | 0.28 / 0.47 | 0.29 / 0.48 | 0.29 / 0.48 | 0.65 / 0.44 |
| ρ_added ngc7320 raw / den | 0.66 / 0.88 | 0.67 / 0.88 | 0.66 / 0.88 | 0.94 / 0.88 |

Seed spread: flux ±0.01, galaxy residual ±0.05, real FWHM ±0.3 px,
ρ_added ±0.01. So every finding of the first arm is the model, not the
draw: the galaxy residual beats starless by 0.6 σ (12 seed spreads), the
0.05 flux shortfall is real, and the low ρ_added on raw halves is a
property of the design, as is its absence on denoised halves. Nothing
here depends on the seed except star FWHM, where s1 is the sharpest and
matches starless; any single-seed sharpness claim under 0.3 px is noise.

λ sweep started 17:41 (0.05 then 1.0, seed 0, ~6 h each): does the χ²
weight move the flux shortfall or the raw-half ρ_added?

### λ sweep, first half — 2026-10-21

λ 0.05 at seed 0, two independent runs of the same recipe (a second
launcher from another session, `scatter_both_lam0p05_s0`, and this one,
`scatter_both_s0_lam0p05`; they shared the GPU overnight and both wrote to
`lam_launch.log`, so that log is interleaved — read the models' own
`.json`). Against λ 0.2 (seed 0):

| metric | λ 0.2 | λ 0.05 (two runs) | starless |
|---|---|---|---|
| synthetic star flux (100–64k) | 0.96–0.97 | 0.97–0.98 | 1.01 |
| galaxy residual σ total | 2.58 | 2.49–2.50 | 3.14 |
| real flux out:in | 1.15 | 1.15–1.16 | 1.20 |
| real FWHM px | 4.1–4.4 | 3.9–4.8 | 3.8–4.3 |
| ρ_added ngc5907 raw / den | 0.28 / 0.47 | 0.29 / 0.46–0.47 | 0.65 / 0.44 |
| ρ_added ngc7320 raw / den | 0.66 / 0.88 | 0.66 / 0.87–0.88 | 0.94 / 0.88 |

The χ² weight moves neither the flux shortfall nor the raw-half ρ_added:
both sit at the seed floor at 0.05 and 0.2. The galaxy residual gain
(0.08 σ) is under two seed spreads. The two λ 0.05 runs, identical recipe
and seed, differ by up to 0.4 px in star FWHM — cuDNN's own
nondeterminism is at the seed floor, so "recoverable" means the
distribution, not the weights. λ 1.0 is running (the other session's,
epoch 25 of 60 at 06:20, done ~10:30); if it too leaves flux at 0.95 the
lever is the head's reach, not the loss.

### Full-frame re-measurement and the production baseline — 2026-10-21

Two corrections to everything above this line in the real-stack tables.

1. **The starless rows were a crop artefact.** The real harness ran
   starless and RL40 on the 3000 px crop. On the crop starless found too
   few stars for its star layer, classified disk clumps of ngc5907 as deep
   star masks and filled them at sky (64 masks on the crop, none on the
   frame), and sharpened what stars it had with rings — the −164 σ "NGC
   7317 hole" was that regime, not the morning render. The harness now
   deblurs the whole half-stack and crops after; ngc6946 is a third holdout.
2. **Production starless does not sharpen stars.** `stars="imaged"` has
   been the default since 2026-10-19 evening: the star layer goes back as
   imaged and only the galaxy is deconvolved. On the full frames the
   starless star columns are the input's (7.5 px, flux 1.000, no ring).

So the comparison that matters is: starless = galaxy RL40 with untouched
stars; learned = galaxy one-kernel-removed AND stars sharpened. Five
models, full-frame, all three holdouts, denoised halves (the pipeline's
order); seeds 0–2 at λ 0.2, plus λ 0.05 and 1.0 at seed 0:

| | starless (prod.) | RL40 | learned, 5 models |
|---|---|---|---|
| star FWHM px (real, den) | 7.1–7.5 (input) | 3.1–3.2 | 3.8–4.3 |
| flux out:in r≤12 | 1.00–1.01 | 1.17–1.20 | 1.11–1.16 |
| deepest ring σ | ≥ −0.03 (none) | −0.9 to −3.9 | −0.47 worst, none in practice |
| ρ_added ngc5907 / ngc7320 / ngc6946 (den) | 0.52 / 0.70 / 0.63 | 0.59 / 0.91 / 0.73 | 0.47 / 0.88 / 0.65 |
| added rms σ (den) | 0.9 / 1.7 / 1.1 | 1.0 / 2.8 / 1.5 | 0.8–0.9 / 1.9–2.1 / 1.0–1.1 |
| synthetic galaxy residual σ | 3.14 | 3.00 | 2.46–2.58 |
| synthetic star flux (100–64k) | 1.01 | 1.00–1.13 | 0.96–0.98 |

Reading: against what actually runs, the learned model sharpens the stars
from 7.5 to ~3.9 px with no rings (RL40 gets 3.1 px and rings every bright
star on a denoised stack), carries 96–98 % of the star flux, and improves
the galaxy at every scale. The one column it loses is split-half agreement
on ngc5907's denoised half (0.47 vs 0.52), with less added change than
starless. On raw halves ρ_added stays well under starless's (0.29 / 0.66 /
0.40 vs 0.60 / 0.85 / 0.87), the noise-sharpening explained above; the
pipeline denoises first. λ 1.0 (seed 0) is the sharpest of the five on
real stars (3.8–3.9 px) at the same flux and ring numbers and the best
synthetic galaxy residual (2.46), and is `learned_default.pt`.

**Decision (owner, 2026-10-21):** not the default yet. Built as the
FOURTH composite of the morning render (`n2n_lrgb_render.py` tag
`learned`; `nn/deblur_model.deblur_channels`, every channel by its own
kernel so flux conservation carries the colour ratio) beside raw, denoised
and starless, to be judged by eye at full resolution in colour on each
morning's targets, including the cases never measured here: nebulae,
crowded fields, Hα planes. What would make it the default: a week of
mornings where the learned composite is the one the owner would post, and
no target it breaks that starless does not.

### First colour composite, and the luminance-only form — 2026-10-21 noon

The first morning render with the fourth composite (ngc6946 HALRGB, every
plane through the network by its own kernel) showed the galaxy as the
sharpest of the four — arm knots resolved to points, dust lanes defined,
no rings on ordinary stars — and two artefacts the luminance measurements
above could not see:

- **Coloured specks over the sky.** Faint sources came out as
  single-pixel dots of red, cyan or yellow. Not misregistration: R, G and B
  star centroids agree to 0.02 JPEG px (0.1 px full-res) in the learned
  channels as in the denoised ones. Each plane sharpens its own residual
  noise into specks that do not coincide across planes; the raw-half
  ρ_added numbers above are the same effect in luminance.
- **A cyan-cored bright star.** The brightest (saturated) star's L core
  stayed a flat saturated disc, but the R plane's core was spread while
  G and B kept theirs (R peak fell to 250/255 with 1 px clipped against
  6 in the denoised plane; G and B unchanged at 10 and 13), so the centre
  reads cyan with a dark inner ring.

Both come from deconvolving the colour planes independently. Owner
decision: the learned composite is now **luminance-only** — L through the
network, R, G, B and Hα as denoised (`--learned-planes L`, default;
`all` is the old form). The colour stays where the blur put it under a
sharpened L, which is what the eye reads; per-plane deconvolution is kept
as a switch because the head conserves flux per pixel, so when it is
wanted the colour ratio survives.

### Promoted — 2026-10-21 afternoon

Re-rendered ngc6946 HALRGB (luminance-only) and ngc1961 LRGB at full
resolution; the owner, on the four-composite panels: "i like the learned
luminance only the best". Decision: **learned luminance-only is the
default.** `color_process.DEBLUR_METHOD = "learned"` (the `process`
command; `stacking/deblur.deblur_channels(method="learned")`, L through
the network, colour planes kept, every distinct plane when there is no L,
starless fallback without torch/checkpoint), and the morning render writes
three composites, raw / denoised / learned, with the starless one as an
optional `--deblur N` extra. The week-of-mornings test now runs on the
default rather than on a fourth picture; the untested cases are the same
(nebulae, crowded fields, narrowband without an L).

### Nebula sweep — 2026-10-21 afternoon

ic1396 HOO, squid HOO, ngc7380 SHO, trunk SHO re-rendered from cached
stacks (ngc6888's lights are not on the Spark). First pass: the render's
luminance-only selection matched nothing in a set with no L and silently
wrote no learned composite — the morning would have dropped every nebula.
Second pass, every narrowband plane through the network by its own kernel
(kernels 6.5–7.4 px from 240–400 stars; normalisation scale 0.02–0.05
against 0.09–0.28 on the galaxies, i.e. far below anything in training):

- **A mesh over the smooth emission.** Confirmed at 1:1 on trunk Hα: the
  denoised narrowband plane carries a faint hatch from the N2N narrowband
  model, invisible at the composite's stretch, and the network sharpens it
  into a visible grid across the nebulosity and the dark lanes. Power at
  2–3 px periods is *lower* in the output, so it is not the dilated window
  aliasing; it is real texture in the input, amplified.
- **Bright rims broken into beads** (trunk, ngc7380): smooth ionisation
  fronts come out clumpy. Not checked by split-half; not trusted.
- **Saturated stars with mottled halos** (ic1396, squid, ngc7380): the
  plateau guard writes the core back as imaged per plane while the wing
  around it is sharpened per plane; the planes disagree and the halo reads
  as rings and colour.
- The faint sky and the faint stars were acceptable; fewer coloured specks
  than ngc6946's broadband planes showed.

Decision: **no luminance plane → starless**, `stacking/deblur` method
"learned" falls back with info["fallback"], and the morning render writes
that product as `deblurred` so its name says which deconvolution made it.
`--learned-planes all` (or a named list) still forces the planes through
the network for experiments. The narrowband fix, if wanted, is training
data: the network has never seen a narrowband stack, and the denoiser's
hatch would have to be out of the training targets too.

### The mesh's source, and M33 — 2026-10-21 evening

**The hatch is the narrowband denoiser's checkerboard.** 2D power spectra
of a 1024 px smooth region of trunk Hα, large scales removed:

| frame | 4 px line (x radial median) | 2 px line |
|---|---|---|
| raw stack | 0.1 | 0.1 |
| denoised, `n2n_pooledNB_300s.pt` | **410** (x-axis), 39 (y) | 34; 230 in the faintest region |
| learned output of that | 393 | 33 |
| raw through the BROADBAND model | 0.2 | 0.1 |
| ngc1961 L through the NARROWBAND model | 37 | 17 |
| ngc1961 L through the broadband model | 1.7 | 8 |

Exactly 4 px and 2 px, axis-aligned, absent from the raw, absent when the
broadband model denoises the same narrowband frame, present when the
narrowband model denoises a broadband frame: it is the model, not the
data. The U-Net decoder upsamples with `ConvTranspose2d(k=2, s=2)`
(nn/noise2noise_model.py:142), the textbook checkerboard source; the
2026-08-19 narrowband checkpoint never learned to cancel it, the 2026-10-09
ladder-groups broadband checkpoint did. Amplitude 0.001–0.004 ADU, under
0.1 % of the emission and far below the raw noise, invisible until the
denoised sky is smooth enough and the deblurrer sharpens it. The deblurrer
is innocent: the line power is the same before and after it.

Fix: retrain the narrowband denoiser with the current ladder-groups
recipe, with "4 px line power under x5" as an acceptance number, and/or
replace the transposed convolution with upsample+conv in the architecture
for every future model. Until then narrowband stays on starless.

**M33 (in the training set; shallow, 25 L frames)**: split-half ρ_added on
three 3000 px crops of the denoised L halves, learned vs starless:
centre 0.797 vs 0.804, NE arm 0.786 vs 0.803, SW arm 0.783 vs 0.805, with
added rms 1.03–1.07 σ vs 1.35–1.44 σ. The granular texture the learned
composite shows across the disc agrees between independent halves as well
as starless's does, from a smaller edit: resolved star clouds, not
hallucination. Panel: ~/Desktop/m33_HALRGB_raw_denoised_learned.jpg.

### Narrowband retrain — 2026-10-21 night

The ladder-groups retrain of the narrowband denoiser (twelve groups, trunk
held out; lab manual step 39) did **not** clear the checkerboard: the worst
cell fell from x410 to x20 but faint emission still carries x14–24 at 4 px
and x75–260 at 2 px, and the broadband model shows x85 on the same trunk
spot. It is the decoder, not the weights; the retrain also kept 0.02–0.06
less extended flux than pooledNB in every bin. Not promoted. Second arm
running: the same recipe with `UNet(upsample="bilinear")` (Upsample +
3×3 conv, cannot make the checkerboard; old checkpoints load unchanged via
`UNet.matching`). Narrowband stays on starless until a denoiser passes the
line test, and only then is per-plane learned on narrowband worth re-checking.

**Arm 2 passed the line test** (23:20): every cell under ×5, at the raw
stack's level (trunk Hα panel ×410 → ×3.1). Rendered with it and
`--learned-planes all`, the learned trunk and IC 1396 have no mesh and the
rim is clean. What remains in the learned narrowband product is the
deconvolver's own: coloured single-pixel specks from per-plane
deconvolution (the same thing that made broadband luminance-only) and a
ringed halo on the saturated star. The bilinear denoiser keeps 0.03–0.05
less extended flux than pooledNB in every bin above 2σ; two bubble-only
arms (both decoders) showed it is the ladder recipe, not the pool and not
the decoder, and on the held-out trunk N2N objective the bilinear
twelve-group model is the best of the four arms (lab manual step 39).
**Promoted 2026-10-22** as the narrowband default in the render and the
deblur harness. Once a narrowband denoiser is
promoted the open question for *this* plan is the specks: per-plane
deconvolution on narrowband has no luminance to hide behind, so the next
lever is either a synthetic luminance (deconvolve the plane sum, carry
colour from the denoised planes) or training the scatter head on
narrowband stacks.

### Synthetic luminance for narrowband — 2026-10-22 morning

Tried, works mechanically, not promoted. `stacking/deblur.synthetic_luminance`
makes an L for a set that has none from the planes as mapped to R, G and B
with the palette's own weights (0.299/0.587/0.114), deconvolves that, and
hands back the deconvolved frame beside its reference (`LSYN` / `LSYN_REF`);
compose scales the picture by the ratio of the two stretched with one black
and the reference's background (`color_process.synthetic_gain`), so an
identical pair composes to the identical picture. Substituting the
deconvolved L as a real L was tried first and relit the faint end 27 %
darker: the per-plane black clip keeps the positive part of each plane
where a summed L carries the deficit. The planes' plateau masks are united
into the sum's (`plateau_union`), since a core clipped in Hα alone is
neither at a ceiling nor flat in the sum.

**What it fixed**, on trunk, ic1396, ngc7380 and squid with the promoted
bilinear denoiser (panels `~/Desktop/*_syntheticL_panel.jpg`): no mesh, no
coloured specks (the colour is the planes' own), rims sharpened, the dark
clouds at the same brightness as the denoised composite (smoothed
luminance ratio 0.94–0.99 above the faintest bin, and in linear ADU the
sky edit is +0.0004).

**What it exposed.** Around every bright star the network collects the
sky along with the halo over a ~110 px square (its receptive field) and
leaves the halo outside: trunk's brightest star, peak 2400 ADU, comes out
3.6× brighter in the core with the sky inside the square 0.37 ADU
(3.5σ of the denoised sky) *below* the local background and the halo
intact beyond 60 px — a dark square inside a halo at full resolution
(`~/Desktop/trunk_syntheticL_bright_star_fullres.jpg`). Saturated stars
(squid, ic1396) keep a mottled halo just outside the kept plateau. The same
over-collection is on broadband: ngc1961's learned L takes its brightest
stars' halos to the sky level out to ~80 px; faint broadband halos hide it,
large bright narrowband reflection halos do not. **It is the model, not the
synthetic path.** Split-half on trunk's synthetic L (halves of 2×16–30
frames, ρ_input only 0.47–0.52): learned ρ_added 0.34–0.38 at 2–5σ added
rms against starless 0.41–0.43 at 1.1σ — the learned nebula edit is larger
and reproduces less, though the two halves' learned frames agree by eye
(`~/Desktop/trunk_syntheticL_two_halves.jpg`; starless rings its bright
stars there, learned does not).

**State.** `stacking/deblur.NO_L_RULE = "starless"` that morning;
`no_l="synthetic"` (render `--no-l synthetic`) selected the synthetic path.
The next lever was the model — the section below — and since the evening
of 2026-10-22 the rule is `"synthetic"`.

### The bright-star moat is the kernel's wing — 2026-10-22 morning

Diagnosed before changing anything (`scripts/deconv_halo_check.py`, on the
promoted `scatter_both_lam1p0_s0`):

- **Not the plateau mask.** On saturated training stars the model pours
  back into the loss-masked plateau exactly what the re-blur took out
  (1.0× on seven stars, ngc1961 / ic1396 / m92): the mask is not a sink.
- **Not the loss.** On the training pairs (the frame re-blurred by its own
  kernel) the model's wing is right to within a sigma from 8 px out, on
  unsaturated 5000 ADU stars and saturated ones alike. The loss saw the wing
  and the model learned it.
- **It is the task.** `measure_psf` follows the wing out to 72 px, where the
  kernel is 1e-5 of its peak and still a 145 px stamp. A 5900 ADU star's
  wing at 45 px (10σ over the sky) *is* that kernel at 1.7e-4 of the peak,
  and a model taught to remove one such kernel removes it: on the frame
  itself the 32–60 px wing came back 63 % removed (mean of 12 stars on four
  stacks; 43–82 %), the core 2.3× within 4 px, the halo beyond the stamp
  untouched — a dark moat inside a halo, square because the stamp is. The
  "receptive field" of the morning's note was the kernel's footprint.
  Re-blurring the answer by the kernel does not give the frame back
  (`cycle` row: +80σ at 16–24 px, +25σ at 24–32), so the one-kernel
  composition assumption fails exactly on bright stars' wings.

**The fix is the kernel to remove, not the network.** 95 % of the kernel's
flux is inside 20 px; the core is the deconvolution and the wing is scene,
to be left as imaged the way starless leaves it. `nn/deblur_data.taper_kernel`
keeps the measured kernel as it is inside 1.5 FWHM and cosine-tapers it to
nothing at 3 FWHM (`KERNEL_TAPER`; 18–24 px on these stacks, holding
94–96 % of the measured kernel's energy), the dataset re-blurs by that
core (`DeblurDataset(kernel_taper=...)`), the checkpoint stamps the taper
and `deblur_model.load` attaches it to the model, and `deblur_frame`
tapers the measured kernel by the model's own setting before building the
kernel plane — so training and inference cannot drift. Checkpoints from
before today carry no taper and behave exactly as they did. Handing the
*promoted* model a tapered kernel at inference (out of its distribution)
already cut the 32–45 px removal from −15σ to −3σ on the 5900 ADU star and
kept 94–97 % of the galaxy's band-pass sharpening, which says the
mechanism is the right one; the retrained model should be identity beyond
the taper by construction. The core sharpening (7.5 → ~4 px) is untouched,
since the core is kept as measured.

**Arm launched 07:09:** `scatter_both_taper_s0` — seed 0, domain both,
λ 1.0, `--taper 1.5,3`, otherwise the promoted recipe (~8.5 h at 510 s per
epoch). Acceptance: `deconv_halo_check.py` against the baseline above
(target: wing 32–60 px removed near 0 %, core gain kept near 2×),
`deconv_train.py score` on the holdouts (star flux and FWHM, the galaxy
scales), the trunk synthetic L (`--no-l synthetic`) and ngc1961's learned
composite by eye for the moat. Promotion means repointing
`local/deconv/models/learned_default.pt` and, if the moat is gone, revisiting
`NO_L_RULE` for narrowband.

**First launch collapsed (07:09–08:00).** Val loss identical to five decimals
over four epochs, train not descending: the softmax had saturated at every
pixel (one weight > 0.999, logits in the thousands), the gradient was zero
and the model was frozen on a hard identity *worse* than the identity it
started from (val 5.96 against 3.65 for a fresh model). Reproduced on the
first 400 steps with the gradient norm and saturation logged: round step
50–100 a burst of 1e3–1e5 gradient norms (clipped to 1, but Adam's
per-parameter steps keep the direction) drives the unnormalised UNet
features until the head saturates; the whole-kernel recipe rides the same
burst out at |logit| < 260 — the core-kernel task is nearer identity, so
the sky's "stay put" pull has less star signal against it. Three
stabilisers tried on those 400 steps: a learning-rate warm-up alone (lr
1e-4 over 500 steps) still saturated 25 % of the pixels at step 300; a
logit-norm penalty `--logit-pen 1e-6` (pen·mean(logits²)) excursed to
|logit| 18 000 at step 59 and was back within ten steps, then settled where
the whole-kernel run sits (|logit| 50–130); it keeps a gradient when the
softmax has none. On the promoted model mean logit² is 16, so the term is
1.6e-5 of the loss on a healthy run. Relaunched 08:05 with it
(`local/deconv/taper_launch.log`; the collapsed log kept as
`taper_launch_collapsed.log`); validation is scored without the penalty so
arms stay comparable.


**Result (15:28, 60 epochs, 7.4 h).** No collapse; val below the promoted run
at every epoch (2.80 by epoch 40 where the promoted run never got under 2.94;
best epoch 57, val 2.51 — a nearer-to-identity task scores lower, so the
number is not a like-for-like gain). Acceptance (`local/deconv/models/scatter_both_taper_s0.pt`):

| measure | promoted (whole kernel) | taper arm (core) |
|---|---|---|
| bright-star wing 32–60 px removed, 12 stars / 4 stacks (`deconv_halo_check.py`) | 62.6 % | 4.9 % |
| core gain r < 4 px | ×2.32 | ×2.16 |
| cycle residual K·out − in at 16–24 px (ngc1961, in σ) | +80 | +37 |
| holdout stars, denoised: FWHM px (30–100 σ bin) ngc5907 / 7320 / 6946 | 3.76 / 3.78 / 3.76 | 3.96 / 3.97 / 3.97 |
| holdout stars, denoised: flux ratio out:in within 12 px | 1.14–1.16 | 1.05–1.06 |
| holdout stars, denoised: deepest ring (σ), 30–1000 σ bins | −0.07 … +0.02 | +0.06 … +0.41 |
| real galaxy band-pass gain 6–12 px, ngc1961 / 6946 / 5907 / 7320 | 1.35 / 1.40 / 1.22 / 1.31 | 1.21 / 1.26 / 1.09 / 1.15 |
| real galaxy band-pass gain 12–24 px | 1.08 / 1.11 / 1.15 / 1.11 | 1.03 / 1.05 / 1.05 / 1.04 |
| rho_added (den) ngc5907 / 7320 / 6946 | 0.47 / 0.88 / 0.65 | 0.45 / 0.83 / 0.61 |

The flux ratio is the moat in numbers: the whole-kernel model pulled 14–16 %
of a star's light from the far wing into r ≤ 12 px; the taper arm pulls 5–6 %,
which is the 1.5–3 FWHM roll-off region. The "ring" column goes slightly
positive because the wing the taper leaves alone is now counted as a
(bright) residual around the sharpened core — it is the star's own halo, as
imaged. By eye (`~/Desktop/taper_{ngc1961,trunk,ic1396}.jpg`: denoised |
promoted | taper, three brightest stars and the galaxy/nebula): the dark
ring between core and halo is gone on every star in all three fields, the
sky around bright stars is no longer scraped darker than the denoised
frame, trunk's and ic1396's synthetic-L stars keep their colour instead of
going blue-white, and the galaxy sharpening on ngc1961 is close to the
promoted run's. The cost is real: the galaxy band-pass gain at 6–24 px is
about two thirds of the promoted run's above unity, because the kernel's
wing blurs the galaxy at those scales too and the core-only task does not
undo it. The synthetic report's 8–32 px octaves (0.32 / 0.29 vs 0.15 / 0.09)
say the same thing and are not a fair test for a core-only kernel: the
synthetic truth is sharper than the task is asked to reach. A wider taper
(2–4 FWHM) would buy some of that back at the price of more wing removal;
untested. Promoted by the owner by eye the same evening:
`learned_default.pt -> scatter_both_taper_s0.pt`, and narrowband sets get
the synthetic L by default (`NO_L_RULE = "synthetic"`, `--no-l starless`
for the old rule).
Scripts: `scratchpad/halo/galaxy_bp.py` (band-pass on the real stacks),
`scratchpad/halo/taper_panel.py` (the panels), `scratchpad/halo/loss_plot.py`.

**The 2–4 FWHM arm (overnight 2026-10-22/23).** `scatter_both_taper24_s0`:
same recipe, `--taper 2,4 --logit-pen 1e-6`, launched 22:29, done 05:45, no
collapse, best epoch 60 val 2.64. The middle ground the section above
asked for, and it is nearer the whole-kernel model's galaxy than the
promoted arm's while keeping almost all of the moat fix:

| measure | whole kernel | core 1.5–3 (promoted) | core 2–4 |
|---|---|---|---|
| bright-star wing 32–60 px removed (12 stars) | 62.6 % | 4.9 % | 7.0 % |
| core gain r < 4 px | ×2.32 | ×2.16 | ×2.30 |
| holdout FWHM, denoised (ngc5907 / 7320 / 6946) | 3.76 / 3.78 / 3.76 | 3.96 / 3.97 / 3.97 | 3.83 / 3.85 / 3.85 |
| star flux pulled within 12 px | 1.14–1.16 | 1.05–1.06 | 1.09–1.10 |
| real galaxy band-pass gain 6–12 px (ngc1961 / 6946 / 5907 / 7320) | 1.35 / 1.40 / 1.22 / 1.31 | 1.21 / 1.26 / 1.09 / 1.15 | 1.26 / 1.33 / 1.13 / 1.19 |
| real galaxy band-pass gain 12–24 px | 1.08 / 1.11 / 1.15 / 1.11 | 1.03 / 1.05 / 1.05 / 1.04 | 1.03 / 1.06 / 1.07 / 1.04 |
| synthetic galaxy residual, total / 8–16 px / 16–32 px | 2.46 / 0.15 / 0.09 | 2.64 / 0.32 / 0.29 | 2.47 / 0.20 / 0.18 |
| corr. of its edit with the whole-kernel edit | 1 | 0.81–0.96 | 0.89–0.98 |

By eye (`~/Desktop/taper24_{ngc1961,trunk,ic1396}.jpg`: denoised |
promoted | 2–4) the two tapered models are hard to tell apart: no moat on
either, the 2–4 stars a touch tighter, the galaxy a touch crisper. The
wing it does remove (7 %) is the 2–4 FWHM roll-off region, 15–32 px, where
the halo is still bright enough that a dark ring would show if it were
over-collected, and it does not. Promoted by the owner the same morning:
`learned_default.pt -> scatter_both_taper24_s0.pt`, `KERNEL_TAPER = (2, 4)`
for every future arm. Outputs: `scratchpad/halo/taper24_*`,
`scratchpad/nebula_taper24/`.
