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
