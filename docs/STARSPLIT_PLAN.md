# Learned star/nebula split (2026-11-05)

Owner: "Where else can we use the GPU" -> ranked list -> "Start 2", and the
same day: a starless version of a nebula is a product in its own right,
not only a step toward better palette renders.

## What it replaces

The palette renders (SHO/HOO with `--gain`) handle stars by hand rules in
`stacking/color_process.apply_palette_gain`: `sep` discs sized by a
peak-to-radius rule (`radius_from_dex`, halo term past 2.5 dex), a second
"quiet" pass on the denoised sum, painted diffraction arms along a measured
axis, a 16 px block-median fill under the discs, and the star light put back
neutral. Every new field found the next special case (red specks, red
spikes, magenta rings, NGC 7317's rings; CLAUDE.md has the history).

## What it is

`nn/starsplit`: a UNet (the N2N one, bilinear decoder, residual "linear" in
asinh units) that takes one background-subtracted plane and returns the
plane without its stars. `stars = plane - starless` exactly. Tiled at 1024
px with a 128 px crop stitch (`split_frame`).

Training pairs (`nn/starsplit_data`): REAL backgrounds with KNOWN stars
injected. The background is the plane with its 3-sigma stars filled by the
palette's own `smooth_fill`; the stars are the field's measured PSF
(`stacking/deblur.measure_psf`, wing and spikes to 72 px) at log-uniform
power-law peaks from 1.5 sigma to 30x saturation, clipped at saturation;
synthetic knots (1.3-3x the PSF) ride in the background so a compact HII
knot is not a star; 10 % flat or fractal backgrounds. Held out: trunk and
ngc7380 (never trained on); val: m76.

Why not hand-made starless targets: they teach the rules' faults; a
teacher-student net reproduces the teacher (LEARNED_DEBLUR_PLAN).

## Acceptance, in order

1. `score` on the holdout planes' synthetic patches (known truth): star
   light removed by brightness bin (want > 0.95 above 20 sigma, no
   over-removal), background preserved away from stars (rms err < 0.1
   sigma) and under stars, residual at bright stars' centres.
2. Real fields by eye, starless and stars-only, IC 410 / Melotte 15 / M1 /
   the trunk crops around the brightest stars, beside today's disc-and-fill
   result: no ring, no hole, no swallowed knot, arms gone.
3. Only then: `apply_palette_gain(star_source="learned")` behind a flag,
   the starless jpg beside each palette composite in shadow, `process ...
   stars=none`. Promotion is the owner's by eye, as for the deblur.

## Known limits of recipe v1

- The PSF stamp ends at 72 px: the far arms of a 3-4 dex star are not in
  the training input, so the net leaves what it never saw. v2: ribbons
  from `deblur._arm_template` beyond the stamp.
- Under a filled disc the background is a block median, smoother than
  nature; the net may learn a slight over-smoothing under stars.
- One plane at a time: the three filters are split independently; the
  palette's neutral-star rule then averages the three star layers.

## Log

- 2026-11-05: built `nn/starsplit_data`, `nn/starsplit`,
  `scripts/starsplit_train.py` (prepare / train / score / split). v1_s0
  launched 07:20 (13 planes, val m76, 137 s/epoch).
- 2026-11-05 07:50, epoch-13 checkpoint on IC 410 Ha (a training plane):
  faint and medium stars gone cleanly, nebula untouched (0.02 ADU away
  from stars), the star layer clean; bright stars (> ~2000 ADU) leave a
  nub inside a dark ring and a faint far spike survives. Diagnosis: the
  injected core was a bilinearly shifted median stamp (broader than a real
  star), bright stars are rare under the power law, the stamp ends at 72
  px. **Recipe v2** (same file, constants `SHIFT_ORDER`, `BRIGHT_FRACTION`,
  `ARM_FROM_SIGMA`): cubic-spline shift; a quarter of the stars drawn
  log-uniform from 100 sigma; past 1500 sigma the azimuthal wing continues
  as a measured power law beyond the stamp (level anchored at the taper's
  start, slope clipped to -3.5..-1.5) and four arms ride out along the
  first measured spike axis and that axis + 90 deg, at the template's
  on-axis level at 53 px falling as 1/r, to the palette's length rule.
  Found on the way: IC 410's and Melotte 15's brightest stars have arms at
  65 and 155 deg (four vanes), but `deblur.measure_psf._spike_axes` read
  the second axis at 119 deg on IC 1396, so `starless.py`'s arm removal
  may take arms off at the wrong angle on some fields; the palette's
  mod-90 fold is right.
- 2026-11-05 09:00, v1_s0 done (best val 0.0206 at ep 39). Held-out TRUNK by
  eye: faint and medium stars gone, nebula untouched (0.003 ADU away from
  stars), star layer clean; the two brightest stars (8-10k ADU) leave a nub
  (541 / 277 ADU at the centre) inside a faint ring, and a faint spike line.
- 2026-11-05 10:30, v2_s0 done (val 0.47, not comparable: harder data). On
  the Trunk it LOSES: faint stars left behind, background 10x less faithful
  (0.034 ADU), the nub BIGGER (2339 / 1656 ADU), and the star layer carries
  the nebula (the pillar is in it). A quarter saturated giants with 400 px
  wings taught it that broad smooth light near a bright thing is star. So
  the bilinear core was not the nub's cause. Panel
  ~/Desktop/trunk_Ha_starless_v1_vs_v2.jpg. Score on the common synthetic
  benchmark (v2 data): v1 removed 0.90/0.95/0.99/0.68 of the light at
  5-20/20-100/100-1000/>1000 sigma, v2 0.92/0.97/0.97/0.81; background away
  from stars v1 0.60 sigma rms, v2 0.86.
- Diagnosis of the nub: the asinh L2. 500 ADU left of a 10,000 ADU peak is
  a few pixels a little off in asinh units; the sky term is satisfied long
  before. **v3** = v1's data (no far wings or arms, bright fraction 0.05,
  spline cores kept) + `lam_frac` 1.0: the flux-weighted fraction of star
  light left behind, the very number `score` reports. 60 epochs, launched
  10:40 as v3_s0.


- 2026-11-05 13:00, v3_s0 done (val 0.126 at ep 60). Synthetic benchmark:
  0.94/0.97/0.996/0.98 of the light removed by bin (v1 0.90/0.95/0.99/0.68).
  Trunk: bright-star residual rms within 12 px 45/29 ADU (v1 69/40), but
  the centre now reads -208 ADU (a hole) where v1 read +541 (a nub); 33
  star-like residuals left of 358 (v1 20); sky fidelity 0.019 ADU (v1
  0.003); smooth light in the star layer equal (0.8 ADU rms). A trade. The
  sign flip on the same footprint says the net cannot represent the real
  bright core: the injected core is the field-median PSF of unsaturated
  stars. **v4**: past 1000 sigma a star is injected as a REAL bright star
  cut from the plane (`real_star_stamps`, up to 24 isolated unsaturated
  stars above 1500 sigma, peak-normalised), v3's loss kept with a floor on
  the fraction term's denominator. Panel ~/Desktop/trunk_Ha_starless_v1_vs_v3.jpg.
\n
- 2026-11-05 15:30, v4_s0 done (val 0.273, converged; the real stamps'
  noise is in the target, so the floor is higher than v3's 0.126). Trunk:
  bright-star residual rms 39/27 ADU (best), the spike lines GONE (the
  stamps carried the real arms), but a dark annulus at 20-40 px (halo
  over-subtracted), 35 faint residuals of 358 (v1 20), sky 0.054 ADU (v1
  0.003). Synthetic: 0.85/0.94/0.994/0.99 by bin. Four recipes each moved
  the bright-star error rather than removing it. **Decision:** stop
  iterating the recipe. v1 is the starless product (best sky and faint-star
  removal); the two or three brightest stars per field go to a HYBRID with
  `starless.py`'s PSF fit (`star_layer`) above ~2000 ADU, the network
  below. Panel ~/Desktop/trunk_bright_stars_v1_v3_v4.jpg. Checkpoints:
  local/starsplit/models/v{1,2,3,4}_s0.pt.
- 2026-11-05 16:20, the PSF-fit hybrid tried on the Trunk (v1 + `deblur.star_layer`
  spliced in at the 15 stars > 2000 ADU): the 12 px rms read 7/3 ADU against
  v1's 69/40, but the panel shows a bright annulus at 15-25 px around every
  star and a hole at one. star_layer was built for deconvolution: it fits
  off the far wing and lifts the core onto a fill, and the halo between
  stays. Not a star remover; dropped. Next: v1 + a disc fill of the compact
  nub only (~10-20 px), the net having already taken the halo and arms.
- 2026-11-05 16:40, **the product path is settled: v1 + a fill of the
  compact residual at bright stars** (`nn/starsplit.fill_bright_residuals`,
  on by default in `split_frame`): compact peaks above 1200 ADU over the sky,
  the disc where the starless differs from its surroundings by 3 sigma
  (8-30 px, +2) filled by `starless.normalised_fill`, feathered 4 px.
  Trunk: residual rms within 12 px 69/40 -> 0/0 ADU, the close pair's
  companion 12 -> 1, compact residuals left 20 -> 14 of 358, nothing else
  touched. Panel ~/Desktop/trunk_bright_stars_v1_vs_corefill.jpg. Next:
  the starless jpg beside each palette composite in shadow and
  `process ... stars=none`; then `apply_palette_gain(star_source="learned")`
  judged on IC 410 / Melotte 15 / NGC 7317 beside the hand rules.

