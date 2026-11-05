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
