# Detecting optical change against a seeing background

Goal: answer "has something changed optically?" — collimation drift, sensor tilt,
a spacer knocked, a mirror slipped — from the frames we already take, without a
star test or a trip to the observatory.

Status: **step 1 of 4 shipped.** The metrics are computed and logged nightly.
Nothing consumes them yet, on purpose (see "Why not build the command first").

---

## The finding that shaped this plan

The obvious approach — trend `optics`' existing metrics night over night — does
not work. Measured on sh2-92, same target, same filter (Ha), best night against
worst night, with **no optical change in between**:

| | 07-31 (good) | 07-13 (bad) | change |
|---|---|---|---|
| median FWHM | 1.88" | 3.11" | +65% |
| stars detected | 718 | 130 | −82% |
| `field_uniformity` | 0.042 | 0.079 | **+88%** |
| `tilt_score` | 0.149 | 0.189 | +27% |
| `coma_score` | +0.263 | +0.154 | **−41%** |
| `collimation_score` | 0.621 | 0.441 | **−29%** |

Weather alone moved every shape metric substantially, and in the direction that
reads as *"the optics improved"* — coma toward 0, collimation toward its 0.5
null. A real collimation drift and a bad-seeing night are not distinguishable in
those numbers, and they push some of them opposite ways. A night-over-night
comparison built on them would not merely be noisy, it would be misleading.

Two mechanisms, both fixable:

1. **Sample selection.** 718 → 130 stars. The survivors are the brightest, which
   are distributed differently across the field and sit nearer saturation.
   `coma_score` is a Pearson r over that population, so its meaning changes when
   the population does.
2. **Contrast dilution.** Seeing adds isotropic blur everywhere, raising FWHM and
   shrinking the relative eccentricity contrast that the optical signature lives
   in. Every correlation-style metric gets dragged toward its null.

The encouraging half: *within* one night those metrics are tight — `coma_score`
of 0.271 / 0.268 / 0.259 across three consecutive frames while FWHM swung
1.80" → 2.08". The frame-to-frame noise floor is around ±0.006. There is real
sensitivity available if the seeing dependence is removed.

---

## The redesigned metrics (shipped)

`fitsfwhm.compute_optics_trend_metrics()`. Every one is chosen to be robust to
seeing rather than merely correlated with optical quality.

| metric | what it detects | why it resists seeing |
|---|---|---|
| `seeing_floor_arcsec` | the seeing itself | p10 of per-cell FWHM — the best patch of field, where optics contribute least |
| `field_excess_arcsec` | overall optical blur | quadrature subtraction: √(median_cell² − floor²). Seeing adds in quadrature everywhere, so subtracting the floor removes it to first order |
| `edge_excess_arcsec` | field curvature / tilt | same, at the p90 cell — the worst part of the field |
| `sweet_spot_x/y/r` | **collimation** | position of the FWHM minimum from a quadratic surface fit. Seeing lifts the whole surface without moving its minimum |
| `radial_fraction` | coma / collimation | mean cos(2Δ) between elongation and the radial direction. +1 all radial, 0 random |
| `uniform_fraction` | **tracking / wind, not optics** | Rayleigh R of doubled elongation angles. 1 = every star elongated the same way |
| `uniform_angle_deg` | which way | direction of that elongation |

Three design rules run through all of it:

- **Fixed-N star sampling.** Always the same number of stars, in a fixed
  brightness band with the top few percent dropped as a saturation guard.
- **Quadrature, never ratios.** The old `field_uniformity` divides by the mean,
  which *builds in* an inverse seeing dependence — that is why it moved most in
  the table above.
- **Pool the night, don't average per-frame metrics.** The spatial fits are
  limited by stars-per-cell.

A useful side effect: `seeing_floor_arcsec` is a better per-night seeing estimate
than median FWHM, because it excludes the optical field degradation that median
FWHM folds in.

### How well it actually works — measured, not claimed

The first cut of this design did **not** deliver seeing-invariance, and the
measurements are worth recording so nobody re-derives them.

*Per-frame metrics, then median across 4 frames*, over the same good/bad night
pair: `field_excess` +269%, `sweet_spot_r` +148%, `radial_fraction` flipped
sign. Barely better than what it replaced. The reason is visible in the output —
`stars_used` came out at 127 on the bad night, because the frame only *contained*
129 stars. Fixed-N cannot equalise a population that does not exist.

*Pooling ~12 frames per night* helped materially. Scatter across three
comparable Ha nights (all 600+ stars/frame, seeing 1.68-1.96"):

| metric | per-frame then median | pooled |
|---|---|---|
| `edge_excess_arcsec` | 35% | **13%** |
| `field_excess_arcsec` | 86% | 47% |
| `radial_fraction` | 123% | 49% |
| `sweet_spot_r` | 41% | 51% |

So: better, and `edge_excess` is genuinely tight, but **these are not
seeing-invariant metrics and should not be described as such.** A 13-50%
night-to-night scatter under *comparable* conditions is the noise any change
detector has to beat. That is exactly why step 2 below is to accumulate a
baseline rather than to start flagging.

Encouragingly, the sweet spot lands in the same quadrant every night
(x ≈ −0.23 to −0.36, y ≈ +0.16 to +0.55) — its *direction* is a stable signal
even where its magnitude wobbles. That, not the scalar, is probably what a
detector should watch.

### The gate, and why it is on frames rather than stars

A night only gets measured if at least 5 frames could each supply the **full**
150-star sample. A bad night records nothing.

The first version gated on pooled star count instead, and that silently failed:
on 2026-07-13 eleven frames of ~130 stars pooled to 1489, sailing past any
total-star threshold while every individual frame was too thin to give a
comparable sample. Pooling short samples reintroduces exactly the seeing
dependence the design exists to remove. Gating on frames refuses that night
outright, which is the correct answer — a clouded or bad-seeing night does not
contain an optics measurement.

### Reading them

- `sweet_spot_r` growing over weeks → collimation drifting. The most
  interpretable single number here.
- `edge_excess` up while `field_excess` flat → tilt or spacing, not collimation.
- `radial_fraction` up with `sweet_spot_r` → coma, consistent with collimation.
- `uniform_fraction` up → **not optics.** Guiding, wind shake, or a cable snag.
  Cross-check against 850 hPa wind (see `weather.SEEING_LEVEL_HPA`).
- Everything moving at once with `seeing_floor` → weather. Ignore.

---

## Roadmap

**1. Compute and log nightly — DONE.** `end.py::_post_optics_trend` runs on the
best frames of each night, logs to `iris.log`, appends to
`<dso_dir>/optics_trend.json`. ~75 s at the very end of the shutdown, after the
roof is closed, wrapped so it can never affect the sequence.

**2. Accumulate a baseline — IN PROGRESS.** Several weeks across varied
conditions. The measured night-to-night scatter under comparable conditions is
13% (`edge_excess`) to ~50% (the rest), so that is the floor a detector must
beat. The question for step 4 is whether the residual scatter is *random* — in
which case more nights and a MAD band handle it — or whether it still tracks the
weather, in which case these metrics need another pass before anything is built
on them. Only nightly data answers that.

**3. Record optical epochs.** Log collimation, spacing changes, camera rotation.
Then "historical" means "since the last change", slow drift stays visible instead
of being absorbed by a rolling baseline, and — the real point — you gain ground
truth to validate the detector against.

**4. The comparison command.** Robust control chart: median + MAD over the
epoch's nights, flag at k·MAD, grouped **by filter** (filters focus differently
and give different star populations — the same confound that made O-III nights
read 0.6" worse than Ha in the seeing analysis). Preferably grouped by target
too.

### Validating it

A change detector nobody has seen fire is not a detector. The cheap positive
control: introduce a small known perturbation — a deliberate defocus offset, or
a collimation tweak you then undo — shoot a short run, and measure how far the
metric moves relative to the night-to-night band. That calibrates sensitivity in
units worth having ("this would catch a collimation shift of X"). Any known past
maintenance date is a natural experiment for the same purpose.

### Why not build the command first

Thresholds set before the baseline exists are guesses, and a change detector that
cries wolf gets ignored — at which point it is worse than nothing, because it
occupies the slot where a working one would go. Logging first costs a nightly 75
seconds and buys the data to set them from.

---

## Known confounds

- **Filter.** Each night here is single-filter, so filter and night are
  perfectly confounded. Baselines must be per-filter.
- **Airmass / atmospheric dispersion.** Without an ADC, stars elongate along the
  zenith direction at low altitude. sh2-92 is shot near the meridian so it has
  not bitten yet, but cross-target comparison would inherit it. This shows up in
  `uniform_fraction`, not the sweet spot.
- **Focus.** Defocus raises FWHM symmetrically and leaves the shape metrics
  alone — which is exactly what makes the shape metrics the right thing to
  trend. `FOCPOS`/`FOCTEMP` are in the FITS headers if a covariate is wanted.
- **Bad frames.** A frame in the 08-03 set measured 0.22" FWHM, which is
  physically impossible. The nightly aggregate takes medians over several frames
  and applies a sanity floor so one bad frame cannot poison a night.
