# Plan: autonomous stretch (picking stretch parameters for a DSO without a human)

## Context

Every color image today gets its stretch by eye. We render a sweep sheet, the owner picks a panel, then re-render (ngc6946 on 10-16 took two sweeps to get to black=60 soft=0.05 white=99.9). Full autonomy (project 3: auto-publish the best image) cannot happen until a machine can choose a stretch the owner would accept.

**Answer to "photon analysis or neural net?" Mostly measurement, then a small learned correction, not a CNN.**
- Black, white and softening each have a measurable target: where the sky sits against its noise, what clips, and how bright the faint signal lands.
- The part that is taste is a few offsets on top of those measurements. That needs tens of labeled picks, not the thousands a pixel CNN would need, and we have about 9 picks.

**What already exists (built 2026-09-12, 652f8c9, off by default):** `color_process.auto_stretch()` (`stacking/color_process.py:696`).
- It sets per-channel blacks from a binned sky-noise model (`sky_noise_model`, :660). A channel with faint signal gets a margin below sky; a channel without it gets its black at sky, which prevents color casts.
- It solves one softening so that sky + 2 binned σ lands at 0.15 of white, capped so the bright p90 stays at or below 0.90.
- The harness is `scripts/auto_stretch_compare.py`.
- Its own commit records the failures:
  - **No clean sky:** on IC 1396, squid and NGC 5907 the noise model's negative half is nebula too, so the black lands at the binned median.
  - **NGC 6888 hits the softening floor:** its shell sets white 240σ above the surrounding emission.
  - **No white-point rule:** white is still a percentile.
- It is reachable only from the n2n scripts (`--auto-stretch`). `process`, `sweep` and `publish` never call it, and the word `auto` in `process` already means "27-variant sweep".

The plan finishes that work: labels first, then fix the measured rule against them, then wire it in, then learn the owner's taste.

## Phase 0: a label set (the real bottleneck)

1. **Record picks properly.** When `process … reuse <opts>` or `publish N` uses a variant from the latest sweep, write `from_sweep` (the sweep's `time` and file) and `panel` into the `process_log.json` entry (`record_render`, color_process.py:1580; the call sites are super_user_commands.py ~6021/6055/6529). Add a `keep` command, `keep <dso> <recipe>`, which marks the latest render as "this is the one". A pick then becomes a clean label with no timestamp guessing.
2. **Back-fill the about 9 existing picks** from the logs:
   - sh2-92: 55/99.5/0.025 and 0.05;
   - m33 HALRGB: 55/99.9 wb=stars;
   - ngc7320: 60/99.9 wb=stars;
   - bubble and ic1396 publishes;
   - ngc7380's hand HOO stretch (in `HAND` in auto_stretch_compare.py);
   - ngc6946: 60/99.9/0.05 wb=stars.

   They go into `local/stretch_labels.json`.
3. **Store each label in units that carry across targets, not percentiles.** Re-measure every pick on its cached channels (`load_cached_channels`, :975) and record:
   - black as σ below sky per channel;
   - white as the fraction of extended signal that clips, and the fraction of stars that clip;
   - softening as the output level of sky + 2 binned σ, and the output level of bright p90.

   p60 on a galaxy and p60 on a nebula mean nothing alike; these numbers do.

Deliverable: `scripts/stretch_labels.py` (back-fill + re-measure), with about 10 labels from the start.

## Phase 1: measured stretch v2, scored against the labels

Changes to `auto_stretch` in `stacking/color_process.py`. Each one is driven by a known failure:

1. **Sky when there is no clean sky.**
   - Mask signal before modelling noise: a sep object mask, dilated, plus the existing star mask from `rgb_stars`.
   - Take the sky from the darkest quartile of mesh tiles, not the whole-field negative half.
   - If fewer than N clean tiles remain, return `confidence: "no clean sky"` and fall back to a sky-anchored percentile. Saying "I don't know" here is what later lets autonomy refuse to publish.
2. **White point rule** (new; today it is the percentile in `shared_white`, :548).
   - Choose white so that at most X% of the extended signal clips: the galaxy core or the nebula's bright rim.
   - Exclude stars from this rule, because stars may clip. Today's first ngc6946 render failed exactly this test: the core was blown at p99.
3. **Softening floor (NGC 6888).** When the bright cap and the faint target conflict, which happens when the dynamic range is too high for one asinh curve, pin the faint target and let the brightest structure compress into the top 10%. Record that the compromise was made.
4. **Optional target-type prior.** Read galaxy, emission or cluster from the instruction or SIMBAD type and apply a small per-type default margin. Use it only if the label residuals show a type split.

**Scoring harness:** extend `scripts/auto_stretch_compare.py` into `scripts/stretch_benchmark.py`.
- For every labeled target, compare auto against the hand pick in output space: sky level, faint-band level, bright p90 and clipped fraction, each as a difference in output units.
- Report a pass/fail per target (all four within tolerance) and render a side-by-side sheet.
- **Leave one out:** a tuning constant may not be chosen on the target being scored.

## Phase 2: wire it in

- **`process <dso> <recipe> stretch=auto`** (the word `auto` is taken). It computes v2 and renders with `compose_prepared` (:935) using the measured blacks, white and softening, and logs every measured number and the confidence in the render entry.
- **The sweep centers on auto.** `auto` sweeps a ±1 step around v2's values instead of the fixed 45/55/65 grid, and panel #1 is always "auto". Every sweep then also tells us whether the owner agrees with auto, which is a free label.
- `describe_options` prints the measured values (for example "black sky−0.4σ, white clips 0.1% of extended signal, soft 0.031") so the reply shows why the stretch looks the way it does.
- **Raw and denoised pairs share one measurement,** made on the raw stack. Same rule as wb_factors: the 10-09 lesson applies to the stretch too.

## Phase 3: learn the owner's taste (only after about 30 labels)

- Model: fit ridge regression from image statistics to the label offsets from v2 (Δmargin, Δwhite fraction, Δfaint output).
  - Inputs: faint and bright fractions, dynamic range, recipe, target type, star density, plus SNR, so that noisy stacks get a darker black.
  - It runs on CPU, sits in a JSON file, and every coefficient can be read.
- Gate: adopt it only if leave-one-out error beats v2 alone.
- A CNN on pixels is only worth revisiting with hundreds of labels and residuals the statistics cannot explain. State that as a non-goal for now.

## Phase 4: hand-off to autonomy (project 3)

A publish candidate must pass automatic checks on the rendered picture:
- the background is not crushed: the fraction of exact zeros in sky tiles is under a threshold;
- no color cast: the sky tiles' chroma is near neutral;
- the extended signal does not clip;
- confidence is not "no clean sky".

If any check fails, it goes to the owner for review instead of being published. The checks become `color_process.render_qc()`, used by the future auto-publish.

## Critical files

- `stacking/color_process.py`: `auto_stretch`, `sky_noise_model`, `shared_white`, `compose_prepared`, `sweep`, `record_render`, `describe_options`.
- `cmd_processing/super_user_commands.py`: `process_cmd`/`_process_run` (:5771), `publish`, the new `keep`.
- `scripts/auto_stretch_compare.py` evolves into `stretch_benchmark.py`; new `scripts/stretch_labels.py`.
- Reuse: `rgb_stars` star mask, `sky_anchored_blacks`, `deblur.sky_sigma`, `_remove_gradient`.

## Verification

- **Unit tests (`tests/test_auto_stretch.py`):** synthetic seeded fields in the style of test_halrgb.py:
  - clean sky + galaxy (black within 0.1σ of truth, core not clipped);
  - field-filling nebula with no clean sky (returns low confidence, does not put black at the median);
  - high-dynamic-range shell (the softening compromise is reported);
  - stars excluded from the white rule.
- **Benchmark:** `stretch_benchmark.py` over all cached targets (abell2151, m33, ngc2146, ngc5907, ngc7320, ngc6946, bubble, ic1396, ngc7380, sh2-92, squid, trunk). Done when v2 passes on at least 80% of labeled targets and flags the rest as low confidence rather than getting them wrong.
- **Owner blind A/B:** for 6 targets, show auto and the hand pick unlabeled, in random order. Phase 2 ships as the default sweep center when the owner can't reliably tell them apart, or prefers auto, on at least 4 of 6.
- **Live:** run `process ngc6946 halrgb reuse stretch=auto` and compare it with today's 60/99.9/0.05.
