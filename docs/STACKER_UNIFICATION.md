# Stacker Unification Plan

Status as of 2026-06-15: the convergence path and the quality gate are unified;
the per-method combine step is still duplicated (see "Remaining work").

## Progress

- **2026-05-19** — Paused pending real-DSO validation on the observatory machine.
- **(earlier)** — `convergence_curve()` refactored onto `_prepare_for_convergence()`,
  which loads + measures FWHM + registers frames (streaming, downscaled). The snr
  golden stack is now the registered, FWHM-weighted `SIGMA_CLIP_FWHM` stack, so the
  curve measures stacking convergence rather than misregistration.
- **2026-06-15 (`677f7b7`)** — Frame rejection unified into a single gate,
  `_select_by_quality()`, used by every stacking path (see below). L-frame output
  validated good on the observatory with the 0.5 / 1.25 thresholds.

## Why

We had two stacking pipelines with **divergent frame selection**:

1. **`stacker.stack()`** — the real one: load → calibrate → measure FWHM → reject blurry frames (≥1.5× median) → register w/ QA (≥10 matched stars, ≤2 px median residual) → nan-aware combine (SIGMA_CLIP_FWHM with MAD-based std + 1/FWHM² weighting) → coverage crop (≥80% of frames) → sky-median fill.

2. **`stacker.convergence_curve()`** — originally operated on **pre-loaded** raw frames with no calibration and no registration, building its "golden" as `arr.mean(axis=0)`. Now goes through `_prepare_for_convergence()` (register + downscale) so its RMSE reflects stacking convergence, not misregistration.

The two paths also rejected frames differently — `stack()` had no star-count gate and kept unmeasured frames, `hr` rejected nothing — so "good frame" meant different things in the analysis vs the output. That is now fixed.

## Unified quality gate (done)

`_select_by_quality(paths, fwhm_values, star_counts, max_fwhm, max_fwhm_multiplier, min_star_fraction)`
is the single place frame rejection lives. A sub is rejected when, relative to the
session medians of the *measured* frames, it is:

- too blurry — FWHM > `max_fwhm` (or > `max_fwhm_multiplier` × median FWHM), or
- too sparse — stars < `min_star_fraction` × median star count, or
- unmeasurable — FWHM 0.0 while peers measured fine (noise-dominated).

Defaults are `max_fwhm_multiplier=1.25`, `min_star_fraction=0.5`. The gate self-tunes
to the night's seeing, no-ops when no frame has a measurable FWHM (hosts without
photutils) or all criteria are disabled, and never rejects every frame.

Callers, all routed through it:

- `stack()` — measures FWHM + star count in one detection pass, then calls the gate.
- `_prepare_for_convergence()` — same, feeding `convergence_curve()` (snr),
  `fits_processing/convergence.py`, the L-files convergence, and `transit`.
- `stack_directory`, `LiveStacker` — call `stack()`, so they inherit the gate.

## Remaining work

The per-method **combine** step is still implemented separately in `stack()`
(tiled, memmap-backed) and `convergence_curve()` (in-memory subsets). The plan
below (factoring out `_prepare_cube` / `_combine`) is the deeper refactor that
remains; the frame-selection half is already shared via `_select_by_quality`.

## The original plan

Factor `stack()` into two reusable pieces and reuse them in `convergence_curve()`:

```
_prepare_cube(paths, …)              → (cube, accepted, fwhm_values)
_combine(cube, method, fwhms, sigma) → ndarray
```

- `_prepare_cube` does load + calibrate + measure FWHM + reject blurry + register.
  (Frame rejection is already shared — `_select_by_quality`; the load/calibrate/
  register half is what still needs factoring out.)
- `_combine` is the per-method combine step (nan-aware MEAN / MEDIAN / SIGMA_CLIP / FWHM_WEIGHTED / SIGMA_CLIP_FWHM).

Then:
- `stack()` becomes `_prepare_cube → _combine → coverage crop + sky-median fill`.
- `convergence_curve(paths, …)` calls `_prepare_cube` once, then for each Fibonacci k samples k indices from the cube and runs `_combine` on that subview. RMSE is computed against the full-cube golden (also from `_combine`).

## Call sites

The three convergence call sites already pass **paths** (the preload-and-pass step
is done):

- `cmd_processing/super_user_commands.py` (`_snr_run`)
- `fits_processing/convergence.py` (background convergence cache)
- `stacking/stacker.py` (internal call in `stack_directory`)

## Testing

**A — Synthetic self-test in `stacker.py`.** A `--self-test` block that builds an in-memory cube (sky + Gaussian stars + injected hot pixels + per-frame NaN footprints), runs each combiner, and asserts: hot pixel rejected (≤ 1.1× sky), star centre preserved, no NaN in output, coverage crop reduces shape when frames don't fully overlap.

**B — Real-data regression.** On the observatory machine, before applying the refactor:

1. Pick a DSO with many frames and varied seeing.
2. Run `stack <dso>` — save the JPG and the FWHM / eccentricity / star count metrics line.
3. Run `snr <dso>` — save the convergence plot.

After the refactor, run both again on the same DSO and compare:

- Stack FWHM / star count: same or better.
- snr convergence curve: similar shape, but lower absolute RMSE (golden is now registered/calibrated/weighted, so it stops measuring misregistration).

## State

- Branch: `main`.
- Quality gate unified in `677f7b7` ("stacker: unify frame rejection across all
  stacking paths"). Validated on real L frames on the observatory.
- Done: convergence path refactored onto `_prepare_for_convergence`; frame
  rejection shared via `_select_by_quality`.
- Remaining: factor the load/calibrate/register + per-method combine into
  `_prepare_cube` / `_combine` and reuse in `convergence_curve`.
