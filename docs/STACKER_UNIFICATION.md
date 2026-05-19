# Stacker Unification Plan

Paused on 2026-05-19 pending real-DSO validation on the observatory machine.

## Why

We have two stacking pipelines:

1. **`stacker.stack()`** — the real one: load → calibrate → measure FWHM → reject blurry frames (≥1.5× median) → register w/ QA (≥10 matched stars, ≤2 px median residual) → nan-aware combine (SIGMA_CLIP_FWHM with MAD-based std + 1/FWHM² weighting) → coverage crop (≥80% of frames) → sky-median fill.

2. **`stacker.convergence_curve()`** — operates on **pre-loaded** raw frames with no calibration and no registration. It builds its "golden" reference as `arr.mean(axis=0)` and computes RMSE of Fibonacci-spaced subset means against that. Used by the `snr` command, `fits_processing/convergence.py`, and an internal call inside `stack_directory` in `stacker.py:817`.

The consequence: the `snr` curve's RMSE is dominated by **misregistration**, not stacking convergence. The "golden" reference is itself smeared because frames aren't aligned. Subset RMSEs are inflated for the same reason.

## The plan

Factor `stack()` into two reusable pieces and reuse them in `convergence_curve()`:

```
_prepare_cube(paths, …)              → (cube, accepted, fwhm_values)
_combine(cube, method, fwhms, sigma) → ndarray
```

- `_prepare_cube` does load + calibrate + measure FWHM + reject blurry + register.
- `_combine` is the per-method combine step (nan-aware MEAN / MEDIAN / SIGMA_CLIP / FWHM_WEIGHTED / SIGMA_CLIP_FWHM).

Then:
- `stack()` becomes `_prepare_cube → _combine → coverage crop + sky-median fill`.
- `convergence_curve(paths, …)` calls `_prepare_cube` once, then for each Fibonacci k samples k indices from the cube and runs `_combine` on that subview. RMSE is computed against the full-cube golden (also from `_combine`).

## Call sites to update

Three places currently preload frames and pass them to `convergence_curve`. After the refactor they pass paths:

- `cmd_processing/super_user_commands.py:1490` (`_snr_run`)
- `fits_processing/convergence.py:139` (background convergence cache)
- `stacking/stacker.py:817` (internal call in `stack_directory`)

## Testing

**A — Synthetic self-test in `stacker.py`.** A `--self-test` block that builds an in-memory cube (sky + Gaussian stars + injected hot pixels + per-frame NaN footprints), runs each combiner, and asserts: hot pixel rejected (≤ 1.1× sky), star centre preserved, no NaN in output, coverage crop reduces shape when frames don't fully overlap.

**B — Real-data regression.** On the observatory machine, before applying the refactor:

1. Pick a DSO with many frames and varied seeing.
2. Run `stack <dso>` — save the JPG and the FWHM / eccentricity / star count metrics line.
3. Run `snr <dso>` — save the convergence plot.

After the refactor, run both again on the same DSO and compare:

- Stack FWHM / star count: same or better.
- snr convergence curve: similar shape, but lower absolute RMSE (golden is now registered/calibrated/weighted, so it stops measuring misregistration).

## State at pause

- Branch: `main`.
- Last commit: `c942455` ("Add SIGMA_CLIP_FWHM method and make it the stack default").
- Version: `2026.5.19.1`.
- No uncommitted code changes related to this refactor.
