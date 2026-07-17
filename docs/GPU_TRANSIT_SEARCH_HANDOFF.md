# Handoff — GPU blind transit / variable-star search (#3)

**For:** a fresh Claude Code session running on the **DGX Spark** (ARM64 / DGX-OS).
**From:** the Windows observatory session. Originating conversation:
`https://claude.ai/code/session_016DjatY7MTdE1UQpi6yH4EQ`
Living plan (all 5 GPU ideas, ranked): `https://claude.ai/code/artifact/99df0392-ed2d-4368-bc8d-ed166cca4440`

This doc is self-contained — you should be able to start Phase 0 from it alone.

---

## Why this exists

The user wants to use their DGX Spark for scientific work on the observatory's
own data. We ranked five GPU projects; the user's priority order is:

1. **#3 Blind transit / variable-star search  ← THIS DOC, build first**
2. #4 Sky/cloud CNN monitor (independent, later)
3. #1 Real-bogus SN classifier (reuses #3's machinery, later)

Parked: #2 pipeline acceleration (speed only), #5 roof-audio embeddings.

**#3 in one sentence:** the existing pipeline scores *one* transit target at a
time (HAT-P-32b validated); do GPU batch photometry over **every field star ×
every frame** to blind-search the whole field for transits *and* variables.

> **CORRECTION (2026-07-17, Spark session).** The sentence above is wrong, and
> so is the Phase 0 build plan that follows from it. `transit_search/transit.py`
> (1475 lines) **already does all-field-stars × all-frames** on CPU: reference
> stack, DAOStarFinder over the whole field (no bright-subset cap), astroalign
> registration, aperture photometry, differential detrending against a
> lowest-RMS comparison ensemble, BLS + a single-transit matched filter,
> permutation false-alarm probability, ASTAP plate solve + Gaia cross-match,
> and candidate plots. **Do not rebuild it.** Phase 0 is a *validate and
> fill-the-gaps* job, not a build job. See "Phase 0 — actual state" below.

## Hardware / architecture (decided)

- The Spark is a **separate ARM/Linux box** from the Windows observatory PC.
  These analyses are an **offline node**, never in the live control loop.
- **Frames arrive via an AM crontab job** that copies each night's new frames to
  the Spark every morning. The frame tree mirrors the Windows layout:
  `<FRAME_ROOT>/<object>/cdk17/<YYYY-MM-DD>/LIGHT/*.fits`
  (confirm `<FRAME_ROOT>` on the Spark — it's the copy job's destination).
- **The user wants the search to run right after the copy finishes** — chain it:
  `copy_new_frames && python run_transit_search.py`, or have the search watch for
  a "copy done" sentinel the copy job writes. Settle this once the job exists.
- **128 GB unified memory** ⇒ no VRAM ceiling. A whole night of QHY600M frames
  (~123 MB each, 9600×6422×16-bit) fits in memory ⇒ **all-field-stars from the
  start**, no bright-subset compromise. Caveat: LPDDR5x ~273 GB/s (not HBM) — the
  win is *capacity*, not peak streaming bandwidth. Plenty for this data volume.
- Software is fine on aarch64: CUDA, CuPy, PyTorch (NVIDIA aarch64 builds),
  astropy/photutils/astroalign/sep, ASTAP (ARM Linux). Only the **science**
  modules of the repo are needed — `fits_processing/`, `iris_astronomy/`. The
  hardware/control modules (kasa, pwi4, NINA, sentry vision) are Windows-only and
  irrelevant here; don't try to import them.

## Scope: this is a variable-star project, not a planet-discovery project

The transit scorer and the variable search share all their machinery, so #3
"covers" exoplanets mechanically — but blind planet *discovery* is the weakest
thing this system can do, and the reasons are physics, not code. Keep the two
modes separate so neither is judged by the other's yardstick:

- **Blind search (the real deliverable) → variables.** Every field has
  variables, and the search rides along on whatever Iris imaged that night.
  Genuine discovery is possible here.
- **Exoplanets → targeted follow-up, not blind search.** The fields Iris images
  are chosen because they're *pretty* (galaxies, clusters, nebulae) — which are
  the wrong fields for planets. Three compounding problems:
  - **Globulars are planet-poor.** Gilliland et al. (2000) watched ~34,000
    stars in 47 Tuc for 8.3 d with *HST* precision, expected ~17 hot Jupiters
    at solar-neighbourhood occurrence, and found **zero** (≥1 order of
    magnitude rarer; low metallicity + crowding). They did recover ~75
    variables — which tells you which half of this project a globular serves.
  - **Per-night odds are ~10⁻⁵/star.** ~0.5–1 % of stars host a hot Jupiter ×
    ~10 % transit geometrically × the few-percent chance a ~2.5 h transit lands
    in a ~3–5 h window. Then cut to stars that are both millimag-precise *and*
    dwarfs (a Jupiter across a giant gives an invisible depth). Order: many
    hundreds of nights per expected detection.
  - **One night can only yield a single-transit candidate**, never a
    confirmation — as `field_z_alert`'s own config comment already says.
- **Where the scope earns real credit: TESS/TOI follow-up via ExoFOP.** Known
  ephemeris ⇒ you point at a transit you know is happening. A 17" CDK at
  0.26 "/px is genuinely competitive, and amateur photometry materially
  contributes (ruling out eclipsing-binary false positives on TESS candidates).
  Same code path as the HAT-P-32b run that's already validated.

## Decisions already locked

- **All field stars** (memory is not the constraint).
- **Wide net:** general variability **and** single-transit dips —
  Lomb–Scargle periodogram + Box Least Squares (BLS) per star.
- **Prototype-first**, because the Windows session could not verify on the Spark.
  Validate correctness on CPU on real frames before any GPU port or deployment.

## Two-phase plan

### Phase 0 — actual state (validate + fill gaps; the pipeline already exists)

Don't build a second pipeline — run `transit_search.transit.run_transit_search(
dso_name, filter_name, image_dir, output_plot_path, progress_cb, cancel_cb)` and
find out where it falls short. Confirmed gaps as of 2026-07-17:

- **No Lomb–Scargle anywhere in the repo.** The variability half of the "wide
  net" is genuinely missing. Every scorer (`_max_dip_sigma`,
  `_single_transit_score`, `_run_bls`) hunts **dips** — an RR Lyrae that ramps
  *upward* will not rank, no matter how good the photometry is. This is the
  main thing Phase 0 has to add.
- **BLS can't reach RR Lyrae on a single night.** `min_bls_cycles: 2` caps the
  searched period at baseline/2 ≈ 1.7 h for a 3.4 h night; RR Lyrae are 7–17 h.
  Not a bug — just means BLS is the transit half only.
- **`min_baseline_days_for_bls: 1.0` is dead config** — defined in
  `config_public.py`, never read by `transit.py`. Either wire it or delete it.
- **ASTAP is not installed on the Spark**, so `identify_candidates` / the Gaia
  cross-match can't run here. Needed for any catalogue comparison. Install the
  ARM Linux build and set `hardware.astap_exe`.
- **NOT a bug** (checked, don't "fix" it): `outlier_high_sigma: 5.0` blanks
  upward excursions, but its threshold is built from each star's *own* MAD, so
  it scales with that star's variability and won't erase a coherent RR Lyrae
  ramp — only isolated single-point spikes.

**FIXED 2026-07-17 — sep pixel-stack overflow on dense fields.** Every frame
failed to register on m13 with `TypeError: Input type for source not supported`.
That message is astroalign swallowing sep's real error: `internal pixel buffer
full: the limit of 300000 active object pixels ... was reached`. sep's default
pixstack is sized for sparse fields; a globular on a 61 MP frame blows through
it. `stacker._count_sources` swallows the same error via `except: return 0`,
silently breaking the reference-frame pick too. Fix is a module-level
`sep.set_extract_pixstack(5_000_000)` in `stacking/stacker.py`. **This is why it
was never caught: HAT-P-32b is a sparse field — the validation targets are the
pathological case.** Any dense-field work (globulars, rich Milky Way fields)
depends on this.

**Validation target — prefer m92, not m13.** The claim above that m13 is "full
of catalogued RR Lyrae" is wrong. M13 is famously RR Lyrae-*poor*: the Clement
catalogue lists ~75 variables but only **~9–11 RR Lyrae**, mostly low-amplitude
RRc, and the literature calls its population "not particularly rich". M92 has
**~17–21 RR Lyrae** out of ~21 variables — far denser ground truth. The frames
on the Spark agree:

| night | R frames | baseline | note |
|---|---|---|---|
| m13 2026-06-19 | 20 | 3.41 h | == `min_frames` (20): **one** registration drop fails the run |
| m92 2026-06-16 | 30 | 4.84 h | best available; **use this** |
| m92 2026-06-15 | 14 | — | below `min_frames` |
| m92 2026-06-18 | 12 | — | below `min_frames` |

**Set the pass/fail criterion honestly.** A 3–5 h baseline **cannot** recover a
0.3–0.7 d period — you cover well under half a cycle. Phase 0 passes if known
variables surface as *significant variability with the right ramp/rise shape*,
NOT if periods are recovered. Period recovery needs multi-night stitching with
per-night zero-points.

Pipeline stages (per target field, one night to start) — and where each already
lives in `transit_search/transit.py`:

| # | stage | already implemented as | gap |
|---|---|---|---|
| 1 | Reference frame + source list | `_build_reference_stack`, `_detect_reference_stars` (DAOStarFinder, all stars) | — |
| 2 | Register every frame | `_register_only` → `stacker._register_frames` (astroalign) | — |
| 3 | Photometry, all stars × all frames | `_photometry_one_frame` (aperture + annulus, NaN-masked); times via `_obs_time_mjd` | — |
| 4 | Differential detrending | `_differential_normalize` (lowest-RMS comparison ensemble) + common-mode correction | no SysRem |
| 5 | Search each light curve | `_run_bls`, `_single_transit_score`, `_max_dip_sigma` | **no Lomb–Scargle, no RMS-vs-mag variability stat** |
| 6 | Rank & report | `_plot_top_candidates`, `_plate_solve_wcs` (ASTAP) + Gaia match, `save_transits` → `local/transits.json`; permutation FAP | needs VSX/Clement cross-match; ASTAP absent on Spark |

So Phase 0's real work is **row 5** (add LS + variability stats and a
non-dip-shaped ranking path) and **row 6** (catalogue cross-match).

**Reuse from the repo (verify signatures on the Spark, don't assume):**
- `fits_processing/fitsfwhm.py` — `_fit_stars(path)` → `(data, [(x,y,fwhm_px,ecc,angle)…])`
  and `calculate_fwhm(path,…)` → `(fwhm_px, fwhm_arcsec, star_count, ecc)`.
  Good for the reference-frame quality pick and source detection.
- `fits_processing/sky_brightness.py` — `measure_sky(path, arcsec_per_pixel=0.26)`
  for background/quality per frame.
- `fits_processing/session_stats.py`, `frame_watcher.py` — existing per-frame /
  per-session aggregation patterns and the `frame_stats.json` cache shape.
- **`transit_search/transit.py` — this IS the pipeline** (the HAT-P-32b work).
  No grepping needed: `run_transit_search()` is the entry point and already runs
  stages 1–6 above over every field star. Extend it; don't reimplement it. Its
  knobs live under `cfg["transit"]` in `configs/config_public.py`.
- `stacking/stacker.py` — `astroalign` registration + quality selection patterns.
- Plate scale is **0.26 arcsec/px** (config `nina.arc_sec_per_pixel`); QHY600M,
  3.76 µm pixels, 2939 mm focal length; field ~9600×6422 px.

**Known caveats to design around:**
- **Crowding:** globular *cores* defeat aperture photometry — use the outskirts
  or PSF fitting for the validation.
- **Short baseline:** one night ≈ a few hours → great for RR Lyrae (sub-day) and
  single transits, not long-period variables. Multi-night stitching needs careful
  per-night zero-points.
- **Single-band, differential:** photometry is relative within one filter — fine
  for variability; no absolute calibration needed for detection.

### Phase 0 — validation results (2026-07-17, DGX Spark)

**PASS.** Run on m92, all 56 R frames across 3 nights (baseline 3.16 d):
12,743 stars detected, ~3,800 well-sampled light curves; a scratchpad
Lomb–Scargle pass (`ls_analysis.py`) + Gaia/VSX cross-match recovered **20
catalogued variables**, including 8+ RR Lyrae with periods to 0.3–5 %:
V1673 Her (RRab, 0.6732 d cat vs 0.671 d ours), V1669 Her (RRc, 0.3773 vs
0.380), CSS J171714.6+425305 (RRc, 0.2666 vs 0.268), V1670/V1668/V1667/
V1662/V1655 Her (the last an exact +1 c/d alias). Misses were physical:
SX Phe (too small/fast), eclipsing binaries out of window, faint Gaia-only
variables. m13 single-night (20 R frames) ran clean end-to-end but found
nothing — expected (RR-poor cluster, dip-only scorers, 1.7 h BLS ceiling).

Working notes:
- **Plate solve without ASTAP:** astroalign accepts bare point sets — match
  Gaia DR3 (Vizier cone, G<16) tangent-projected at 0.26 "/px against the
  brightest ~120 non-core detections. Field is **mirrored, rotated 67.4°**;
  refined affine residual 0.3 px. See `scratchpad/crossmatch.py`.
- **New gap — saturation veto for candidates.** The m92 top BLS candidate
  (score 39.6, field_z 23.6 → two Pushover alerts) is a **saturation
  artifact**: G=9.8 foreground star (Gaia DR3 1360406946570559232, plx
  3.2 mas) pegged at 64–65.5k ADU all of 06-16; the 16 % "transit" is
  clipped photons tracking seeing. The comparison ensemble excludes the
  brightest 5 % but candidates are never vetoed — add a per-star peak-pixel
  check (reject/flag above ~55k in any frame) before scoring/alerting.
- Multi-night beats single-night more than expected: with 3 nights the LS
  periods genuinely resolve (0.3 % on the best), despite day gaps causing
  ±1 c/d aliases. Core blending makes ~10 apertures echo the same bright
  RRab (P≈0.75 d cluster of "detections" around the core) — PSF photometry
  or a blend-dedup pass is the fix if the core matters.
- Registered-frame star positions differ from raw-frame pixels by up to
  ~60 px across nights — always work in reference-frame coordinates.

### Phase 1 — GPU-scale & deploy

Once Phase 0 is correct: port the two hot loops to the GPU — **batch photometry**
(all stars × all frames) and the **period search** (LS/BLS across thousands of
light curves) — with CuPy/torch; 128 GB holds every light curve at once. Wrap as
the morning job chained after the copy. Post results to the webchat over the
Tailnet (HTTP/MQTT to the Windows social server), or write result files + plots.

## First step on the Spark

**`<FRAME_ROOT>` is `/home/taylor/Desktop/Targets`** — the destination of
`scripts/sync_nina_targets_to_spark.bsh` (cron, 06:00 daily, rsync from the
Windows box at `iriso@100.95.7.19`). Tree confirmed:
`<FRAME_ROOT>/<object>/cdk17/<YYYY-MM-DD>/LIGHT/*.fits`. Note there are also
legacy `M 13` / `NGC 5033` dirs (space in the name, `calibrated/Light_*`
subdirs); `_find_dso_dir` normalises spaces so both match, but the legacy dirs
contain no `LIGHT/` dir and lose the mtime tie-break, so `m13/` wins. Frames are
uncalibrated 9576×6388 16-bit, 300 s, **no WCS in the headers** (hence ASTAP).

Env on the Spark is ready: py3.13, astropy 7.2, sep 1.4.1, photutils 2.3,
astroalign 2.6.2, numpy 2.4, torch 2.12+cu130. **cupy is NOT installed** (Phase
1). `configs/config_private.py` exists. ASTAP is **not** installed.

Run it like this (venv at `.venv`, activate first):

```python
from pathlib import Path
from transit_search import transit
entry = transit.run_transit_search(
    dso_name="m92", filter_name="R",
    image_dir=Path("/home/taylor/Desktop/Targets"),
    output_plot_path=Path("out/m92_R.png"),
    progress_cb=print,
)
```

Use **m92 2026-06-16** (30 R frames, 4.84 h) — not m13, see above. Registration
is single-threaded astroalign on 61 MP frames: budget ~10+ min for that stage
alone before anything is printed. Report what it finds before touching the GPU.
