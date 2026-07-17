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

## Decisions already locked

- **All field stars** (memory is not the constraint).
- **Wide net:** general variability **and** single-transit dips —
  Lomb–Scargle periodogram + Box Least Squares (BLS) per star.
- **Prototype-first**, because the Windows session could not verify on the Spark.
  Validate correctness on CPU on real frames before any GPU port or deployment.

## Two-phase plan

### Phase 0 — prototype & validate (do this first, correctness over speed)

Build the pipeline and **prove it recovers known variables** before it runs
unattended.

**Validation target: m13 or m92** (globular clusters). They are dense fields
*full of catalogued RR Lyrae* (periods ~0.3–0.7 d, amplitudes ~0.5–1 mag), so
they are ground truth: if the pipeline recovers their known RR Lyrae from the
user's own frames, it works. The Clement catalogue lists the known variables per
cluster for cross-check.

Pipeline stages (per target field, one night to start):
1. **Reference frame + source list.** Pick the best frame (most stars / lowest
   FWHM). Detect sources (`sep` or photutils `DAOStarFinder`) → master (x,y) list.
2. **Register every frame** to the reference (`astroalign` — already a repo dep),
   or plate-solve per frame (ASTAP) and match by sky coords.
3. **Photometry** for every master source in every frame (photutils aperture, or
   PSF in crowded cores) → raw flux time series. Read timestamps from FITS
   `DATE-OBS`; compute airmass per frame.
4. **Differential detrending.** Normalise each star against an ensemble of stable
   comparison stars (or SysRem) to strip atmospheric/systematic trends.
5. **Search each light curve:** Lomb–Scargle (`astropy.timeseries.LombScargle`)
   for periodic signals + BLS (`astropy.timeseries.BoxLeastSquares`) for
   transit-shaped dips. Also simple variability stats (RMS-vs-magnitude outliers).
6. **Rank & report:** flag high LS power / low false-alarm-probability and high
   BLS SDE with box shape. Cross-match top hits against known variables (VSX /
   Clement) for the m13/m92 validation. Output light-curve plots + a table.

**Reuse from the repo (verify signatures on the Spark, don't assume):**
- `fits_processing/fitsfwhm.py` — `_fit_stars(path)` → `(data, [(x,y,fwhm_px,ecc,angle)…])`
  and `calculate_fwhm(path,…)` → `(fwhm_px, fwhm_arcsec, star_count, ecc)`.
  Good for the reference-frame quality pick and source detection.
- `fits_processing/sky_brightness.py` — `measure_sky(path, arcsec_per_pixel=0.26)`
  for background/quality per frame.
- `fits_processing/session_stats.py`, `frame_watcher.py` — existing per-frame /
  per-session aggregation patterns and the `frame_stats.json` cache shape.
- The **existing single-transit scoring** (the HAT-P-32b work) — grep the repo
  for the BLS / box-score logic and reuse/extend its scoring for the transit half.
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

### Phase 1 — GPU-scale & deploy

Once Phase 0 is correct: port the two hot loops to the GPU — **batch photometry**
(all stars × all frames) and the **period search** (LS/BLS across thousands of
light curves) — with CuPy/torch; 128 GB holds every light curve at once. Wrap as
the morning job chained after the copy. Post results to the webchat over the
Tailnet (HTTP/MQTT to the Windows social server), or write result files + plots.

## First step on the Spark

`git pull`, then run Phase 0 on a night of **m13** (or **m92**) frames from
`<FRAME_ROOT>/m13/cdk17/<date>/LIGHT/`, build the all-star light curves, run
LS+BLS, and check whether known RR Lyrae fall out. Report what it finds before
touching the GPU.
