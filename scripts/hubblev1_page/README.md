# Hubble V1 light-curve page

The scripts that built `~/Desktop/hubble_v1_lightcurve.html` on 2026-11-12,
copied from a session scratchpad so the page can be rebuilt when the
rising-branch nights (2026-11-15 to 11-18) arrive. The plan is
`docs/HUBBLE_V1_ARTICLE_PLAN.md`; its step 0 turns this folder into one
build command and removes the hard-coded paths below.

Every script takes the work directory as its first argument and reads and
writes there. Order: `lc_fit.py` (the PC's `local/lightcurves/hubblev1.json`
copied to `<dir>/pc/hubblev1.json` -> `lc_model.json`), `charts.py`
(`lc_time.svg`, `lc_fold.svg`), `stack_v1.py` (all L frames ->
`hubblev1_L_raw.npy` + `.json`; spawn pool, keep the `__main__` guard),
`denoise_runner.py` (`hubblev1_L_den.npy`), `solve.py` (plate solve of the
stack against Gaia DR3 -> `stack_solution.json`), `comps.py` +
`comps_chart.py` (comparison-star constancy -> `comps.json`, `comps.svg`,
`comps_stats.json`), `images.py` (`field_v1.jpg`, `crop_v1.jpg`,
`m31_footprint.jpg` from the SkyView DSS2 red cutout at `dl/m31_c.jpg`,
`geometry.json`), `build_html.py <dir> <out.html>`.

Hard-coded for now: the frame root `~/Desktop/Targets/hubblev1/cdk17`
(`stack_v1.py`, `comps.py`, `solve.py`), the repo root and the denoiser
checkpoint (`denoise_runner.py`, `lc_fit.py`, `stack_v1.py`). The JSON
files kept here are the 2026-11-12 results, for reference.
