# <Object> — <common name>

> One or two sentences: what this object is, and why it was worth pointing a
> 17-inch at it. Lead with the interesting thing, not the catalogue entry.

| | |
|---|---|
| Catalogue | |
| Type | |
| Constellation | |
| Distance | |
| Apparent magnitude | |
| Angular size | |

## Observations

| Night | Filter | Subs | Exposure | Integration | Median FWHM | Notes |
|---|---|---|---|---|---|---|
| YYYY-MM-DD | | | | | | |

Conditions worth recording: sky background (ADU/s above pedestal), seeing proxy
(850 hPa wind — NOT the jet stream, see below), dew-point spread, measured PM2.5
if elevated.

Measured 2026-08-04 over 9 nights of sh2-92, against median FWHM: 850 hPa wind
ρ=+0.87, dew-point spread ρ=+0.88 (humid nights are the sharp ones), and the
250 hPa jet ρ=+0.33 — no relationship. The wind correlation decays monotonically
with altitude, so the turbulence that matters here is low-level, as expected for
a site inside the boundary layer rather than above it. Cloud cover ρ=-0.07: it
costs frames, not sharpness. Re-run `scripts/seeing_vs_weather.py` as nights
accumulate — nine is a thin calibration and wind and humidity are entangled
(ρ=-0.65), so which one is causal is still open.

## Image

![<object>](<path or URL>)

Acquisition and processing in one line: total integration, filters combined,
registration and stacking method.

## Analysis

What was measured and how. Name the script that produced each figure and the
night(s) it ran on. Include the method, not just the conclusion — someone should
be able to redo it from this section.

## Result

What the data actually shows. Quantitative where possible, with uncertainties.

## Limitations

What would change the answer: undersampling, missing calibration frames, a
single night's data, an unvalidated threshold, a modelled rather than measured
input. Be specific — this section is what makes the rest trustworthy.

---
*Imaged and analysed by [Iris](../../README.md), an autonomous observatory.*
