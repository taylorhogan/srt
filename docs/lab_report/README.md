# Iris Observatory — Lab Report

A running record of the deep-sky objects imaged by **Iris**, an autonomous
backyard observatory, and the science done with the data.

Iris runs unattended: it picks a target each night, opens the roof, images,
closes up and reports. Everything here comes from its own frames — no survey
data is used except where explicitly noted for calibration or cross-matching.

## Instrument

| | |
|---|---|
| Telescope | PlaneWave CDK17 (17″ corrected Dall-Kirkham) |
| Camera | QHY600M (monochrome, IMX455 full-frame) |
| Mount | PlaneWave, PWI4-controlled |
| Plate scale | 0.26 ″/pixel |
| Site | Connecticut, USA — approx. 100 m elevation |
| Control software | [SRT](../../README.md) (this repository) + N.I.N.A |

Sensor is cooled and calibrated against measured BIAS/DARK sets; sky background
is reported in ADU/s above a temperature-keyed bias pedestal rather than raw
pixel value.

## Entries

Objects Iris has imaged. Entries are written up as the analysis is done, so
"imaged" and "written up" are not the same thing.

| Object | Type | Entry |
|---|---|---|
| M92 | Globular cluster | _not yet written_ |
| M13 | Globular cluster | _not yet written_ |
| Sh2-92 | Emission nebula | _not yet written_ |
| NGC 5907 | Edge-on spiral | _not yet written_ |
| Abell 2151 | Galaxy cluster (Hercules) | _not yet written_ |
| Abell 2218 | Galaxy cluster | _not yet written_ |
| HAT-P-32 | Exoplanet host (transit) | _not yet written_ |
| HAT-P-16 | Exoplanet host (transit) | _not yet written_ |

## Conventions

- **One file per object**, in this directory. Add the entry, then link it in the
  table above.
- **Every number traceable to a frame.** State the filter, exposure, sub count
  and night. If a value is modelled or estimated rather than measured, say so.
- **Negative results are kept.** A search that found nothing is a result; the
  transient search on NGC 5907 turned up only galaxy dipole artefacts and that
  is recorded rather than dropped.
- **Figures** are produced by this repository (`photometry/cmd_diagram.py`,
  `fits_processing/session_stats.py`, `transit_search/`, `transient_search/`).
  Note which script and which night produced each one.

See [TEMPLATE.md](TEMPLATE.md) for the entry skeleton.
