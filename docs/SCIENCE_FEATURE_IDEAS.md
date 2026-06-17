# Science feature ideas

A running backlog of science/photometry features for SRT, building on the
`transit` (exoplanet light curves) and `hr` (Gaia-calibrated colour–magnitude
diagram) features.

These mostly **reuse the existing pipeline** rather than starting fresh:
plate-solving (ASTAP) + Gaia DR3 cross-match + `sep` aperture photometry +
nightly autonomous cadence + LRGB / Ha / SII / OIII filters on a 17" CDK.

Ordered roughly by bang-for-buck.

## Highest leverage (small code, big payoff)

### 1. Gaia cluster-membership filtering — upgrades `hr`
`photometry/cmd_diagram.py::_query_gaia` already pulls `ra/dec/G/BP/RP`. Add
`pmra, pmdec, parallax` to the ADQL and cut field stars by their shared proper
motion + distance → cluster members vs. foreground separated. Turns a smeared
CMD into a clean cluster locus. Near one-line query change + a clustering cut;
sets up isochrone fitting (#6).

### 2. Variable-star light curves + period finding — extends `transit`
The transit search is already time-series differential photometry. Generalise
it: run a **Lomb–Scargle periodogram** to auto-find the period, phase-fold, and
post the curve (RR Lyrae, Cepheids, eclipsing binaries). RR Lyrae tie-in:
**M92's horizontal branch is full of them**. Results are submittable to
**AAVSO** (real citizen science). Likely a new `var <dso>` command.

## Flagship "wow" features

### 3. Transient / supernova discovery via image differencing
Difference tonight's stack of a galaxy against a reference (prior night, or
DSS/PanSTARRS); cross-match any new point source against Gaia to reject known
stars → SN/nova candidate. Reuses stacking + plate-solve + Gaia matching. A 17"
on nightly autonomous cadence is well-suited; amateurs genuinely discover SNe
this way.

### 4. Asteroid astrometry — "what moved through my field tonight?"
Cross-match the plate-solved field against `astroquery` Skybot / JPL Horizons to
flag known minor planets, and detect uncatalogued movers across subs. Outputs
astrometry (reportable to the MPC) or rotation light curves (brightness vs.
time → spin period). Pure reuse of plate-solving + multi-sub registration.

## Uses the narrowband filters for real physics

### 5. Emission-line ratio maps (SII/Ha, OIII/Ha)
Pixel-wise line ratios diagnose nebula physics: shock- vs photo-ionization, find
planetary nebulae and supernova remnants, map ionization structure. A scientific
use of the Ha/SII/OIII filters beyond pretty pictures.

## Natural extension of `hr`

### 6. Isochrone fitting → cluster age
Overlay a MIST/PARSEC isochrone (right metallicity/distance/reddening) and fit
the main-sequence turn-off → an age in Gyr, not just a shape. Pairs with #1
(clean members make the turn-off crisp).

### 7. Extinction / transparency monitoring (near-free)
The photometry already derives a per-night zero-point. Track it vs. airmass →
measure the atmospheric extinction coefficient and sky transparency
automatically. Doubles as observatory-health telemetry and a season-long
atmospheric dataset.

## Suggested order

Quick wins first: **#1 (Gaia membership)** then **#2 (variable-star light
curves)**. **#3 (SN discovery)** is the most exciting flagship if you want a
bigger build.
