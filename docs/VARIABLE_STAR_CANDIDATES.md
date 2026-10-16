# Short-period variable-star candidates (beyond M31 V1)

Written 2026-10-15 from a conversation about Cepheids shorter than Hubble's V1
(P = 31.4 d, monitored as `hubblev1`). Periods, ranges and coordinates are from
AAVSO VSX (VizieR `B/vsx/vsx`), queried 2026-10-15. Always queue with explicit
RA/Dec; never resolve by name (SIMBAD's "M31 V1" is a nova).

## The catch for every Galactic target: they are bright

The light-curve tool's comparison stars are G 13.5–17 because brighter stars
saturate at the template's 300 s (`photometry/lightcurve.py` `COMP_MAG`). A
V ≈ 11 Cepheid needs roughly 10–20 s frames, and the comparison range has to
shift by 2.5 log10(300 / exposure) to match. A full QHY600M frame is ~120 MB,
so short frames fill the disk fast: one point per night only needs ~20–60 frames.

## Ranked candidates

| # | Target | Type | Period | V range | RA, Dec (J2000, deg) | Notes |
|---|---|---|---|---|---|---|
| 1 | **CF Cas** (NGC 7790) | DCEP | 4.87522 d | 10.80–11.47 | 359.57492, +61.22106 | Cluster Cepheid; E = JD 2437022.191 |
| 1 | **CE Cas A** (NGC 7790) | DCEP | 5.1410583 d | 10.63–11.15 | 359.53934, +61.21372 | 2.5″ from CE Cas B: blended in a 3″ aperture unless PSF-fit |
| 1 | **CE Cas B** (NGC 7790) | DCEP | 4.4793014 d | 10.62–11.30 | 359.53796, +61.21361 | |
| 1 | **CG Cas** (Berkeley 58) | DCEP | 4.36554 d | 10.89–11.73 | 0.24688, +60.95897 | ~25′ from CF Cas: aim between them and all four share one field |
| 2 | **DL Cas** (NGC 129) | DCEP | 8.000669 d | 8.63–9.26 | 7.49413, +60.21197 | Cluster Cepheid; needs ~1–3 s frames |
| 3 | **DY Peg** | SX Phe (HADS) | 0.07292611 d = 1.75 h | 10.00–10.56 | 347.2133, +17.21556 | Whole light curve in one night; E = JD 2454710.722 |
| 4 | **M15 RR Lyrae** | RRAB/RRC | 0.3–0.7 d | ~15.5–16 | M15 ≈ 322.49, +12.17 | Dozens in one field; fits the 300 s template as is; sets early by November |
| 5 | Other M31 Cepheids in the hubblev1 field | DCEP | 8–20 d | ~20–21 | — | No extra telescope time, but faint and crowded; needs a published-catalog cross-match |
| — | SZ Cas | DCEPS | 13.639 d | 9.61–10.03 | 36.80738, +59.46061 | Low amplitude (0.4 mag) |

Skip: δ Cep (5.37 d), η Aql, SU Cas. They saturate at any practical exposure,
and no comparison stars of similar brightness fit in the field.

Why NGC 7790 + Berkeley 58 is the top pick: four Cepheids in one pointing, a
full cycle in under 5 days, and two clusters whose main-sequence-fitting
distances can be checked against the period-luminosity distance.

## DY Peg: best 2 hours (centered on transit, 66°)

The period is 1.75 h, so any 2 h window covers a full cycle; only altitude
matters. It stays above ~55° within an hour of transit.

| Date | Transit | Best 2 h | Dark and above 30° |
|---|---|---|---|
| 2026-10-15 | 22:25 EDT | 21:25–23:25 | 19:45–02:40 |
| 2026-10-25 | 21:45 EDT | 20:45–22:45 | 19:30–02:00 |
| 2026-11-05 | 20:00 EST | 19:00–21:00 | 18:20–00:15 |
| 2026-11-20 | 19:00 EST | 18:05–20:05 | 18:05–23:20 |

Moon: full ~Oct 25–26, and up for most of the Nov 20 night. That costs some
noise on a V ≈ 10 star, not a reason to skip a night. The scheduler puts a
`max_hours` target into a gap or at a dusk/dawn edge, not at transit; the
early-evening edge is nearly as good (50–66°).

## Status

- `ngc7790` queued 2026-10-15 as **held** (scheduler only takes `waiting`):
  pointing 359.911, +61.088 (between NGC 7790 and Berkeley 58),
  filter_plan L:2 G:1, `exposure_s` {L: 10, G: 20}, max_hours 0.5, lightcurve
  on, and a `stars` list: CF Cas, CE Cas A+B (one blended aperture at the
  pair's midpoint, no period), CG Cas. Offsets from the pointing are about
  ±10′ east–west and ±8′ north–south, inside the 28′ frame height whichever
  way the camera is turned.
- Built 2026-10-15 (uncommitted at the time of writing):
  1. `exposure_s` in `nina_gen`: written into each named block's
     TakeExposure before its count is sized, the dither trigger rescaled to
     stay ~300 s apart, and checked both before the write and in the
     read-back from disk. Explicit filter plans only; ignored with a warning
     otherwise. Dry run on the real template: L 90 × 10 s, G 22 × 20 s for
     0.5 h.
  2. `lightcurve` comparison range = COMP_MAG shifted by
     2.5 log10(300 / EXPTIME) per frame (10 s: G 9.8–13.3; 20 s: 10.6–14.1).
     Each star gets 25 Gaia comparison stars at both exposures.
  3. Several stars per entry: one plate solve per frame, one post, plot and
     file set per star. hubblev1 re-measured identically through the new
     code (L 19.353 ± 0.033, 12 frames).
- To go live: push → CI → `update`, then set `ngc7790` to `waiting`.
- First night: check the brightest Cepheid's peak ADU on the first frames
  (CF Cas / CE Cas peak near G 10.5, the bright end of the 10 s range), and
  shorten the exposures if any frame clips.
- Darks: not needed for the light curve (raw frames, local annulus); only
  for a stacked picture of the cluster.
