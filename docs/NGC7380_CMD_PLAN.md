# NGC 7380: colour–magnitude diagram and Hα-excess census

Status: PLAN, written 2026-09-12 after the first SHO night on ngc7380
(2026-09-11: 22 × 300 s Hα, 15 × O-III, 7 × S-II at gain 0). Nothing built yet.

## Why this target

NGC 7380 is the young open cluster inside the Wizard nebula (Sh2-142). It is
the opposite end of stellar evolution from the two clusters the `hr` command
has been run on so far:

| | M13 / M92 | NGC 7380 |
|---|---|---|
| age | ~12 Gyr | ~4 Myr |
| what the CMD shows | turnoff, giant branch, horizontal branch | vertical upper main sequence, no turnoff, pre-main-sequence stars peeling off to the red below ~15 mag |
| age indicator | turnoff | pre-main-sequence turn-on |
| reddening | small | E(B−V) ≈ 0.6, uneven across the field |
| distance | halo | ≈ 2.6 kpc in the Perseus arm |

The brightest member is DH Cephei, an O-star binary that ionises the nebula.
The cluster has a known population of accreting pre-main-sequence stars
(classical T Tauri stars), identified in the literature by Hα emission and
infrared excess. That population is what the second part of this plan goes
after, and tonight's Hα frames are the first half of the data for it.

## Part 1: Gaia-calibrated CMD (existing tooling, new frames)

`hr ngc7380 B R` already does: stack each filter, plate-solve (ASTAP),
detect + aperture-photometer (`sep`, local background), cross-match filters,
Gaia DR3 cone query, per-filter zero point (B→BP, R→RP), member selection by
proper motion + parallax, plot with the Gaia field CMD behind.

**Frames to shoot:** two sets per filter, at gain 0. Moon-tolerant: the
targets are stars, not nebula, so this fits a night that is useless for
narrowband. Point at the cluster (22h47m16s +58°07'30"), which is what the
sequence generator does by default.

| set | purpose | B | R |
|---|---|---|---|
| deep | pre-main-sequence stars (the age), 15–18 mag, B extinguished ~2.7 mag | 10 × 300 s | 10 × 300 s |
| short | O/B members (the reddening anchor), 8.6–11 mag | 10 × 30 s | 10 × 30 s |

Why two sets: the M13/M92 `hr` runs used 300 s B/R and show essentially no
saturated stars, so the 300 s saturation limit in R is near 11 mag. That was
harmless for globulars (only the giant-branch tip is lost) but here the top
5–10 members, DH Cep included, define the upper main sequence. Shorter subs
alone would protect them at the cost of the faint end: a 300 s R frame's sky
is only ~60 ADU above the pedestal, i.e. still near read-noise limited, so
depth lost to short subs is not bought back by more frames. `hr` takes one
set at a time; run it on the deep set first, and merge the two per star as a
small addition alongside Part 2.

**What to expect:**
- Membership will work. Parallax ≈ 0.4 mas is resolved by Gaia to G ≈ 17,
  and the cluster proper motion is distinct from the Perseus-arm field.
  Field contamination is heavy at b ≈ +1°, so the member cut is what makes the
  diagram legible.
- The member sequence sits well to the red of the Gaia field locus. That
  offset is the reddening. It is uneven (nebula), so the sequence will be
  thicker than M13's. Real, not a defect.
- B loses ≈ 2.7 mag to extinction, so the pre-main-sequence stars that carry
  the age are faint in B. Reaching them is the reason for the 300 s set; a
  token few frames gets only the OB stars.
- Gaia calibration stops at 19 mag (`gaia_mag_limit`), which is roughly where
  the useful faint members run out anyway.
- Stars on the bright nebular rim will scatter; members in the clear part of
  the field are the clean sample.

**Deliverable:** the CMD JPEG the command already produces, plus a short
science note for the lab site (needs an Abstract paragraph stating what it
FOUND, per the site convention).

## Part 2: Hα-excess census (small extension, reuses tonight's Hα)

**Method.** Plot (R − Hα) against (B − R). Ordinary photospheres form a
narrow locus in this plane; stars with Hα in emission sit ABOVE it (more
Hα flux than a photosphere of that colour). This is the standard wide-field
survey technique for finding T Tauri stars in young clusters, and its key
property is that R − Hα is almost insensitive to reddening because the two
bands are so close in wavelength, whereas B − R is not. So the excess stands
out even through 2 mag of uneven extinction.

**Calibration.** Hα has no Gaia counterpart, so its zero point is
instrumental. The zero-excess locus is defined empirically from the
non-emission majority of the field stars, fitted as a function of B − R, and
excess is measured as the vertical distance above that locus in units of the
photometric error. Threshold for a candidate: excess > 3σ, and then require
Gaia membership so field Hα emitters (Be stars, nearby M dwarfs) drop out.

**Data.** Hα: tonight's 22 × 300 s (already on disk). B and R: the deep
set from Part 1 (the short set only matters for members that saturate). All three stacks must be registered to ONE shared reference
so the aperture lands on the same star in each (the `process` command already
solves this for LRGB/SHO; reuse `stack_channels`-style shared-reference
stacking rather than three independent stacks).

**Photometry on nebula.** The Hα stack has the nebula in it; the local
background annulus handles smooth nebulosity but not the bright rim. Flag
any star whose annulus RMS is far above the field median and exclude it from
the census rather than let it masquerade as an emitter.

**Deliverables:** (R − Hα) vs (B − R) plot with the locus, the candidate
list (Gaia ID, position, B, R, Hα, excess in σ, member flag), and their
positions marked on the Hα image. Cross-check against SIMBAD: the known
emission-line stars in the cluster should come out as candidates. That is the
validation test.

## Build steps

1. Shoot B and R, deep and short sets (Part 1 needs nothing else).
2. Run `hr ngc7380 B R`. If the member sequence and reddening offset look as
   described, write the science note.
3. Extend `photometry/cmd_diagram.py` (or a sibling `ha_excess.py`) with:
   three-filter shared-reference stacking, instrumental Hα photometry,
   locus fit, excess + σ, membership join, plot + table.
4. Validate on the known emission-line members. If they are not recovered,
   the locus fit or the Hα aperture is wrong; do not publish candidates.
5. Science note for the lab site.

## Risks

- Uneven reddening thickens the CMD sequence; acceptable for the CMD, and
  Part 2 is designed to be insensitive to it.
- Nebular Hα in the aperture inflates R − Hα for stars on the rim. Handled by
  the annulus-RMS flag; it will cost some members near DH Cep.
- Gain change plan ([narrowband high gain] in memory) affects only future
  narrowband subs; tonight's Hα at gain 0 is fine for this, since the stars
  are bright.
