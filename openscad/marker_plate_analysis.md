# Why the marker plate can be printed in PLA

Written 2026-08-17, answering "should I print this in PLA?" for
`marker_plate.scad`. Recorded because the instinctive answer was wrong and the
arithmetic is short enough that nobody should have to redo it.

## The worry that did not survive

The reflex objection to PLA is creep: it softens near 55–60 °C, an observatory
closed in summer sun gets hot, and this part is a cantilever under permanent
load. Worse, the failure mode of creep is **loss of flatness** — which is
exactly what killed the paper marker it replaces, where a buckled tag still
produced a detectable quadrilateral but undecodable bits.

That reasoning is sound and the conclusion is still wrong, because creep only
matters at an appreciable fraction of yield stress and this part is nowhere
near it.

## Load case

Worst case: the plate cantilevered horizontally off the arm, so its whole
weight acts at half the plate width from the arm root.

| quantity | value |
|---|---|
| volume (plate + rim + ribs + arm + gussets) | ~58 cm³ |
| PLA density | 1.24 g/cm³ |
| mass | ~72 g |
| weight | 0.71 N |
| moment arm (half of 101.6 mm) | 50.8 mm |
| bending moment at the arm root | 0.036 N·m |
| arm section (25 × 6 mm), Z = bh²/6 | 150 mm³ |
| **bending stress** | **0.24 MPa** |

## Result

| material | yield | stress as % of yield | safety factor |
|---|---|---|---|
| PLA | ~50 MPa | 0.48 % | **208×** |
| PETG | ~45 MPa | 0.53 % | 187× |
| ASA | ~40 MPa | 0.60 % | 167× |

The part is **stiffness-driven, not strength-driven**. The rim, ribs and
gussets exist to stop it flexing and twisting, not to stop it breaking, which
is why it ends up so far from its strength limit. At 0.5 % of yield, PLA will
not creep measurably at observatory temperatures.

## Is the conclusion robust to a bad mass estimate?

The volume figure is a rough sum, not a slicer number, so it is worth asking
how wrong it could be before the answer changes. Creep becomes a design concern
somewhere around 10–20 % of yield for a semi-crystalline thermoplastic under
sustained load at elevated temperature.

Reaching 10 % of yield from 0.48 % needs the mass to be **21× higher** — 1.5 kg
instead of 72 g. Even a threefold error in the estimate leaves a safety factor
near 70. The conclusion does not depend on the estimate being good.

## Thermal expansion

PLA CTE ≈ 68 µm/m/K. Over the 101.6 mm plate across a 40 K swing:

```
0.276 mm  ->  0.35 px at the camera's 1.25 px/mm
```

Against 0.12 px of corner-measurement noise and a 69 px (3°) tolerance, that is
irrelevant. Expansion is not a reason to prefer one material over another here.

## What this analysis does NOT cover

Two real risks remain, neither of them structural, and neither addressed by
changing material:

* **Warping during printing.** A 4″ flat plate is the classic warp geometry,
  and warp destroys the one property the part exists to provide. Print face
  down with a brim on a levelled bed, and check the result against a known flat
  surface before mounting. A plate that rocks is worse than the paper it
  replaces.
* **UV over years.** The roof opens, so the part sees direct sun. PLA
  embrittles faster than PETG, and much faster than ASA. This is a
  multi-year cosmetic-then-structural concern, not a first-season one, and the
  part is a two-hour reprint from parameters already in the file.

## Reproducing this

```python
inch = 25.4
plate, t = 4*inch, 3.0
vol_cm3 = 58.0                      # plate + rim + ribs + arm + gussets
mass_kg = vol_cm3 * 1.24 / 1000
W = mass_kg * 9.81                  # N
M = W * (plate/2) / 1000            # N.m
Z = 25.0 * 6.0**2 / 6               # mm^3, arm section modulus
sigma = M * 1000 / Z                # MPa
print(sigma, "MPa;  PLA safety factor", 50/sigma)
```

If the arm or plate dimensions change in `marker_plate.scad`, only `Z` and the
moment arm move, and both appear above.
