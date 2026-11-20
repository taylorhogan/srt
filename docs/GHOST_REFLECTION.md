# The reflection ghost (NGC 1055, 2026-11-19/20)

A note for whoever picks this up on the observatory computer. Written on the Spark
(Claude session 01VMwWTjqasp1xY8ihRzw1ka); nothing here has moved hardware.

## What was found

The four-night learned LRGB of NGC 1055 (2026-11-15..18) carries three coloured
line features the owner took for asteroid trails. Measured on the linear stacks and
on every one of the 101 frames:

1. **Upper left, a rainbow line pointing at the bright yellow-white star beside the
   galaxy: a diffraction spike.** The spike's length grows with wavelength, so the R
   arm reaches farther than G and B and the colour balance paints the arm red, green,
   blue along its length. Identical on all four nights. Real starlight; not removable
   at the source.
2. **Lower right, a long thin greenish line (stack ~(8023,5672)-(8118,6152)) and a
   fainter parallel one right of the galaxy (~(6307,4689)-(6397,5049)): reflection
   ghosts.** Per night the light is one short dash on that line, in L, R, G and B
   alike at the same place, so it does not move with time (464 Megaira would have
   moved 123 px/h). It moves with the POINTING: on the one night the pointing jumped
   mid-run the dash appears twice, and across 18 strong frame pairs the dash moves
   1.8-2.1 times the pointing change, along one line, 5 px rms across it. An image
   inverted through a centre behaves exactly so (in sky-aligned frames a ghost at
   -m x the star's offset moves (1+m) x the pointing). m is about 0.8-1.1. The
   four nights' dashes, each smeared by the night's drift, add up to the line.
3. **The real Megaira remnant** is a faint blue/red smudge beside the bright blue star
   lower right of the galaxy, 535" from the galaxy where JPL puts it on 11-15. The
   clip removed most of it, as the Megaira note predicted; more frames finish it.

## Why the ghost is rare

The ghost is about 2e-5 of the star's light. No star ON the sensor explains both
ghosts (no centre and magnification puts both opposite any of the field's bright
stars), so the source is off the chip. Both ghosts sit lower right of the field
centre, on the side away from the upper left; the upper left is west (the Megaira
motion runs that way), and **delta Ceti, V 4.1, sits 35' west-south-west of the field
centre, just beyond the long edge.** Two ghosts at different distances on the far
side are two reflecting surfaces returning one star. A 4th-magnitude star within 40'
of a field centre is roughly a 1-in-50 field, and a 6th-magnitude star gives a ghost
six times fainter: 85 Ha frames of Melotte 15 and 16 L + 47 Ha frames of IC 405 (AE
Aur, V 6) show nothing measurable (one weak candidate at IC 405's top edge, not
interpretable).

Why the stacker's clip never removes it: per frame the ghost is ~60 ADU, two sigma a
pixel, and it sits in every frame of a night, so a pixel on the dash carries it in
10-16 of 37 frames. Measured on synthetic cubes, the clip leaves a two-sigma outlier
in full (4 frames at 60 ADU leave 5 ADU), which is the 4-16 ADU excess measured along
the streak. More nights make the line longer, not fainter.

## The test (written, not yet run)

`scripts/ghost_test_sequence.py` writes a N.I.N.A sequence from the normal template:
alpha Trianguli (V 3.4, 78 deg up at transit ~21:30 local in late November) centred
twice (L,R,G then L,B; the template carries three filters a container), then 8' east
and west and 6' north and south in L; 4 x 30 s per filter; the template's plate-solved
Center before each container, so the pointing is solved, not assumed. It carries the
template's roof open, park and flats like any night's file: **the conductor's roof and
mount rules apply; the owner loads and starts it.**

- Written for the evening of 2026-11-20, slots 19:30-20:42, and **copied to the PC:
  `C:\Users\iriso\Documents\N.I.N.A\Sequences\ghost_test_2026-11-20.json`** (442,643
  bytes). Regenerate for another evening with `--date YYYY-MM-DD` (`--start HH:MM`).
- By hand at the centred position, if time allows: focuser 1 mm in, two frames, 1 mm
  out, two frames. A ghost from a curved surface defocuses at a different rate than
  the star; one from a flat plate near the sensor defocuses with it.
- Worth adding before the run (not yet in the script): three positions with the star
  10, 20 and 30 arcminutes OUTSIDE the long edge, since the real culprit is a star off
  the chip; that measures how the ghost fades as the star leaves the field.

## Reading the frames

`scripts/ghost_test_measure.py <LIGHT folder> [--pattern '*_L_*'] [--json out.json]`
finds the saturated test star and every compact source in each frame, drops what
moves with the field (one-to-one with the pointing) or not at all (hot pixels), and
fits `ghost = A + M.star` in detector pixels over the frames. `M = -m I` is an
inversion through the fixed point `C = (I-M)^-1 A` with magnification m; M near the
identity with an offset is a tilted flat; anything else names a tilted or decentred
element. Validated on a synthetic field with a planted ghost (C to 1 px, m exact;
`tests/test_ghost_test.py`). It drowns in nebula knots (IC 405), so it is for the
sparse test field.

What the numbers mean afterwards: m with the centre tells which surface (an in-focus
inverted ghost at m near 1 is the sensor's own reflection returned by a curved lens
surface in the corrector/flattener; a flat window or filter cannot invert); a centre
off the sensor centre means that surface is decentred or tilted relative to the
camera. Then: check camera tilt/corrector spacing; let the planner keep bright stars'
ghosts off the subject (ghost position is now predictable); and the pipeline fix.

## The pipeline fix (proposed, not built)

A per-night mask pass in the stacker: `stacker.stack(keep_frames_dir=)` keeps the
registered, levelled frames; stack each night on its own, find the compact light one
night's stack has where the other nights have none, blank those pixels (NaN) in that
night's frames, then the normal combine (`_combine_tile` already treats NaN as
non-coverage). Removes the ghost and the Megaira remnant alike; the spike stays, being
the star's own light. About a day with tests. Owner has not yet chosen between this, a
one-off hand mask for NGC 1055, and leaving the picture.

## Where things are

- Scratch analysis (Spark, this session): `scratchpad/ast/` (per-night stacks
  `nights_A.png`, `nights_UL.png`, the pair search `ghost_pairs*.log`, the per-frame
  spot fit `spots.json`), `scratchpad/megaira/` (the note's figures).
- Scripts (committed with this note): `scripts/ghost_test_sequence.py`,
  `scripts/ghost_test_measure.py`, `tests/test_ghost_test.py`.
- The sequence on the PC as above; on the Spark `local/ghost_test_2026-11-20.json`.
- The site note on Megaira: irisscience.org/#asteroid-megaira. Its "more nights will
  bury the asteroid" still holds; the lines the owner saw afterwards are not it.
