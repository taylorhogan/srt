# A stellar parallax from Iris (2026-11-20)

The site has an asteroid (Megaira) and a Cepheid distance (Hubble's V1). Parallax is
the rung under the Cepheid, the one Bessel climbed first in 1838, and nothing on it
is borrowed: the star's yearly ellipse against the background is measured here, and
the distance follows from geometry alone. This document is the plan, the star choice
and what was built. Code: `photometry/parallax.py`; chat `parallax`; tests
`tests/test_parallax.py`.

## Choosing the stars

For eighteen nearby stars (every well-known one north of −10) a script computed,
night by night for the next thirteen months, the minutes each spends above this site's
tree line (`configs/my.hrz`) in astronomical darkness, the parallax factors at those
times, and the formal error of a parallax fitted jointly with a proper motion to one
point per observable night (the leverage this site's sampling gives, before brightness
is considered). Site 41.81 N, 72.83 W; camera 0.26"/px, 42' x 28'.

| star | V | Gaia G | parallax mas | nights / 13 mo | months seen | σ(π)/σ(night) | note |
|---|---|---|---|---|---|---|---|
| **Ross 248** | 12.3 | 10.4 | 316 | 217 | Nov-Jan, Jul-Dec | 0.095 | chosen |
| **Wolf 359** | 13.5 | 11.0 | 415 | 197 | Nov-Apr | 0.115 | chosen |
| Kruger 60 | 9.8 | | 249 | 218 | Nov-Dec, Jun-Dec | 0.088 | close binary, 44 yr orbit: the photocentre moves |
| Groombridge 34 | 8.1 | 7.2 | 281 | 215 | Nov-Jan, Jul-Dec | 0.099 | 1 s frames |
| GJ 412 A | 8.8 | | 206 | 213 | Nov-May | 0.093 | |
| Lalande 21185 | 7.5 | 6.5 | 393 | 212 | Nov-May | 0.095 | 0.5 s frames, thin field |
| EV Lac | 10.2 | | 198 | 215 | Nov-Dec, Jul-Dec | 0.091 | |
| Stein 2051 A | 11.0 | | 181 | 223 | Sep-Mar | 0.122 | |
| Teegarden's | 15.1 | | 261 | 207 | Aug-Feb | 0.124 | faint, 60 s frames would do |
| Groombridge 1830 | 6.4 | | 109 | 203 | Nov-May | 0.096 | 7"/yr motion, small parallax |
| Luyten's (GJ 273) | 9.9 | | 263 | 192 | Oct-Mar | 0.143 | low in the south |
| Van Maanen's | 12.4 | | 232 | 186 | Jul-Dec | 0.114 | white dwarf |
| 61 Cygni A | 5.2 | 4.8 | 286 | 184 | Nov, Jun-Nov | 0.110 | Bessel's star; saturates in 0.1 s |
| AD Leo | 9.4 | | 201 | 206 | Nov-Apr | 0.106 | |
| Ross 128 | 11.1 | | 296 | 173 | Dec-May | 0.136 | on the equator |
| Gliese 687 | 9.2 | | 220 | 246 | Feb-Oct | 0.125 | not until February |
| Struve 2398 A | 8.9 | | 283 | 202 | Apr-Oct | 0.191 | not until April |
| Barnard's | 9.5 | 8.2 | 547 | 129 | Mar-Jul | 0.866 | one side of the ellipse only |

The leverage is nearly the same for every star visible now (the sampling, not the
star, sets it), so the choice came down to the camera and the story:

* **Brightness.** The light-curve tool measured that a G 13.5 star saturates this
  camera at 300 s in L. A target must be unsaturated, and the reference stars must be
  deep in the SAME frames, so a target of G 10-11 is the sweet spot: 10-15 s frames in R
  hold it at a third of full well and still reach G 16 references at a flux SNR of 15.
  A V 7 star needs sub-second frames with only the brightest references in them; 61
  Cygni at G 4.8 saturates in a tenth of a second.
* **R, not L.** Differential colour refraction is the classic parallax systematic: a
  red dwarf and bluer references refract differently, by tens of milliarcseconds at
  airmass 1.3 across the L band, and it correlates with the hour angle and so with the
  season. R is narrower and the targets are bright in it. The fit also carries an
  optional refraction term as a check.
* **References.** Ross 248 sits at galactic latitude −17: 822 Gaia DR3 stars to G 17.5
  within 15'. Wolf 359 at +56 has 126, enough for a second-order plate most nights.
* **Season.** Ross 248 is overhead now (transits 2 degrees from the zenith) and its
  parallax factor peaks mid-December; the other side of the ellipse comes in July to
  September mornings. Wolf 359 is a morning star now, an evening star by March, and gone
  in May; its factor peaks late November. Between them the sky is covered until the
  first one closes its year.
* **Story.** Ross 248 (10.3 ly) will be the nearest star to the Sun in about 36,000
  years and Voyager 2 passes it in 40,000; Wolf 359 (7.9 ly) is the fifth nearest system
  and the largest parallax this site can reach.

Barnard's Star (the largest parallax of all, 10.4"/yr of motion, 40 pixels) rises in
March; from this site only one side of its ellipse is seen, so its parallax separates
from its motion to about 5 percent at best. It goes in the queue in March for the
motion, with `parallax add barnard`. The catalogue (`configs/parallax_stars.json`)
also carries 61 Cygni, Lalande 21185 and Groombridge 34 as reserves with their
numbers.

Gaia DR3 reference catalogues for the four fields with refs were pulled on 2026-11-20
into `configs/parallax_refs/` (the ESA archive closes 25 Nov to 2 Dec for DR4), so the
measurement needs no network: the frames, the files, ASTAP.

## The queue

```
parallax add ross248
parallax add wolf359
```

Each writes a row like the Cepheid monitors': `"parallax": true`, `parallax_star`,
`filter_plan {"R": 1}`, `exposure_s {"R": 10}` (15 for Wolf 359), `max_hours 0.25`,
priority 50, `obj_type "star"`, and ra/dec at the star's position TODAY (Gaia's 2016
place moved by the proper motion; Barnard's has moved 1.9' since). The planner treats
a capped row as it does the Cepheids: never a main slot, a 15-minute block in a gap or
at an edge of the night, pinned by its priority, at most two such rows reserved a
night (`slot_plan.plan_slots`). The sequence generator sizes the block's frames to the
cap and the entry's exposure. The filter advisor, auto-stop, the registry, the audit,
the target board and the convergence refresh all read `parallax` as they read
`lightcurve`: a monitor, judged only by its own record.

**Cost.** About 40 frames of 10 s (Ross 248) and 30 of 15 s (Wolf 359) a night, 120 MB
each: 8-9 GB a clear night for the two, roughly half a terabyte over the year. Raw
lights are never pruned; the disk floors on both machines stand.

## The measurement

Per frame (`parallax.measure_frame`): ASTAP solves the frame (the light-curve tool's
`solve_frame`); the Gaia references within 10' of the target are placed where they are
TONIGHT (each one's proper motion and parallax applied from 2016.0), centroided in a
12 px window with a local sky annulus, and those with a flux SNR above 10, a peak above
5 sigma, unsaturated, and within 4 px of their predicted place are fitted pixel to
tangent plane, weighted by SNR, affine with fewer than 25 references and second order
from 25, with 3-sigma clipping. The target's centroid goes through that plate. The
tangent point is the target's own Gaia position at 2016.0 with NO motion applied, so
each frame yields the target's displacement from where Gaia saw it, in arcseconds east
and north. A frame whose target peak reaches 60000 ADU is dropped as saturated and
counted.

Per night (`nightly_epoch`): the 3-sigma-clipped mean of the frames, error the scatter
over root n with a 3 mas floor, plus the mean tan z, parallactic angle and airmass.

Over the year (`fit_motion`): weighted least squares for offset, proper motion and
parallax, the parallax factors from the Earth's barycentric position (astropy's
built-in ephemeris); covariance scaled by the reduced chi-square when above one. With
15 nights a second fit adds one differential-refraction term along the parallactic
direction and the report shows both. `coverage` calls the parallax measured when the
nights span 365 days, there are 30 of them, and the error is under 5 percent of the
value; auto-stop retires the row on that and nothing else (`auto_stop.parallax_coverage`,
`parallax.parallax_verdict`).

Validated on synthetic frames (`test_a_planted_year_of_frames_reads_back_the_parallax`):
60 references with their own motions, a planted 316.5 mas parallax and Ross 248's
proper motion, pointing and rotation wandering night to night; the fit returns the
planted parallax within its error.

**Expected precision.** One frame's plate is 10-20 mas with forty references; a night
of 30-40 frames gives a few mas, which the 3 mas floor acknowledges as the plate's
systematic level. Against a 316 or 415 mas parallax that is a one-percent measurement
by the time the ellipse closes, if the systematics behave; the chi-square and the
refraction-term fit are how we will know.

**What can go wrong.** (1) Proper motion and parallax are correlated when a star is
seen one season only: Wolf 359's Nov-Apr run separates them by March, Ross 248's Nov-Jan
run does not until July. The reports say the error honestly; the picture of the
ellipse with the motion removed is the figure. (2) Both targets are flare stars; a flare
changes the brightness, not the centroid, and a saturated flare frame is dropped.
(3) Focus and temperature change the plate scale between nights; the per-frame plate
absorbs that. (4) A reference list without proper motions (the VizieR fallback) is
wrong by the references' own motions, a few mas a year; the committed files have them.
(5) Gaia DR4 (2 December 2026) will move every reference by its new solution; the
files stay DR3 until replaced on purpose, and the comparison is with DR3 either way.

## What happens next

* **This week:** the owner queues both; the first nights give positions and the plot
  draws two points. From five nights the fit runs; the parallax error will be large
  and the report says so.
* **By March:** Wolf 359's factor has swung from +1 to near 0 and the ellipse separates
  from the line; the first parallax with an honest error.
* **March:** `parallax add barnard` for the proper motion.
* **July-September:** Ross 248's other side.
* **November 2027:** Ross 248 completes; auto-stop retires it; the article: three
  stars' distances measured from the back garden with the first rung's own method,
  beside Gaia's answers.

## Where things are

`photometry/parallax.py` (module), `configs/parallax_stars.json` (catalogue),
`configs/parallax_refs/*.json` (Gaia DR3 references), `local/parallax/<key>.json` and
`.png` (records, fit, plot, on the machine that measures), `tests/test_parallax.py`,
`docs/PARALLAX_PLAN.md` (this). The candidate computation lives in the session's
scratch (`scratchpad/plx/choose.py`) and its table above.
