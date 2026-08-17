# Measuring a galaxy rotation curve from Iris

What it would take to repeat Vera Rubin's 1970 result with the CDK17, and read
dark matter out of it. Written 2026-08-17. Nothing here is built; this is a
plan, and the numbers are computed for this telescope at this latitude.

## What she actually measured

Rubin and Kent Ford put a long slit along the major axis of M31 and photographed
the spectrum. Gas on the approaching side blueshifts Hα; the receding side
redshifts it. Reading the line's wavelength as a function of position along the
slit gives orbital speed as a function of radius.

Newtonian gravity makes a hard prediction. Outside most of the visible mass the
enclosed mass stops growing, so speed should fall as `v ∝ r^−1/2` — the reason
Neptune orbits slower than Mercury. It does not happen. The curve goes flat and
stays flat, far beyond where the starlight has faded.

**The result is the gap between two curves, and it is worth naming them
precisely:**

* **A — measured.** Orbital speed against radius from the Hα Doppler shifts.
  Stage 3 below; the spectrograph.
* **B — predicted.** Orbital speed against radius computed from the light:
  `v(r) = √(G·M_lum(<r)/r)`, with `M_lum(<r)` from surface photometry times a
  mass-to-light ratio. Stage 4; the camera already on the telescope.

Same axes, same quantity, two independent instruments. Note that **B is not
`r^−1/2` everywhere** — inside the disk the enclosed mass is still growing, so B
rises along with A, and only outside the light does it peel away. That shared
inner section is what makes the outer divergence hard to blame on calibration or
distance.

## Why the present rig cannot do it

The CDK17 has ample aperture. The problem is that a rotation curve is a set of
Doppler shifts, and 250 km/s at Hα is 5.5 Å — about 0.08% of the wavelength. No
filter resolves that. The light has to be dispersed.

| CDK17 | value | consequence |
|-------|-------|-------------|
| aperture | 432 mm | ample; aperture is not the limit |
| focal length | 2939 mm, f/6.8 | plate scale 70.2″/mm |
| 23 µm slit | 1.61″ on sky | well matched to ~1.7″ seeing |
| ASI432MM | 9 µm pixels | large pixels suit a spectrograph — use this, not the imaging camera |

## The instrument

Two requirements narrow it fast. It must be a **long slit**, so position along
the galaxy survives — a fibre-fed spectrograph averages the whole galaxy into
one spectrum and destroys the measurement. And it needs `R ≥ 4000`, because
below that the entire rotation curve is a couple of resolution elements tall.

| instrument | R | km/s per element | centroided | verdict |
|------------|---|------------------|------------|---------|
| Star Analyser | ~100 | 3000 | 300 | no — classification only |
| Alpy 600 | 600 | 500 | 50 | no — one element is twice the signal |
| Baader DADOS 900 ℓ/mm | 4000 | 75 | ~7 | workable minimum |
| LowSpec 3 (self-built) | 10000 | 30 | ~3 | excellent value if you will build it |
| **Lhires III, 2400 ℓ/mm** | **17000** | **17.6** | **~1.8** | **the recommendation** |

Against a ~250 km/s signal the Lhires gives over a hundred resolution elements
across the curve — a well-sampled measurement, not a marginal detection.

## What to buy

Everything attaches to the existing focuser and uses the ASI432MM as detector.
Prices move; the US dealers are noted because they save an import.

**Rotation curves — the recommendation**

* **Lhires III** — [Shelyak](https://www.shelyak.com/produit/spectroscope-lhires-iii/?lang=en)
  · [Telescopes.net](https://telescopes.net/es0002-lhires-iii-littrow-high-resolution-spectrograph.html)
  · [Astroshop](https://www.astroshop.eu/spectroscopes/shelyak-spectroscope-lhires-iii/p,50969).
  Ships with the 2400 ℓ/mm grating, a 15/19/23/35 µm slit set, a built-in neon
  calibration lamp and a slit-viewing guide port. On this f/6.8 those slits are
  1.05″ / 1.33″ / 1.61″ / 2.46″, so the 23 µm already matches the seeing.
* Optional [photometric slit](https://optcorp.com/products/shelyak-photometric-slit-lhires-iii-spectrograph-se0143)
  if flux-calibrated line strengths are wanted later.

**Budget alternative**

* **DADOS** — [Baader](https://www.baader-planetarium.com/en/instruments/spectroscopy/dados-slit-spectrograph.html)
  · [Alpine Astronomical](https://alpineastro.com/products/dados-slit-spectrograph).
  Ships with a 200 ℓ/mm grating that is **not** sufficient; the
  [900 ℓ/mm](https://www.baader-planetarium.com/en/instruments/spectroscopy/dados-slit-spectrograph/baader-blaze-reflection-gratings-1200-lmm.html)
  is a required extra, not an upgrade. Budget for it with the instrument.

**Do not buy an Alpy 600 for rotation curves**, however often it is recommended
as the beginner's spectrograph. At R≈600 one resolution element is 500 km/s,
twice the whole signal. It is an excellent instrument for the wrong problem.

**For the cluster route instead** (see the last section), the requirement
inverts and the Alpy becomes right:
[Alpy 600](https://www.shelyak.com/produit/spectroscope-alpy-600/?lang=en) ·
[First Light Optics](https://www.firstlightoptics.com/spectroscopy/shelyak-alpy-600-spectrograph.html),
plus its guide module and
[calibration module](https://optcorp.com/products/shelyak-alpy-calibration-module-pf0037)
— the Lhires has that lamp built in, the Alpy does not.
[Alpy user guide (PDF)](https://www.shelyak.com/wp-content/uploads/DC0016B_Doc_Alpy_600_EN-1.pdf).

## Targets worth this latitude

Angular size, surface brightness (the real limit, not total magnitude),
inclination, and strong Hα from star formation to give a bright emission line
rather than a shallow absorption one.

| galaxy | incl. | observed amplitude | max alt. | notes |
|--------|-------|--------------------|----------|-------|
| **M31** | 77° | **253 km/s** | 90° | Rubin's own target, passes overhead, bright HII regions along the major axis. Start here. |
| NGC 7331 | 76° | 238 km/s | 83° | compact and bright — best second target |
| M81 | 59° | 206 km/s | 63° | bright, strong Hα, well placed in spring |
| NGC 2903 | 65° | 168 km/s | 70° | vigorous star formation, good line strength |
| M33 | 54° | 89 km/s | 79° | NGC 604 is superb but the amplitude is small |
| M104 | 84° | 348 km/s | 37° | biggest signal, worst altitude, weak Hα (old stars) |

M31 is the start not only for sentiment: its HII regions are individually bright
enough that you are effectively taking spectra of nebulae rather than of diffuse
disk light, which is what makes the exposures survivable.

## Why inclination decides the target list

A Doppler shift measures only the line-of-sight component. Rotation projects as
`v_obs = v_rot · sin i · cos θ`, θ measured in the disk plane from the major
axis. A **face-on galaxy has `i = 0`, so `v_obs = 0` everywhere** and cannot
yield a rotation curve at all. M51 is magnificent and worthless here.

This is the one failure no amount of aperture or exposure can rescue. A fainter
target is answerable with more hours; a face-on one is not answerable, because
the quantity has no component along the line of sight. **Check inclination
before magnitude.**

Inclination then hurts twice — it scales the signal by `sin i` and amplifies any
error in `i` by `cot i`:

| inclination | signal kept | error from a 5° mistake in i | |
|-------------|-------------|------------------------------|---|
| 90° | 100% | 0% | dust lane, line-of-sight blending |
| **77° (M31)** | **97%** | **2%** | ideal |
| **65°** | **91%** | **4%** | ideal |
| 54° (M33) | 81% | 6% | workable |
| 45° | 71% | 9% | marginal |
| 25° | 42% | 19% | not worth the nights |
| 5° | 9% | 100% | impossible |

The useful band is **55°–80°**, and it stops short of edge-on deliberately: at
90° you see the whole rotation but look through the entire disk, so gas at many
true radii blends into one line profile and "the speed at radius r" stops being
well defined — and the midplane dust hides the inner disk.

Inclination need not be taken on trust. Measure it from your own images via the
apparent axis ratio:

```
cos²i = (q² − q₀²)/(1 − q₀²)      q = b/a measured, q₀ ≈ 0.2 disk thickness
```

| q = b/a | implied i |
|---------|-----------|
| 0.95 | 19° |
| 0.80 | 38° |
| 0.60 | 55° |
| 0.45 | 66° |
| 0.30 | 77° |
| 0.22 | 85° |

**Keep the slit on the major axis.** The `cos θ` term is why position angle
matters: 20° off-axis costs 6% of the signal, which is survivable but is a
systematic that mimics slower rotation — so it biases toward *not* finding dark
matter.

## The programme

Each stage exists to prove the instrument can do the next. Skipping ahead gives
a null result you cannot interpret, because you will not know whether the galaxy
failed or the spectrograph did.

**1. Prove you can measure a velocity.** Spectra of two or three radial-velocity
standard stars; recover their published values. Until you can hit a catalogue
value to better than 10 km/s, nothing downstream means anything.

**2. Recover a known systemic velocity.** M31's bulge sits near −300 km/s — one
of the largest and easiest shifts in the sky, and a strong check that the
wavelength scale and barycentric correction are right. Success: −300 ± 20 km/s
from a single exposure.

**3. Build the rotation curve.** Slit along the major axis; step outward taking
a spectrum at each position, working from the bright inner disk toward the faint
outskirts so that clouds ending a night cost the least valuable points. Measure
the Hα centroid at each radius, convert with
`v = c(λ_obs − λ_rest)/λ_rest`, subtract systemic, divide by `sin i`. Budget
several nights per galaxy — the outer points are the whole result and the
hardest to get.

**4. Weigh the visible matter — with the rig you already have.** Pure imaging.
Photometer the galaxy in a calibrated band, build a surface-brightness profile
along the same axis, convert light to stellar mass. The existing stacking and
photometry pipeline does this.

**Do not assume the mass-to-light ratio — fit it.** Choose the largest M/L that
still lets curve B match curve A across the inner disk (the maximum-disk
approach). You have then been as generous as physically possible to the
ordinary-matter explanation, so any discrepancy left in the outskirts cannot be
dismissed as having undercounted the stars. That is the difference between "I
assumed a number and found dark matter" and "I gave ordinary matter its best
case and it still failed".

**5. Compare.** `M_dyn(<r) = v²r/G`. Report the result as the ratio
`M_dyn(<r)/M_lum(<r)` against radius, not as a vertical gap between lines — near
1 across the inner disk because that is where M/L was fitted, climbing outward,
perhaps to five or ten at the last point reachable. That climb is the
measurement.

## Where this will actually hurt

* **Surface brightness sets the exposure, not magnitude.** Dispersing light
  spreads an already faint object very thin. Expect one to three hours per slit
  position: a season-long project per galaxy, not a night.
* **The barycentric correction is not a detail.** Earth's orbital motion is
  ±30 km/s, over 10% of the signal. Every measurement must be reduced to the
  solar-system barycentre or the curve is quietly wrong.
* **Flexure will bite.** The spectrograph hangs off a moving telescope and its
  internal geometry shifts with orientation. That is what bracketing lamp
  exposures are for; skipping them costs the night.
* **Inclination enters as a divisor.** Everything scales by `1/sin i`, so a
  poorly known inclination propagates into the masses. Quote the uncertainty.

## A cheaper path to the same conclusion

If the spectrograph is a step too far, there is a second route to dark matter
needing far less resolving power — Zwicky's, not Rubin's. Measure redshifts for
a few dozen galaxies in a cluster, take the velocity dispersion, get the virial
mass. Cluster dispersions run to ~1000 km/s, so `R ≈ 1000` suffices and an Alpy
600 is enough instrument.

**Abell 2151**, the Hercules cluster, is already imaged from here. Same target,
much cheaper spectrograph, same conclusion by a different road. The trade is
many faint objects instead of one bright one — but each spectrum is easy, and a
partial result is still a result.
