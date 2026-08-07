# Transient search: run-to-run non-determinism — handoff

**Status: cause IDENTIFIED and FIXED 2026-08-07 on the Spark.** Two defects,
one triggering the other. The amplifier is fixed in `_robust_scale_bg`; the
trigger (astroalign's unseeded RANSAC) is deliberately left alone, because the
new estimator absorbs it. The acceptance test is now reproducible — but the
recovery floor moved from 400/800 ADU to a stable 1600 ADU. Read
"Validation" before quoting a limiting magnitude: the old floor was flattering
itself, and the new one is limited by something else that is worth fixing next.

## The answer, in one paragraph

`astroalign._ransac` shuffles its candidate triangle list with an **unseeded**
`np.random.default_rng()` (`astroalign.py:610`) and keeps the *first* seed
triangle that clears `min_matches`. For a marginal frame, different seeds
converge to genuinely different transforms — so one frame of 46 lands on one of
two footprints at random. That changes the covered-pixel count by ~15k of 61M.
`_robust_scale_bg` then draws its subsample with
`np.random.default_rng(0).choice(xs.size, 200_000)`: the *seed* is fixed but the
*draw* is a function of `xs.size`, so a coverage change does not perturb the
subsample, it **replaces** it (measured overlap: 0.33%). And that estimator has
a **73% peak-to-peak spread in `s` across subsamples of identical data**,
because three rounds of 3σ clipping strip out the bright pixels that are the
only leverage on a slope. So a coin flip in registration redraws `s` from a
wildly unstable distribution, and `s` sets the difference image, its background
RMS, and every threshold downstream.

The registration coin flip is the *trigger*. The scale fit is the *amplifier*,
and it is much the larger defect: it was converting a 1e-6 relative change into
a 3% one, and would do the same for any other perturbation.

## The symptom

The same transient search, on the same frames, with no code change between
invocations, produced materially different results:

| run | raw residuals | difference-image RMS |
|-----|---------------|----------------------|
| 1   | 824           | 1.566897             |
| 2   | 893           | 1.365186             |

An 8% change in detections and **13% in background RMS**. Downstream, the
injection test's measured 50% recovery floor moved between runs:

| run | 400 ADU | 800 ADU | floor    |
|-----|---------|---------|----------|
| A   | 3/3     | 3/3     | 400 ADU  |
| B   | 0/3     | 3/3     | 800 ADU  |

A factor of two in the quoted sensitivity. This blocks publishing a limiting
magnitude: there is no stable number to convert into one. The monotonicity
assertion held in both runs, which is why the tests assert curve *shape* rather
than a pinned value (see `utils/injection_checks.py`).

## Dataset

`ngc5907`, filter `R` — 46 LIGHT frames over 5 nights. The pipeline splits it
into a 36-frame template (4 nights, 2026-06-02 → 06-08) and a 10-frame science
night (06-09); baseline 7.0 d. About 5.6 GB at ~122 MB/frame.

On the Spark: `/home/taylor/Desktop/Targets/ngc5907/cdk17/*/LIGHT/`, already
populated by `scripts/sync_nina_targets_to_spark.bsh`. `configs/config_public.py`
maps host `spark-3129` to that root, so no path edits are needed.

**Peak memory ~11 GB** (each frame is 9576x6388 float32 ≈ 245 MB once
registered). This is why the test kept dying on the observatory PC — run one
process at a time and do not run two searches in one interpreter.

## Established facts (measured, trust these)

1. **astroalign is not bit-reproducible for a minority of frames.** Repeating
   `find_transform` on the same inputs: 2 of 13 frames differed, by up to
   `9.1e-13` in the transform matrix. A *robust* pair repeats exactly, so a
   single-pair test is not sufficient — that mistake was made here first.
   **Amended 2026-08-07:** the 1e-13 figure badly understates it. On the full
   46-frame set the differing frames move by *far* more than a last bit —
   frame 014's registered footprint changed by 14,825 finite pixels between
   runs, and its max by 2,542 ADU. Those are two different transforms, not one
   transform ± rounding.
2. ~~Its RNG is not controlled by `np.random.seed` in current scikit-image~~
   **Wrong location.** The RNG is not scikit-image's at all — astroalign 2.6.2
   ships its own RANSAC, and `astroalign.py:610` calls
   `_np.random.default_rng().shuffle(all_idxs)` with no seed, reseeding from OS
   entropy on every call. Nothing in this tree can seed it. Note the loop below
   that shuffle tries *every* index before raising `MaxIterError`, so whether a
   frame registers at all is order-independent — only *which* transform it
   converges to is random. That is why this never showed up as a frame failing.
3. `ois` is **not installed**, so `_psf_match_and_subtract` takes the scipy
   Gaussian path. Do not chase Alard–Lupton.
4. ~~`_robust_scale_bg` subsamples with `np.random.default_rng(0)` — **seeded**,
   not a source.~~ **Wrong conclusion from a true premise.** It is seeded, but
   `choice(xs.size, 200_000)` is a function of `xs.size`, and `xs.size` is the
   covered-pixel count. Any coverage change replaces the whole subsample —
   measured overlap between draws at `N` and `N-14825`: **650/200000 (0.33%)**.
   This is the amplifier; see below.
5. `_fill` and `_gaussian_match` are deterministic given their inputs.
6. `_frame_fwhm` returns a **median** over detected sources — a discrete
   selection, so it *can* convert a last-bit input change into a finite jump.
   It did not do so here: it was never the first divergence in any run pair.
7. No `sep` pixstack overflow/retry occurred (checked in `iris.log`), so the
   adaptive-threshold path is not involved.
8. **The accepted frame set does NOT change.** `n_accepted=46` in all four
   runs. The "leading hypothesis" below — a marginal frame passing QA in one
   run and failing in another — is **false** for this dataset. The frames are
   all accepted; one of them is simply registered differently.

## The wrong diagnosis (do not repeat)

It was concluded that astroalign's 1e-13 jitter propagates through registration
into `_frame_fwhm`, whose median flips discretely, changing the PSF-matching
kernel width and hence the whole difference image.

**That chain is plausible but was never demonstrated, and the arithmetic does
not support it.** A direct test perturbed the science image by `1e-7` — six
orders of magnitude *larger* than the real jitter — and produced only 633 vs 636
detections. A far larger perturbation gives a far smaller effect than the real
824 vs 893. Something else is producing the large effect.

Critically, **the FWHM was never observed to differ between the two original
runs.** That was inferred backwards from the symptom.

## What the bisection actually measured (2026-08-07, Spark, 4 runs)

Four runs of `scripts/diag_transient_determinism.py ngc5907 R`. Exactly two of
46 registered frames ever differ, and they differ by different amounts:

| run | frame 013 | frame 014 | `s` | residuals (5σ) | diff RMS |
|-----|-----------|-----------|-----|----------------|----------|
| 1   | A         | A         | 0.2768903 | 921 | 1.24101 |
| 2   | A         | B         | 0.2686137 | 940 | 1.21979 |
| 3   | B         | B         | 0.2686109 | 940 | 1.21977 |
| 4   | A         | B         | 0.2686137 | 940 | 1.21979 |

Runs 2 and 4 are bit-identical throughout. Read the table by column:

* **frame 014** flips between two footprints differing by 14,825 pixels. When
  it flips, `s` moves 3% and the detection count moves 19.
* **frame 013** flips between footprints differing by only 501 pixels. When it
  flips *alone* (run 2 → run 3), `s` moves by 0.001% and the detection count
  does not move at all.

That contrast is the proof. A small footprint change leaves the covered-pixel
count alone, so the subsample is not redrawn and the estimator is reproducible
to six decimals. A large one changes the count, redraws the subsample, and `s`
jumps. `n_accepted=46` throughout.

### Supporting measurements

All on the run-2 template/science pair, PSF-matched exactly as production does:

* `s` across 8 subsamples of **identical** data: 0.2549 → 0.4497, **61.6%**
  peak-to-peak. The production values 0.2769 and 0.2686 sit inside that spread.
* Holding the subsample **fixed** and perturbing the values by 25× the real
  disturbance moves `s` by **≤0.03%** — so the values are not the carrier, the
  redraw is.
* Why the estimator is unstable: per clipping round, the surviving template
  pixels' 1–99 percentile stays ~[241, 262] while the *max* collapses
  10923 → 1521 → 496, `corr(xs, ys)` falls 0.92 → 0.81 → 0.46, and `cond(A)`
  climbs 1.7e3 → 1.3e4. Three rounds of 3σ clipping remove precisely the bright
  pixels that give a slope any leverage, leaving a near-constant sky regressed
  on a near-constant sky. Only the **product** `s·sky + b` stays pinned — it is
  229.56 / 229.20 / 229.78 across three wildly different `(s, b)` pairs.
* The residual sky is dominated by *noise*, so the OLS slope suffers classic
  attenuation bias toward zero. That is why `s ≈ 0.27` rather than ≈1.

## A proposed fix, deliberately NOT in the tree — now superseded

**Superseded 2026-08-07.** The bisection never showed `FWHM value=` as the first
divergence, so by the rule below this fix does not apply. Do not re-apply it.
Kept for the record only.

A candidate fix was written and then **reverted on purpose**, so this diagnostic
bisects unmodified production code. An unproven fix in the tree would change the
very behaviour being measured.

The change was two lines in `transient_search/difference.py`: quantise
`_frame_fwhm`'s return value to a 0.05 px grid.

```python
_FWHM_QUANTUM_PX = 0.05        # module level
...
q = round(val / _FWHM_QUANTUM_PX) * _FWHM_QUANTUM_PX
return q if q > 0 else default # in place of: return val
```

It does what it claims in isolation — a `1e-12` perturbation no longer moves the
returned value — and is defensible on its own terms, since seeing is not knowable
to better than ~0.1 px and selecting a convolution kernel from a full-precision
float is spurious precision. But it did **not** stabilise the difference image in
the proxy test (635 vs 634 detections with it enabled), and it is **not**
established as the fix.

Re-apply it only if the bisection shows `FWHM value=` as the first divergence.
`scripts/diag_transient_determinism.py` prints `quantum=ABSENT` when the constant
is missing, so it works either way.

## The experiment to run

`scripts/diag_transient_determinism.py` fingerprints every pipeline stage —
each registered frame, the combined science and template images, every
`_frame_fwhm` value, the scale/background fit, the difference image, and every
detection call — so two runs can be diffed and the **first** stage that
disagrees identified. Bisect; do not reason backwards.

```bash
cd ~/Documents/srt && source .venv/bin/activate
python scripts/diag_transient_determinism.py ngc5907 R > /tmp/run1.txt
python scripts/diag_transient_determinism.py ngc5907 R > /tmp/run2.txt
diff /tmp/run1.txt /tmp/run2.txt | head -40
```

Run at least 3 pairs — the effect may be intermittent, and two identical runs
prove nothing on their own.

A fifth instrument was added after the fact: `SCALE n_fit=` logs the fit
population size. Without it the diff cannot distinguish "the subsample was
redrawn" from "the values shifted slightly", and those have different fixes.

### Reading the diff

- **First divergence at a `REG frame[...]` line** → registration is the origin.
  ~~Check whether the *set* of accepted frames also changed (`n_accepted`)~~ —
  checked, it does not; `n_accepted=46` in all four runs. What matters instead
  is the *size* of the footprint change, and whether `SCALE n_fit=` moves with
  it. This is what actually happened.
- **`REG` identical but `IN science`/`IN template` differ** → the combine is at
  fault, not registration.
- **Inputs identical, `FWHM value=` differs** → the original hypothesis is
  correct after all; the quantisation fix stands and its comment can be kept.
- **Inputs and FWHM identical, `SCALE`/`DIFF` differ** → look inside
  `_gaussian_match` / scipy, or non-associative float reduction.
- **Everything identical up to `DIFF`, only `DET` differs** → non-determinism
  inside `sep` itself; every residual count is then approximate and thresholds
  need revisiting.

## What to fix — the choice is not obvious

Three candidates, all measured on the run-2 pair under the *same* 14,825-pixel
disturbance that frame 014 produces. "p2p" is peak-to-peak spread in `s`:

| candidate | `s` | p2p under disturbance | verdict |
|-----------|-----|-----------------------|---------|
| today: seeded 200k subsample | 0.25–0.45 | **100%** | the bug |
| drop the subsample, fit all 61M | 0.586 | **5.0%** | deterministic, still unstable |
| fixed stride `xs[::step]` | 0.222 | **98%** | *worse* — see below |
| sky from medians, slope from bright pixels | 0.960 | **0.026%** | stable |

Two of these deserve comment.

**The stride is a trap.** It looks like the obvious size-independent subsample,
but dropping pixels shifts every subsequent index, so it re-rolls just as
thoroughly as `choice` does — 98% spread, no better than today.

**Removing the subsample is not enough.** It makes the estimator a deterministic
function of the data, but the estimator is still ill-conditioned, so it still
swings 5% under a disturbance this small. 5% in `s` is the same order as the
13% background-RMS swing that started this investigation. It would not deliver a
number stable enough to quote a limiting magnitude from.

**The fourth is a real change to the science, not just a determinism fix.** Sky
is additive and independent per epoch, so take it from the medians; source flux
scales, so take the slope only from pixels well above sky, where a slope has
leverage. Measured consequences on the ngc5907 R pair:

* `s` goes 0.27 → 0.96, and is stable to 0.026%. It converges toward ~0.975 as
  the brightness cut rises (the residual attenuation shrinking), which is the
  physically sensible answer for two medians of the same instrument.
* median |residual| at template 2000–10000 ADU: 2414 → 537. At >10000 ADU:
  11884 → 2461. Peak difference-image value: 43823 → 7876.
* but the difference-image background RMS **more than doubles**, 1.53 → 3.51,
  because a template subtracted at nearly full weight contributes nearly all of
  its noise. Today's small `s` buys a quiet background by not subtracting.
* and median |residual| at template 600–2000 ADU gets *worse*, 124 → 387,
  which is probably PSF-matching error that the degenerate fit was masking.

So option 4 fixes the determinism and the bright-source residuals, at the cost
of a louder background and a false-positive profile that the shape cuts, the
bright-source mask and the newness gate were all tuned against. Those cuts have
their own hard-won history (see the comments in `_reject_and_rank`), and every
one of them would need re-validating against `test_injection.py`.

**Do not skip the trigger either.** Even with option 4 the registration coin
flip remains — `stacking/stacker.py` shares `_register_frames`, so the stacker
and the published convergence curves inherit it. Options: vendor a seeded
`_ransac` into the tree, or seed the global numpy legacy RNG (does **not** work
— astroalign uses `default_rng()`, not the legacy global), or accept it and
rely on a scale estimator insensitive to it. The last is what option 4 buys.

### Then

Once a fix is chosen, re-run the acceptance test to confirm the floor is
stable across runs:

```bash
python transient_search/test_injection.py ngc5907 R    # exits non-zero on failure
```

Two consecutive runs must give the same 50%-recovery flux. Only then is a
limiting magnitude quotable, via the photometric zero-point.

## Validation (2026-08-07, after the fix)

**Determinism — fixed.** Two more diagnostic runs, with the leverage-based
estimator in place. A frame still flips (frame 015 this time, not 013/014 —
the RANSAC coin flip moves around, as expected), and `SCALE n_fit=` shows the
covered-pixel count moving with it, 60815856 vs 60815921. That 65-pixel change
is exactly what used to reroll the whole 200k subsample. Now:

| quantity | before fix (runs 1 vs 2) | after fix (runs 5 vs 6) |
|----------|--------------------------|-------------------------|
| `s` | 0.2768903 / 0.2686137 (**3%**) | 1.1348284 / 1.1348282 (**2.4e-7**) |
| diff RMS | 1.24101 / 1.21979 (**1.7%**) | 3.937774 / 3.937753 (**5e-6**) |
| raw residuals (5σ) | 921 / 940 | 215 / 216 |

`n_fit=` also settles the last open question in the diagnosis: the covered-pixel
count really does move between runs, so the redraw was the carrier — not merely
inferred from the fixed-index perturbation test.

Not bit-identical: one raw residual of ~215 crosses the 5σ line differently,
because a 2.4e-7 change in `s` can still flip a source sitting exactly on the
threshold. That is discreteness, not instability, and it does not propagate —
the reported candidate list and the acceptance test are identical.

**Acceptance test — reproducible, but less sensitive.** Both runs, identically:

```
   100  0/3     800  0/3      3200  3/3        false positives : 0
   200  0/3    1600  3/3      6400  3/3        floor           : 1600 ADU
   400  0/3                                    monotonicity    : ok
```

False positives went 3 → 0. The floor went 400/800 → **1600 ADU**, i.e. the
search lost roughly 2–4× in flux, about 1.5 mag.

**The old floor was not real.** It was measured against a background RMS of
1.24, and that RMS was low only because `s ≈ 0.27` barely subtracted the
template. The same difference image carried 11884 ADU of median residual above
10000 ADU of template and a peak of 45499 — systematic error far larger than
the "noise" the threshold was scaled to. A 5σ cut against an RMS that excludes
the dominant error term is not a 5σ cut.

**But the new floor is not photon-limited either, and that is the next lever.**
If the extra RMS were just template noise it should have risen ~16%: the
template is a median of 36 frames against the science's 10, so
`sqrt(σ_sci² + s²σ_tmpl²) ≈ 1.16 σ_sci`. It rose **3.2×** (1.24 → 3.94). The
surplus is structured subtraction residual at stars — the same effect as the
600–2000 ADU band getting worse in the bench test — now exposed rather than
suppressed by under-subtraction. Its likely source is the PSF match:
`_gaussian_match` matches a single scalar FWHM with a Gaussian kernel, and
neither the PSF nor its variation across a 61 MP field is Gaussian.

The intended remedy is already anticipated in the module docstring: `ois`
(Alard–Lupton) fits a spatially-varying matching kernel and is not installed
(established fact 3). Installing it and re-running this same acceptance test is
the obvious next experiment, and it should buy back much of the 1.5 mag. Do NOT
quote a limiting magnitude as a property of the telescope until that is settled
— 1600 ADU is a property of the current subtraction, not of the optics.

## Also worth checking

`stacking/stacker.py` shares `_register_frames`, so ~~if registration is the
origin~~ — it is — the convergence curves published on the lab site inherit the
same instability. Frame 014 of this set lands on one of two transforms at
random, and nothing in the stacker is any more robust to that than the
difference pipeline was. Worth measuring before the next curve is published.

### The difference pipeline applies NO calibration

Noticed 2026-08-06 and not yet judged. `_register_only` loads frames through
`stacker._load_fits_2d`, which returns the raw FITS as float32 — **no bias, no
dark, no flat**. `stacking/stacker.py` performs full calibration with cached
masters, so this path bypasses machinery that already exists in the tree.

Largely defensible, and possibly deliberate:

* the same optics imprint both template and science, so vignetting and dust
  shadows are common to both and mostly cancel in the subtraction;
* `_robust_scale_bg` fits `science ≈ s·template + b`, absorbing the global scale
  and the additive bias/dark pedestal;
* hot pixels are fixed to the sensor and appear in both epochs, so they subtract.

Where it stops being defensible:

* **flat-field response is multiplicative and spatially varying**, and a single
  global `s` cannot correct it. Anything that changed the flat between epochs — a
  dust mote moving, the focuser rotating, the camera being refitted — leaves
  structured residuals indistinguishable from the artifacts the shape cuts fight;
* hot pixels only cancel if both epochs ran at the same sensor temperature;
* the template spans four nights and the science image one, so their noise
  differs in ways calibration would partly equalise.

On the ngc5907 R set the template is 2026-06-02 → 06-08 and the science night is
06-09, so the flats were plausibly stable across that week — which may be the
only reason this works. Over a longer baseline, or across an optics disturbance,
it would degrade quietly.

**Relevance to this investigation:** if calibration were applied, frame-to-frame
differences would shrink, which could change how close registration QA sits to
its thresholds. Worth knowing before concluding anything about marginal frames.
Test cheaply by calibrating the same set through the stacker's masters and
re-running the bisection.
