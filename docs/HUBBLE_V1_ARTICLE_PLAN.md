# Hubble's V1: the science article, in three reading levels

Written 2026-11-12. Status: **waiting for data**. The trigger is the four
clear nights forecast for 2026-11-15 to 11-18, which cover the rising
branch and the second maximum of the Cepheid M31 V1. Nothing below starts
before the frames of 11-18 have synced to the Spark.

## Why wait, and what the wait buys

The page on the Desktop (`~/Desktop/hubble_v1_lightcurve.html`, built by
`scripts/hubblev1_page/`) folds 19 L nights and 3 G nights on the 31.4-day
period. Phase bins 7 and 8 of 10 are empty, and those are the steep rise
to maximum: the one feature the text calls the classical Cepheid
signature, and the bright branch that sets the intensity mean the
distance is computed from. Today the red line through that gap is the
Fourier fit's guess.

The four nights, predicted from the current fit at mid-slot (07:00 UT):

| Night | Phase | Bin | Predicted G |
|---|---|---|---|
| 11-14 (if it clears) | 0.80 | 7 | 18.82 |
| 11-15 | 0.83 | 8 | 18.63 |
| 11-16 | 0.86 | 8 | 18.47 |
| 11-17 | 0.90 | 8 | 18.36 |
| 11-18 | 0.93 | 9 | 18.30, the maximum |

Three points in bin 8 and the maximum turn the drawn rise into a measured
one, and they are the article's test: the predictions are already drawn
on the page as hollow orange circles, and the new points either land on
them or do not. Bin 7 (phases 0.70-0.80) belongs to 11-13/14; without it
the fold is 9 of 10 bins, which is enough, and the gap is named. The next
chance at that phase is ~12-14 to 12-16 with M31 lower each night.

The queue needs nothing: Hubble V1 is on a `reopen` hold through 11-22.
The moon is a waxing crescent to first quarter those nights and is down
by the slot.

## Step 0, before the data: make the page rebuildable

The page was built from a session scratchpad. The scripts were copied to
`scripts/hubblev1_page/` on 2026-11-12 (uncommitted at the time of
writing). Before the rebuild:

1. Make them run from the repo: a `Makefile`-style `build.sh` (or one
   `python -m scripts.hubblev1_page.build`) that runs, in order,
   `lc_fit.py` (records -> `lc_model.json`), `charts.py` (time and fold
   SVGs), `comps.py` + `comps_chart.py` (comparison-star constancy),
   `stack_v1.py` + `denoise_runner.py` (the L stack, only when the frame
   count changed), `solve.py` + `images.py` (plate solve, field, crop,
   M31 footprint), `build_html.py`. Every script keeps its
   `if __name__ == "__main__"` guard: two of them start spawn pools.
2. The records come from the PC's `local/lightcurves/hubblev1.json`,
   read over ssh. The Spark never writes to it.
3. Every number in the prose that can change (night count, date span,
   max/min, amplitude, rise fraction, mean, distance, next maximum,
   coverage) is already injected from `lc_model.json`. Keep it that way
   for all three reading levels (below), or the levels will disagree the
   first time the fit moves.
4. Commit the folder with a version bump. No test is needed for a page
   builder, but `tests/test_docs_tasks.py`-style import check is cheap.

## Step 1, after 11-18: refit and judge the predictions

1. Sync, then rerun the photometry for the new nights on the PC
   (`lightcurve` already runs end of night; `aavso hubblev1` for the
   Extended Format file, the owner uploads by hand).
2. Rebuild the page. Read three things off it:
   - Did the four points land on the orange circles? Record each night's
     residual from the OLD fit before refitting; that table goes in the
     article. A systematic offset on all four means the period or epoch
     wants revising (fit the epoch free; the period stays fixed at the
     literature value, 19 nights cannot improve on it).
   - The new rise fraction and amplitude against the current 0.30 and
     1.12 mag.
   - The new intensity mean and distance against the current 18.805 and
     2.12 Mly (accepted 2.54). The missing bright branch biases the mean
     faint, so the distance should move toward the accepted value; by
     how much is a result either way.
3. If 11-14 was lost, the fold text says bin 7 is the next cycle's job.

## Step 2: the three reading levels, as a staircase

The owner's goal (2026-11-12): offer something completely understandable
first, then give the reader a challenge to understand more, and draw their
curiosity up. So the levels are a staircase, not a menu of alternatives
for different readers. Design decisions, in the order they matter:

- **Everyone starts at the bottom.** The page opens on the simplest text,
  which must be complete on its own: a reader who never climbs still
  gets the whole story, period to luminosity to distance, with the
  pictures and the real numbers. The deeper levels are the reward for
  wanting more, never the apology for the simple one.
- **Each level ends in a question the next one answers.** The hook is
  specific, not "read more": the story level closes section 1 with "How
  do we know it is 31.4 days and not 30?", and the next level answers
  that with the fold. The physics level closes with "How does the fit
  decide where the line goes?", and the top level answers with the
  Fourier terms and the weights. Every section at every level but the
  top ends on such a question, with the step up beside it.
- **Curiosity is local, so the step up is per section.** A reader who
  wants more about why the star pulsates should be able to climb that
  section alone and stay at the story level everywhere else. The switch
  at the top sets the whole page; the link at the end of each section
  deepens that section only. The current depth of each section is shown
  (three small steps, the lit one the reader's), so they can see how
  far they have climbed and what is left.
- **The deep text is still the source of truth.** It is written first
  (it is what the page holds today) and the two simpler texts are
  compressed from it, never written independently. Every fact and
  number in a simpler text appears in the deep one; a simpler text adds
  no claim. The questions that end each level are chosen from what the
  next level actually answers.
- **Three readers, named by what they are about to do, not by grade.**
  Grade labels make adults choose the top one out of pride and stop
  reading; staircase labels invite the climb. Working names: *The
  story*, *How we know*, *All of it*. The readers behind them: one
  curious twelve-year-old; someone who has had a physics class (inverse
  square law, magnitudes as a log scale, what a relation with a zero
  point is, what an error bar means; equations shown once, not
  derived); and the present text (Fourier fit, weighted least squares,
  the Leavitt coefficients and their derivation, the plate solve, the
  comparison-star ensemble).
- **Figures are shared.** The same SVGs at every level; only the caption
  climbs with the text. A separate "kid-friendly" chart is a fourth
  product nobody maintains.
- **Watch for simplification drift.** Each level down is a chance to say
  something false for clarity. "Bigger stars blink slower" is wrong;
  "the star swells and shrinks, and a bigger, puffier star takes longer
  to swell and shrink" is right. The reviewer's one job on the simpler
  texts is to check them against the deep text.
- **Structure.** The seven sections keep their order at every level, so a
  reader who climbs one section lands in the same place. At the story
  level, section 7 (how the numbers were made) is three sentences and
  section 6 (the physics) is one picture's worth; both grow on the way
  up.
- **Gallery cards are unchanged.** The two-paragraph entry on the site is
  one voice. The staircase belongs to the science article.

Mechanics on the page: three `<div data-level>` blocks per section; a
three-step switch at the top of `<main>` that sets every section; a
"go deeper" link at the end of each section's block that raises that
section one level and scrolls to the same heading; per-section depth
remembered in `localStorage` (wrapped in try/except, the page renders at
the story level without it); no animation; print shows each section at
its chosen depth.

## Step 3: review

- The deep text is checked against `lc_model.json` and the AAVSO file
  (every number), and against the references for the history (Leavitt
  1908/1912, Hertzsprung 1913, Baade 1952, Hubble 1925, Benedict et al.
  2007).
- Each simpler text is checked against the deep text, sentence by
  sentence, for claims it does not contain, and each closing question
  is checked against the next level up: it must be answered there. The Spark's writer/reviewer
  pair (`scripts/article_prose.py`, gemma4 writes, qwen3-vl reviews) can
  do a first pass on this with a prompt that gives it the deep text as
  the only source, but this first article is hand-written: the point is
  to learn what the simple versions actually need before automating.
- One real twelve-year-old reads the story level, if one is available.

## Step 4: publish

- The article goes to the site as a science note, the way the other notes
  were published, with the field picture and the two curves. The gallery
  entry for Hubble V1 links it. American spelling throughout.
- `scripts/site_push.py --yes` (the subscriber announcement) is the
  owner's call and is not run or suggested by the pipeline.
- The AAVSO file for the new nights is uploaded by hand to WebObs under
  THCA.

## Done when

- The fold shows at least 9 of 10 bins with measured points on the
  rising branch and the second maximum.
- The article states each of the four predictions beside its measured
  magnitude.
- All three levels read correctly against the deep text, every closing
  question is answered one level up, the numbers come from one model
  file, and the whole-page switch and the per-section step both work
  with and without browser storage.
- The article is on the site and the entry links it.

## If the weather wins

Publish anyway at the end of November with the fold as it is, the
predictions table left in as predictions, and a line that the rising
branch is the next cycle's work. The three levels do not depend on the
new points; only the numbers do, and they are injected.
