# Plan: the filter advisor (tonight's filter split without a human)

Status 2026-10-18: approved by the owner; step 1 (shadow) being built.

## Why

On 2026-10-18 the owner and Claude changed three targets' filter plans by hand
before noon, and each change was a judgement the existing planner could not
make:

| Hand change | What the planner missed |
|---|---|
| ngc1961 L-heavy | `filter_need` (§2c, convergence) measures NOISE only. 51 L frames read as converged and got no time, though their median FWHM was a soft 2.55" (7 under 2.4"). The detail channel needs SHARP frames. |
| ngc6946 Hα back in | A galaxy's filter set is fixed at L/R/G/B. Hα never joins, even with Hα data and HII regions, and a fifth filter cannot rotate through the 4-block template. |
| hubblev1 L only | Monitor targets need a fixed recipe; "need" does not apply. |
| (next day) ngc6946 B back | Explicit `filters` plans never expire, so a one-night change silently becomes permanent. |

This is one of the autonomy gaps (project 3): every one of those calls has to
be made before the noon plan writes the sequence, and today nobody makes it.

## What it decides

For each target in the noon plan's slots, a filter RATIO (the same form as
`filters`, read by the generator as a share of the slot's hours) plus the
reasons, in words, for the chat post and the journal.

1. **Need from quality, not only count.** From the target's own
   `frame_stats.json` (FWHM per frame, written by the frame watcher):
   - a frame is *usable* when it would pass the stacker's gate
     (FWHM <= 1.25 x that filter's median, stacker.quality_gate);
   - the detail channel (L for broadband, Hα for narrowband) counts only
     *sharp* frames: FWHM <= `SHARP_ARCSEC` (2.4", the site's good-seeing
     band from weather.seeing_from_wind's calibration);
   - need = goal - have, per filter, floored at 0. Goals (300 s frames) are
     per recipe and configurable: L 60 sharp, R/G/B 24 each, Hα 36 for a
     galaxy that carries it; Hα/O-III/S-II 40 each for a nebula.
2. **Filter set.** Broadband L/R/G/B; Hα joins a galaxy when the target already
   has Hα frames or its queue entry asks for HALRGB. Narrowband Hα/O-III/S-II.
   More than four filters with need: the smallest need sits out tonight. It
   grows relative to the others as they fill, so the starved filter rotates in
   on its own.
3. **Tonight's conditions.**
   - Seeing (850 hPa wind over the slot, weather.seeing_from_wind): good
     -> detail channel x1.5. Poor/bad -> the detail channel gets a token
     share (ratio 1) and the other filters split the night, each weighted at
     least a quarter of its goal so colour at goal keeps deepening (colour is
     binned and blurred by the stretch and does not need the seeing). Changed
     2026-10-19 from "x0.5", which still gave ngc1961 most of five hours of L
     in 42 km/h of wind: a soft night makes almost no sharp frames.
   - Moon up during the slot and > 50 % lit: L and B x0.5, Hα x1.5, O-III x0.7.
4. **Leave alone.** Monitor targets (`lightcurve: true`) keep their plan.
5. **All goals met.** The advisor says so (the target is a candidate to retire)
   and returns a plain 2:1:1:1 / 1:1:1 deepening split.

Explicit `filters` plans still win. Step 3 adds an expiry so a one-night change
stays one night.

## Rollout

1. **Shadow (2026-10-18).** After the noon sequence is written, the advisor
   runs for every slot target in a background thread and posts one chat
   message: what it would plan, why, and what is actually planned. Journaled
   as a `FILTER_ADVICE` note. It changes nothing.
2. **Default (wired 2026-10-30, C1 of docs/AUTOMATION_PLAN.md; the flag is
   still `shadow`).** The advisor now runs INSIDE the noon plan
   (`control/night_plan.apply_filters`), and each slot of the `NIGHT_PLAN`
   record carries `filters` (the ratio), `filter_source` (explicit > monitor
   > advisor > type) and `filter_reasons`; the record's `filter_mode` is
   `cfg["nina"]["filter_advisor"]` as the planner saw it. The scheduler reads
   the ratio off the plan and hands it to the sequence generator as a
   *preset* (`nina_sequence_gen._plan_for(preset=)`), which takes the same
   ratio path as an explicit `filters` plan and is read back the same way.
   `shadow`: recorded and posted, the generator decides as before.
   `default`: the plan's ratio is the sequence's. Flip the key and `update`;
   the next noon plan carries it. The clean criterion is seven noon plans
   whose `filters` equal the sequence read-back with no hand change.
3. **Expiring overrides.** `filters <dso> ... for N` (nights), after which the
   advisor takes over again.

## Code

- `fits_processing/filter_advisor.py`: `advise()` is pure (dict inputs,
  testable on CI); `advise_for()` gathers the live inputs and never raises.
- The decision: `control/night_plan.slot_filters` / `apply_filters` (pure),
  `read_filters` (live), `describe_filters` (the chat text).
- The post: `end_points/scheduler_server._post_filters`, after every
  `generate_sequence_task` of the driven path (noon, replan, a sunset
  regeneration): reads the written sequence back and journals `FILTER_ADVICE`
  with `{mode, slots: [{dso, source, filters, sequence, reasons}]}`.
- Tests: `tests/test_night_plan_filters.py`; the preset path in
  `tests/test_filter_plan_sizing.py` and `tests/test_nina_slots.py`.
- Tests: `tests/test_filter_advisor.py`, one per rule above.
