# Daylight roof detection — capture protocol and decision tree

Status as of 2026-08-03: **collecting evidence. No safety behaviour changed yet.**

## The problem, in the user's words

> I would open the roof sooner, but during the day sunlight streams in and the
> camera cannot determine if the roof is indeed open, thus I am in the habit of
> opening the roof later.

That habit is what keeps the mirror from getting a head start on ambient — the
NINA prelude powers the fans at `sunset − 10 min` (`scheduler_server.py:245`),
which is when ambient starts falling fastest. Fixing daytime roof detection is
upstream of the whole fan/thermal question.

## Diagnosis

It is probably not the template matcher. `visual_status()` calls
`take_snapshot()`, whose exposure sweep is scored by `best_exposure_score`
(`sentry/inside_camera_server.py`) — a **whole-frame** metric:

```python
over         = np.sum(L > 240) / L.size
clip_penalty = max(0, under - 0.30)*3 + max(0, over - 0.05)*3
mean_reward  = np.exp(-8.0 * (mean_lum - 0.45)**2)
score        = std_lum * mean_reward - clip_penalty
```

With the roof open in daylight, a blown sky patch covering a quarter of the
frame gives `clip_penalty = (0.25 − 0.05) × 3 = 0.6`, while `std_lum ×
mean_reward` cannot exceed about 0.5. **The penalty dominates the whole score**,
so the sweep picks whatever exposure tames the sky — and the three markers,
which are the only thing the decision depends on, go dark.

TM_CCOEFF_NORMED is already invariant to uniform brightness and contrast change.
What it cannot survive is *saturation* (information destroyed) and *local*
illumination change (a sunbeam across part of a marker alters the pattern, not
just its scale). Exposure addresses the first. Only per-illumination templates
or fiducials address the second.

## The experiment (live now)

`visual_status` already sweeps ten exposures (−2…−11) every call and discards
nine frames. Those are now kept, and every frame is scored with all three real
templates. Nothing extra is captured; no camera time is added.

**Captured when:** the sun is above `exposure_capture_min_sun_alt` (default −6°,
civil twilight) **or** the roof read open on the previous call. The second
condition exists because two separate blocked items need a genuine roof-open
frame and neither needs daylight — see "What else this unblocks".

**Written to:** `base_images/exposure_sets/<timestamp>/` — ten JPEGs plus
`meta.json` holding, per frame: exposure, scorer score, luma, clipped %, and for
each of parked/closed/open the match confidence, position error, and pass/fail.
Sun altitude and azimuth are recorded per set. Rolling cap 30 sets.

**Logged per capture:**

```
exposure ladder 2026-08-03_14-22-01: scorer chose exp -4 (parked_conf 0.31),
    best parked_conf 0.78 at exp -9  <-- DISAGREE
   exp -2   luma 210.4 clip 31.2% score -0.412  parked 0.22/412px  closed 0.19/380px  open 0.31/210px
   exp -9   luma  88.1 clip  0.4% score +0.203  parked 0.78/ 34px  closed 0.71/ 22px  open 0.66/ 61px
```

## How to collect

1. Run `status` a few times with the roof **open in daylight** — the target
   case. Two or three different times of day is much better than three in a row,
   because shadow pattern moves with sun azimuth and that is the failure mode no
   exposure setting can fix.
2. Roof **closed** in daylight is still worth having as the baseline.
3. Roof **open at night** now captures automatically and unblocks two other
   items (below).

Then point me at `docs/daylight_roof_detection.md` and I will read
`base_images/exposure_sets/*/meta.json`.

## Decision tree once the data exists

**If `DISAGREE` fires consistently in daylight** — the diagnosis holds. Fix is a
new scorer passed to `take_snapshot`: score each candidate exposure by the
template match confidence in the three marker regions rather than by whole-frame
brightness. The `scorer` kwarg already exists; this is a small change plus an
A/B against the captured ladders.

**If some exposure in the ladder reaches `parked_ok` / `open_ok` but the scorer
never picks it** — same fix, and the captured ladders prove it offline before
anything safety-critical changes.

**If NO exposure in the ladder gets a good match** — exposure is not the
problem, and a new scorer would be wasted work. Escalate to, in order:
per-illumination template variants binned by sun azimuth (same pattern as the
roof-audio good library), then ArUco fiducials (`cv2.aruco` is already in
OpenCV; two printed tags retire this entire class of problem).

## What else this unblocks

Both have been waiting on a genuine roof-open frame, and the night capture now
produces one:

- **Centre-of-match bug** (`vision_safety.py:56`): `((x + w) / 2, ...)` where it
  means `x + w/2`. Reported displacement is half the true one, so `accuracy =
  150` is really a 300 px tolerance. Corrected positions measured 2026-07-27:
  `parked pos` → (892, 526), `closed pos` → (1538, 277). `open pos` is
  **unmeasured** — it needs a roof-open frame. Line 56 and all three config
  positions must change in one commit, or every roof operation refuses to run.
- **`match_confidence` threshold**: the open template scores **0.684 against a
  CLOSED scene** — a false positive above the 0.6 gate; only the position check
  keeps `open=False` today. If true-open lands at 0.85–0.95, raise the threshold
  to ~0.8. If it also lands near 0.68, confidence cannot separate them and the
  open template needs re-capturing.

## Known gap, not yet addressed

There is a `min_trust_luma = 25` floor but **no ceiling**. An overexposed frame
fails the matches and reports "not parked", indistinguishable from a genuinely
unparked scope. Worth adding alongside the scorer fix so an unusable frame says
so instead of looking like a hardware state.

## Safety notes

Every path here is diagnostics hanging off a safety check. Capture and scoring
run after the verdict is computed, are wrapped so they cannot raise into the
roof-move preconditions, and cannot alter parked/closed/open. Turn the whole
thing off with `exposure_capture: False`.
