# Knowing whether the roof is open

Notes on why the current answer is unreliable and what would replace it. Written
2026-08-15 after a roof-open daylight ladder came back ambiguous, and after a
limit switch had previously failed by asserting the wrong state.

## The question is mechanical; the measurement is not

`vision_safety` decides the roof's position by template-matching three small
markers in a webcam frame, across a ten-rung exposure ladder, and voting. That
requires the system to be right about **appearance** in order to answer a question
about **geometry**, under lighting that spans a lit interior at night, daylight
through a shut roof, and daylight through an open one.

It is worth being precise about the failure, because it is not "the camera is
bad".

## What actually happened, 2026-08-15 17:18

Roof genuinely open, scope parked, mid-afternoon.

```
exp -7   parked 0.92/ 34px   closed 0.73/489px   open 0.65/ 39px   <- correct
exp -9   parked 0.80/ 33px   closed 0.65/ 63px   open 0.69/582px   <- spoiled it
```

The open marker resolved correctly at exp -7, 39 px from its expected position.
But at exp -9 the **closed** template matched 63 px from *its* expected position —
and the closed marker's expected position, (829, 152), is upper-right, which an
open roof turns into sky and foliage. The matcher found a tree that resembles the
marker. With `accuracy = 150 px` that counts as a valid vote.

`_decide_from_rungs` then refuses, correctly by its own rule: a state wins only
if the opposite state scores zero. Open 1, closed 1 → ambiguous → unknown.

Two things this disproves:

* The docstring's premise that wrong matches "miss by 120-400px, never
  narrowly". This one missed by 63 px, and reproduced twice on independent
  captures (0.65 then 0.63 confidence). A 150 px window cannot separate a
  phantom at 63 px from true matches at 33 and 39 px.
* The premise that the vote is unanimous. It was, across all 13 ladders captured
  before this one. Daylight-with-roof-open is the condition that breaks it, and
  it had not been sampled.

## Adapting the exposure does not fix it — measured

The obvious response is that the frame is blown out (clip 82-100% on most rungs)
so the ladder should extend darker. It was tried, roof open, daylight:

```
exp   luma    clip%  | parked        closed         open
-9    156.6    8.4   | 0.76/ 33px    0.63/ 63px     0.69/582px
-10    99.5    2.7   | 0.69/ 33px    0.58/354px     0.70/581px
-11    60.8    1.3   | 0.73/ 33px    0.58/218px     0.72/582px
-12    27.7    0.3   | 0.76/ 33px    0.58/214px     0.71/582px
-13    20.1    0.1   | 0.77/ 33px    0.56/491px     0.70/582px
-14    19.9    0.1   | 0.77/ 33px    0.58/482px     0.70/582px
```

The open marker sits at 581-582 px on **every** darker rung and never resolves.
The camera also bottoms out: -13 and -14 return the same luma, and both fall
below `min_trust_luma` of 25 so would be discarded anyway.

The open marker is readable at the **bright** end, not the dark end — it resolved
only at exp -7, where the frame is 82% clipped. The likely reason is that the
582 px decoy is a darker feature which saturates flat at -7 and stops matching,
letting the true marker win. Going darker strengthens the decoy. So the ladder is
not mis-ranged; the method is picking between two things that look alike.

## The principle worth designing to

A limit switch was tried previously and **failed by asserting the wrong state**.
That is the same class of fault as the phantom tree: a single component failing
in a way that produces a *confident wrong answer* rather than no answer.

So the design goal is not a better sensor. It is that **no single failure can
produce a confident wrong answer**. Three rules follow.

**Prefer a number to a boolean.** A switch has no way to say "I am broken". A
distance reading does: out of range, unchanged when it should have moved,
implausible rate of change. It also answers *partially open*, which a pair of
end-stops cannot and which matters here — the all-sky camera can prove some
aperture exists but not that the roof completed its travel.

**Prefer modalities that fail differently.** Vision fails on appearance. A
mechanical switch fails on contacts. A current trace fails on the supply. Two
sensors that fail the same way are one sensor.

**Keep unknown as an outcome.** Tonight's refusal was correct — the system truly
could not tell. Every complaint here is about false negatives, which cost a
night; the property that has never hurt us is refusing to move on doubt.

## What I would change, in order

**1. Replace marker matching with a region test.** Software only, no hardware,
and testable immediately against the 13 ladders already captured plus the ones
from tonight. When the roof opens, a large region changes from wood underside to
sky — an enormous difference. The all-sky camera needs 0.5 ms open against ~2 s
closed for the same exposure level, a factor of roughly 4000. Region brightness,
variance and texture over the aperture are hard to confuse; a 20x20 correlation
peak is easy to confuse. This is the highest value change and the cheapest.

**2. Add a position sensor that reports a distance.** A time-of-flight laser
(VL53L1X class) along the roof's travel gives continuous position: closed, open,
and every intermediate. Mount it where a stuck roof reads differently from a
travelled one. Cross-check it against what is already recorded — if the motor
drew current for 22 s and the distance did not change, something is wrong and
that is now *detectable* rather than silent.

**3. Require agreement.** Two independent signals must agree, or the state is
unknown. The observatory already records two things about every move that were
both correct last night: the **current signature** (Shelly 3EM channel 0) and the
**roof audio classifier**, which auto-filed the 21:01 open as good. Those
establish that a move happened and looked normal. What is missing is position at
rest, which is what item 2 supplies.

## The camera for item 1 is already installed

`Iris cam` (KC410S at 192.168.87.65) hangs inside the observatory and frames the
scope and the roof underside together, at **2560x1440** against the 1920x1080 of
the DirectShow webcam `vision_safety` drives. On the morning after the failure
above it returned a correctly exposed frame of that scene on the first attempt,
with no exposure ladder at all.

That is item 1's region test, pre-framed: the area that changes from wood
planking to sky is the middle of this camera's field, at roughly twice the pixels
and without the ten-rung sweep that manufactured the phantom.

Two properties constrain how it can be used, and the second is the interesting
one.

**It auto-exposes**, like the outdoor Kasa. Absolute brightness is not comparable
between frames, so the test has to be built on structure — texture, edge density,
the geometry of the aperture — rather than a brightness threshold. Those survive
renormalisation; a threshold does not.

**It pans and tilts.** A camera that can move is a camera whose regions can
quietly stop meaning what they meant, which is precisely the confident-wrong-
answer failure this document is written against. Someone nudging it from the
Kasa app would leave a roof test comparing the correct statistic over the wrong
patch of wall, and nothing in the frame would announce that.

The saving grace is that the motor reports a **number**: `get_position` returns
`{x, y}`, so the reference frame is checkable. Store the position the regions
were calibrated at, read it before trusting a verdict, and a camera that has been
moved becomes *unknown* rather than *wrong* — the same argument as item 2, for
free, on hardware already installed. It is worth building that check in from the
start rather than discovering the need for it later.

Both facilities are cloud-only; the LAN exposes the stream and nothing else. The
interface, the port map behind that claim, and the reason the sky camera must
never be pointed are in [[KASA_CAMERA_PTZ]].

## Measured on a real move, 2026-08-16

A full open and close through `roof!!`, recorded on Iris cam. Both the picture
and the microphone were captured off the one 19443 stream by
`scripts/iriscam_record.py`, ~11 min after sunrise, sun on the building.

**The region test separates cleanly.** Statistics over the aperture — the patch
that is roof underside when shut and sky when open, `img[100:1400, 0:700]`:

| metric | roof closed | roof open |
|--------|-------------|-----------|
| mean | 118–127 | 68–71 |
| Canny edge density | 11.8–15.4 % | 22.9–25.9 % |
| green excess `G−(B+R)/2` | −6.1 … −2.8 | **+2.2 … +3.3** |

Twenty-four samples open, twenty closed, across two independent moves, and no
overlap on any of the three. Green excess is the strongest because it **changes
sign** — interior pine and foil insulation are red-dominant, foliage is
green-dominant — so it is a zero crossing rather than a threshold on a
magnitude, and auto-exposure renormalisation cannot move it. Compare the
marker matcher, whose true matches scored 0.65–0.92 and whose phantom tree
scored 0.63–0.73: overlapping.

Note the mean moves the *wrong* way — the interior gets **darker** when the roof
opens, because auto-exposure stops down for the new sky in frame. Brightness is
usable here but only as a relative shift, which is the point made above about
building on structure rather than level.

**Two things that would confound a naive implementation.** The frame at t=150 s
in the close recording reads mean 170, edge 3.5 % — that is the post-move vision
confirm switching the interior lights on, and it looks like neither state. And
the mid-travel frame (t=70 s, mean 81, green excess −0.74) sits between the two
populations, exactly as it should. Any region test needs to treat "in motion"
and "lights on" as their own outcomes rather than forcing a binary.

**The microphone is better than expected.** Against a room floor of rms 3–5,
peak 8–16:

| | peak rms | peak sample | audible travel |
|---|---|---|---|
| open | 1846 | 29052 | 11.5 s |
| close | 957 | 9852 | 11.5 s |

That is ~500x the noise floor, and the two directions have *different envelope
shapes*: the open is an impulse that decays, the close is a slow ramp to an
impact at the end. Direction looks inferable from sound alone.

**On the first pair the existing camera got it right**, returning "Roof: open;
scope: parked" correctly. That is no longer the whole picture — see below.

## It reproduced, and it blocks the close

Second pair, 2026-08-16 10:23, sun alt **+46.7°** against +17.3° in the
morning. The existing camera could not confirm the roof it had *itself just
opened*:

```
Roof open not confirmed (attempt 1/5) — open template: ambiguous
(closed 2 rung(s), open 2 rung(s)) (conf 0.65, off by 44px), waiting 5 min
```

The same signature as 2026-08-15: a phantom closed match tens of pixels from
where the marker belongs, cancelling a correct open vote. `iris.log` shows it is
**recurrent**, not a one-off — 17:11, 17:18, 17:25, 19:46 and 19:53 on
2026-08-15, then today. So the earlier note above understated it: this is not
only about margin.

At the same moment the Iris cam separated cleanly — green gap 5.59, sign
separated, registration shift ~0 px.

**The operational half is worse than the missed verdict.** `open_roof` holds
`_roof_lock` across the whole confirm loop — five attempts, five minutes apart.
An unconfirmable roof therefore blocks every later roof command for up to
**25 minutes**. Measured today: `roof!! close` was refused with "another roof
command is already running", and only ran after the stuck job was cancelled
through `/api/jobs/<id>/cancel`. A roof that cannot be *verified* open is
currently also a roof that cannot be *commanded* shut, which is the wrong way
round — verification is advisory, closing is safety. Worth separating.

## Watch the open side, not the gap

Three runs so far:

| time | sun alt | shut green excess | open green excess | gap |
|------|---------|-------------------|-------------------|-----|
| 07:40 | +17.3° | −6.09 … −3.76 | +2.18 … +2.76 | 5.94 |
| 07:48 | +18.8° | −5.84 … −3.99 | +2.31 … +3.22 | 6.30 |
| 10:23 | +46.7° | −6.03 … −5.20 | **+0.39** … +1.90 | 5.59 |

The gap looks stable, and that is misleading. Sign separation depends on the
**open** side staying above zero, and its margin fell from +2.18 to +0.39 as
the sun rose — the gap held only because the shut side moved further negative
to compensate. Direct sun on foliage and more blue sky in the aperture both
push the open region away from green-dominant. If the trend continues at
midday the sign test could fail while the gap still reads healthy, so track the
open-side minimum, not just the gap.

All of the above is **daylight only**. Green excess is worthless at night, when
the camera switches to IR and returns a monochrome frame — expect the night
signature to be a dark aperture against a lit interior, which is a different
discriminator that has not been sampled yet. Do not ship a region test until it
has been measured after dark, and on an overcast day.

## The camera and its microphone are ONE sensor, not two

Measured 2026-08-16, and it bears directly on replacing both the safety webcam
and the roof microphone with this one camera.

The camera has its own enable, separate from the plug it is powered from
(`smartlife.cam.ipcamera.switch`, see [[KASA_CAMERA_PTZ]]). Switching it off is
a real disable: the 19443 stream goes from HTTP 200 with 76 video and 84 audio
parts to **HTTP 503 with no parts at all**. Video and microphone stop together.

That is not only about the switch. Everything upstream is shared too — one
network path, one plug, one firmware, one app toggle, one cloud account.

**An earlier version of this section claimed that consolidating onto the Kasa
camera would turn two independent channels into one. That was wrong.** The roof
microphone is part of the USB webcam, so the existing picture and sound already
share a single USB connection and already fail together. Moving to the Kasa
camera is not a loss of independence — if anything it is a gain, trading a USB
bus on a PC for a network path and mains power.

The independence argument therefore does not weigh against the Kasa camera. It
still says the same thing about where the *second* channel must come from: not
from that camera's own sound.

This does not sink the idea — the margins measured above are far better than
the marker matcher's, and the failure is at least **loud**: a 503 is an
unmistakable "I cannot answer", not a confident wrong answer, which is the
class of fault that actually costs equipment. But it means:

* `get_is_enable` has to be checked before any verdict from this camera is
  trusted, because switched-off and unreachable look identical.
* The independent channel has to come from somewhere else. The Shelly 3EM
  current signature already is one and fails on the supply, not on appearance
  or on the network — so keep it in the agreement rule of item 3, and do not
  count the Kasa camera's picture and sound as the two independent signals.

## Three candidate fixes, measured — only one survived

Tested 2026-08-16 against 18 labelled ladders (12 open, 6 closed) spanning
night to sun altitude 48°, replayed offline with no hardware. Ground truth from
the commands run plus the Iris cam recordings.

| variant | correct | unknown | NOT-parked | wrong |
|---------|---------|---------|------------|-------|
| baseline (shipped) | 10 | 4 | 4 | 0 |
| clip guard ≤95% | 10 | 4 | 4 | 0 |
| chroma check ≤0.05 | 5 | 9 | 4 | 0 |
| **competitive scoring** | **14** | **0** | 4 | 0 |

**Competitive scoring shipped.** Instead of letting either state veto the
other, the two are scored against each other — `conf − error/accuracy` — and
the winner must lead by `_ROOF_TIEBREAK_MARGIN`. Zero wrong verdicts at *every*
margin from 0.00 to 0.60, with a flat plateau 0.00–0.20, so 0.15 is not tuned
to an edge; above 0.30 it degrades back to the old behaviour.

**The clip guard did nothing.** Plausible — seven of ten rungs were 100%
clipped and formally eligible to vote — but on this corpus those rungs never
carried the deciding vote anyway. Kept as a known latent hazard, not shipped:
it adds a threshold without buying anything measurable.

**The colour check made it worse, for an instructive reason.** The idea was
that a yellow star on brown and a blue triangle on silver are trivially
separable by hue, and the matcher throws colour away (`IMREAD_COLOR` then
`cvtColor(BGR2GRAY)`). But the *templates* are nearly the same colour in
aggregate: chromaticity `[0.378, 0.331]` closed vs `[0.357, 0.347]` open, 0.026
apart — because each tile is mostly background, so it compares brown against
silver, not star against triangle. The metric even ranks backwards: true closed
matches scored 0.065–0.119 while phantoms scored 0.037–0.041.

Colour is not useless here; the *aggregate* of a mostly-background tile is. A
colour test would need to isolate marker pixels, or the physical markers would
need backgrounds that differ. Neither is a code change alone.

Also disproved along the way: "the phantom only appears in blown-out frames."
The exp −11 phantom appeared at **14.3%** clip.

## The first held-out sample falsified the margin

11:50, roof manually opened, judged three ways. Worth recording because two of
the three were wrong and the corpus above did not predict it.

| judge | verdict | correct |
|-------|---------|---------|
| incumbent, live | ambiguous | no |
| competitive logic at margin 0.15 | UNKNOWN | no |
| Iris cam metrics | open (green excess +1.79) | yes |
| Iris cam registration guard | refused | — |

The tie-break failed **by 0.003**. Open genuinely scored better (+0.333 against
closed's +0.187) but led by 0.147 against a required 0.15. Re-measured across
all 19 ladders, margins 0.00–0.12 give 15 correct / 0 unknown / **0 wrong**
while 0.15 gives 14 — so the plateau's upper edge is between 0.12 and 0.15, not
the 0.20 that 18 ladders suggested. Margin is now **0.10**.

This is retuning after a failed prediction, on evidence of one sample. The new
value is better supported than the old, and 19 ladders is still a small corpus.
Expect it to move again.

**The Iris cam guard was also wrong, in the opposite direction.** It refused a
frame whose camera had demonstrably not moved. Exactly four patches passed
scoring — the minimum — and one of those was a confident false match
(+147/−224 px at score 0.774), leaving three in agreement against a fixed
requirement of four. The agreement test worked; the arithmetic around it had no
slack. Now 8–10 patches with a majority rule (60% of matched patches, floor of
3). Verified: the refused frame certifies at −6/−1 px, injected shifts of +40
and +120 px are still recovered to within the same −6 baseline, and +300 px is
refused rather than mis-reported because it exceeds the search window.

## Parked, first measurements — and the trap in them

2026-08-16, scope moved by hand to three poses with the roof open.
`scripts/scope_parked_probe.py` scores normalised correlation of the scope
region against a stored parked reference, after registration.

| pose | score |
|------|-------|
| reference pose, simply re-photographed | 0.964–0.990 |
| **parked again after actually moving it** | **0.841** |
| moderate off-park (OTA swung off the panel) | 0.609 |
| large off-park (pointed up) | 0.344 |

**The first row is not the parked population.** Those three samples never moved
the scope; they measure how repeatable the *camera and lighting* are, not how
repeatably the mount parks. Treating them as bounds gave a floor of 0.96 and a
threshold around 0.93 — which the very next genuine re-park, at 0.841, would
have rejected. A safety gate tuned that way would refuse to move the roof for a
correctly parked scope.

So the usable margin is 0.841 against 0.609, not 0.96 against 0.61. Separated,
but on ONE sample of the only thing that matters.

Two caveats in opposite directions. This was a *manual* re-park; the real
system parks through PWI4 with encoders, which should repeat far better, so
0.841 is probably pessimistic. But every sample here was taken within fifteen
minutes in one lighting condition, so the parked floor across a day and a night
is unmeasured, and that can only push it down.

**What would settle it:** several software park cycles (`mount_park()`), each
scored, to get the real parked distribution — not more hand-placed poses. Until
that exists there is no defensible threshold, and this must not gate anything.

Sensitivity, for calibration: translating the parked frame by 5 px scores
0.888, 10 px 0.853, 20 px 0.790, 50 px 0.652. At ~23 px/degree the metric
resolves a fraction of a degree, so it is not short of discrimination — the
open question is entirely where the parked population actually sits.

## Why parked never failed, and what that implies

The failure rate tracks template AREA almost exactly:

| marker | size | area | behaviour |
|--------|------|------|-----------|
| parked | 551x171 | 94,221 px² | never ambiguous |
| closed | 215x110 | 23,650 px² | occasional phantom |
| open | 94x132 | 12,408 px² | the one that breaks |

The open marker is **7.6x smaller** than the parked one. A 94x132 patch of
dark-on-bright is something backlit branches supply endlessly at every scale,
which is exactly the phantom traced above. Nothing in the scene resembles a
551x171 target.

There is a second reason parked is easy, visible in any stored ladder frame:
**the existing camera is mounted ON the scope**, looking up and forward. A
small pose change swings the entire field of view, so "parked" is not a subtle
measurement — the whole scene either matches or it does not. The Iris cam sees
the scope side-on from across the room, where the same pose change moves a
modest number of pixels. That is why the side-on parked score needed a
reference library to reach 0.234 of margin while the old camera's parked check
never needed anything.

**The lesson is about fiducials, not cameras.** The reliable measurement is a
large, purpose-built, high-contrast target. Two ways to get one:

* **Move a camera to the scope-top position.** Inherits the sensitivity for
  free. Costs: a pan/tilt camera on a moving mount adds a second uncontrolled
  pose variable on top of the backlash in [[KASA_CAMERA_PTZ]], needs power and
  network on a slewing platform with cable-wrap risk, and gives up the wide
  interior view that produced the clean roof-region separation.
* **Leave the camera fixed and put a large marker ON the scope**, facing the
  Iris cam when parked. This is the old system's actual mechanism — a big
  target — without relocating anything, and it can be in COLOUR, which the old
  greyscale matcher could not use. The Iris cam already frames the OTA and fork
  across most of its view, so there is room for a target far larger than
  551x171 at 2560x1440.

The second is cheaper to try and reversible. It also attacks the measured
weakness directly: the side-on parked score is weak because there is nothing
distinctive to match, not because the vantage is wrong.

## What this does NOT fix

Four ladders still fail, all `NOT-parked`, all in the 10:28–10:34 high-sun
window where only 2 of 10 rungs can establish parked at all. That is the
*parked* detector, untouched by any of the above, and it is the more
safety-critical of the two — a wrong "parked" is what drives the roof into the
OTA. Sample it before trusting it.

## What not to do

* **Do not tighten `accuracy` alone.** 50 px would separate tonight's phantom
  from tonight's true matches, but it is one afternoon's evidence, it tightens a
  hardware-safety gate, and it interacts with the centre-of-match convention in
  `find_template_rectangle` that has deliberately been left alone.
* **Do not rely on the all-sky camera as the roof gate.** It proves an aperture
  exists, not that the roof completed travel. Useful as corroboration, not as the
  answer.
* **Do not extend the exposure ladder.** Measured above; it does not work, and
  the camera has no range left below -13.

## A separate, small fix

The Pushover card sent on refusal shows the rung the *scorer* chose, not the rung
that *decided* the verdict. On 2026-08-15 that produced a photograph plainly
showing an open roof, with the marker correctly boxed, captioned "roof is not
open, stopping". When a state is refused for ambiguity the useful picture is the
contradicting rung.
