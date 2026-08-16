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
