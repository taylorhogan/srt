# All-sky camera (ASI120MC-S): bringing it up

The ZWO ASI120MC-S fisheye on `iris-pc`, 1280x960, colour, gain 0-100. It sits
**under the roof**, so it photographs sky only while the roof is open — with the
roof shut it takes a well-exposed picture of the roof's own underside, and the
star detector, pointed at that texture, returns hundreds of detections of which
essentially all are false. Everything below exists because of that fact.

Distinct from the **Kasa KC420WS** ("sky camera"), which is bolted outside and
already publishes to the live panel. The two share code but no numbers: separate
config profiles, separate plate solutions, separate thresholds.

## The order matters

Each step needs the one before it. The threshold has to be set before the
exposure sweep means anything, and the sweep has to run before there is a frame
worth plate-solving.

### 1. Master darks — after dark, roof still SHUT

```
python sentry/asi_allsky.py --dark
```

Roughly 6 minutes for the full grid. This is the only step that wants the roof
closed, so it goes first.

Why it is not optional: the sensor is uncooled, and its hot pixels are
single-pixel spikes far brighter than any star — exactly what a point-source
detector is built to find. The Kasa keeps them out with a 6-pixel minimum blob
size, but this lens puts a focused star into fewer pixels than that, so the
floor comes down to 4 and the dark has to do the job instead.

It **refuses in daylight**, and should: with the roof shut at midday enough
light gets past the seals to give a clearly exposed picture of the roof, and
adopting that as a dark would subtract the roof out of every night frame
afterwards. If it says `REFUSED: 100% of the frame is lit`, it is not dark
enough yet.

### 2. Set the detection threshold — roof open, one frame

```
python sentry/asi_allsky.py --capture --exposure 8 --gain 100
python sentry/star_count.py local/allsky_frames/allsky_<stamp>.fits \
       --profile "allsky camera" --sweep
```

The sweep prints, for each candidate threshold, the count from the real frame
beside the count the identical detector returns on the **negated** residual —
where every hit is by construction a false positive. Pick the threshold where
purity is high without throwing away most of the stars; the Kasa's 12 ADU was
chosen this way and lands at 99.5%.

Put the number in `configs/config_public.py` under `"allsky camera"` →
`star_threshold_adu`, and set `foliage_resid_adu` in the same units.

The shipped values (300 / 200) are **placeholders on a 16-bit scale** and are
almost certainly wrong. If the next step rejects every rung on purity, this is
why — it says so.

### 3. Auto-exposure sweep — roof open, dark sky

```
python sentry/asi_allsky.py --autoexpose --save
```

Roughly 5 minutes: 6 exposures x 2 gains, each measured with the real detector.

This does **not** meter brightness. Metering optimises the picture; the quantity
wanted here is the star count, and past some point a longer exposure raises the
sky as fast as the stars and buys nothing. So every rung is scored by how many
trustworthy detections it actually yields, and the winner is cached in
`local/allsky_settings.json` along with the whole table — so a clear win can be
told from a coin toss between two rungs one star apart.

Rungs that clip past 0.5% are disqualified outright, and so are rungs the
detector does not trust: on a frame full of cloud the raw count goes *up*, so an
unguarded "most detections" objective would reliably pick the worst frame.

### 4. Focus, if the sweep says so

The sweep ends with a verdict on star width, because focus and exposure look
identical from the count alone:

- **under ~1.8 px** — stars are at or under the detector's blob floor and some
  are being missed for being *too small*, not too faint. Defocusing very
  slightly should raise the count.
- **over ~5 px** — soft; focusing raises both count and limiting magnitude.

Live meter, which is the tool for actually turning the ring:

```
python scripts/asi_focus.py 120 --camera 120 --exposure 2000000 --gain 100
```

Watch the `clip%` column. A clipped frame inverts the meter — a saturated region
is flat, contributes no high-frequency power, and *drags the score down*, so the
"best focus" becomes whichever frame happened to be darkest.

Re-run step 3 after refocusing: the best exposure moves with the focus.

### 5. Plate solve — once, then never again

```
python sentry/plate_solve.py local/allsky_frames/allsky_<stamp>.fits \
       --profile "allsky camera" --save --report
```

Minutes, not seconds — it is a blind search over orientation and focal length.
The camera is bolted down, so once is enough; everything afterwards verifies
against the stored solution and re-solves only if it stops fitting, which is
also how a knocked camera announces itself.

**The check that matters is not the residual.** It is whether the named stars
come out as *neighbours*. A correct solve names one contiguous patch of sky; a
coincidence names stars scattered across unrelated constellations.

### 6. Measure this lens's usable radius

`measured_radius_px` is deliberately **absent** from the config. The Kasa's 800
px was measured on a different optic and transfers to nothing. Until it is set,
the completeness table is computed over the whole frame, which mixes the sky
with the lens's own off-axis falloff — the `--report` output says so.

To set it: run `--report` and read where completeness collapses to zero with
radius. On the Kasa that was a clean cliff — 23-40% out to 800 px, then exactly
zero, with 127 visible catalogue stars beyond it and none matched.

### 7. Publish

```
python scripts/allsky_monitor.py --no-push      # check the JSON first
python scripts/allsky_monitor.py
```

Then, to run it every 5 minutes (needs UAC, like its siblings):

```
schtasks /Create /TN IrisAllSkyMonitor /SC MINUTE /MO 5 ^
  /TR "C:\Users\iriso\Documents\development\srt\scripts\allsky_monitor.bat" ^
  /RU iriso /NP /RL LIMITED /ST 00:03
```

Offset the start minute from `IrisSkyMonitor` and `IrisLiveSkymap`, which are
already 2.5 minutes apart — each job takes about a minute and the PC is also
running N.I.N.A.

The lab-site panel is in `taylorhogan.github.io/index.html`; it reads
`/live/allsky.json` and sits below the existing sky-camera view.

## The roof gate

Nothing from this camera is published as sky unless the roof is known open, and
"unknown" counts as not open. Two independent ways to know, because neither
covers the whole night:

- **The stars themselves.** If the stored plate solution puts 8 or more
  catalogue stars on top of detections, the camera is looking at open sky —
  there is no other way for that to happen, and the underside of a roof cannot
  fake it. Free, since the solve is already being verified for the limiting
  magnitude, and it is the only test that works during an imaging run.
- **The safety camera.** The observatory's authority on roof state, and it works
  in daylight and through cloud. But it reads the roof only when the scope is
  confirmed *parked*, so during an imaging run — exactly when the roof is
  certainly open — it cannot answer.

Vision verdicts are cached for 15 minutes, **except "open"**, which is always
taken fresh: a cached open would keep publishing pictures for the length of the
cache after the roof had shut, which is the one failure the gate exists to
prevent.

When the gate does not say open, the JPEG is not pushed at all — not merely
hidden on the far side. A picture sitting at a public URL is published whatever
the page chooses to render.

## Files

| Path | What |
| --- | --- |
| `sentry/asi_allsky.py` | camera, darks, auto-exposure |
| `scripts/allsky_monitor.py` | capture → measure → gate → publish |
| `scripts/allsky_monitor.bat` | the scheduled wrapper |
| `local/allsky_settings.json` | chosen exposure/gain + the sweep behind it |
| `local/allsky_darks/` | master darks, keyed exposure and gain |
| `local/allsky_frames/` | FITS archive + display JPEGs |
| `local/allsky_solution.json` | this camera's plate solution |
