# Pointing a Kasa camera

Findings from 2026-08-16 on `Iris cam` (KC410S(US), sw 2.3.27, 192.168.87.65).
Nothing here is wired into SRT; this records the interface so the next person
does not have to re-probe it. The same API is present on the KC420WS sky camera,
which is exactly why the last section exists.

## Both cameras pan and tilt, and it is not on the LAN

The KC410S is the "Kasa Spot Pan Tilt" and the KC420WS is its outdoor sibling.
Both have motors. Neither exposes them locally.

A full 65535-port sweep of each camera returns the same five ports, and every
one of them was followed to the end:

| port | what it is |
|------|-----------|
| 9999 (UDP) | discovery only — `system.get_sysinfo` answers, `get_time`, `get_dev_icon` and `cnCloud.get_info` are all silent |
| 9999 (TCP) | accepts a connection, returns zero bytes, ever |
| 10443 | HTTPS. **302 to `https://<host>:4444/` for every path**, and 4444 is closed |
| 17443 | HTTPS, `/vod/` playback routes — `/vod/data/mixstream` gives 503, everything else 404 |
| 18443 | HTTPS, no route found |
| 19443 | HTTPS, the live stream — `/https/stream/mixed?video=h264&audio=g711` |

Two traps in that table. The 10443 redirect is invisible through `requests`,
which reports a bare `ConnectionError` for the 302-plus-`Connection: close`; it
only appears if the bytes are written onto the TLS socket by hand. And 17443's
`/vod/data/mixstream` is the one path in a 123-path sweep that answered anything
other than 404, which is what identified that server's route table at all.

So: motion is **cloud-only**, same conclusion as the speaker in
[[SKY_CAMERA_AUDIO]].

## The interface, through the cloud passthrough

`hardware_control/kasa_cloud.py` already speaks `passthrough`, which relays
arbitrary JSON to a device. Cameras accept it. The module is
`smartlife.cam.ipcamera.ptz`:

```python
kc._call("passthrough", {
    "deviceId": dev["deviceId"],
    "requestData": json.dumps({
        "smartlife.cam.ipcamera.ptz": {"get_position": {}}})})
```

Four methods exist; every other name returns `-10008`:

| method | argument | reply |
|--------|----------|-------|
| `get_position` | `{}` | `{"x": -123, "y": 363, "err_code": 0}` |
| `get_capability` | `{}` | `{"max_x": 1934, "max_y": 387, "err_code": 0}` |
| `set_move` | `{"direction": "left", "speed": 1}` | `{"err_code": 0}` |
| `set_stop` | `{}` | `{"err_code": 0}` |

Also live on the same passthrough: `led.get_status` → `{"value": "off"}`,
`switch.get_is_enable` → `{"value": "on"}`, and `dateTime.get_status` →
timezone, area and `epoch_sec`.

`scripts/kasa_ptz.py` wraps all of this — `pos`, `move`, `goto`, `stop`, with
the sky camera refused by default.

### The direction words are left / right / **top** / **bottom**

Not `up`/`down`, which are rejected, and matched lowercase and exactly —
`Left` and `LEFT` both fail. `speed` is not optional either; `{"direction":
"left"}` alone is a format error.

### `speed` is step *size*, not rate

Valid 1–10; 0 and 50 are rejected. Measured on the y axis:

| speed | 1 | 2 | 3 | 5 | 10 |
|-------|---|---|---|---|----|
| units | 50 | 55 | 62 | 83 | 333 |

4 and 6–9 were never measured. The minimum step being 50 has a real
consequence: **`set_move` cannot make a small correction, and most targets are
off the lattice.** From y=387, `bottom` reaches 337 and `top` returns to 387;
363 lies between them and a greedy step-toward-the-target loop oscillates
forever. Reaching it means composing steps that cancel — `+83 +83 −50 −50 −62`
nets exactly `+4`. `kasa_ptz.solve()` does that search.

## Backlash: the reading does not tell you where the camera points

The single most important finding here, and it cost a real mistake.

`Iris cam` was returned to exactly `(-123, 363)` — the same numbers it started
at — and the frame was **121 px away** from the original. Re-entering the same
coordinate from the opposite side brought it back to **5 px**, with the phase
correlation response rising from 0.044 to 0.282.

So the encoder reading is *not* a position. It is repeatable only if the
approach direction is repeatable. A control pair of frames taken at the same
reading with **no motion in between** agreed to 2.8 px, which is what made this
easy to believe wrongly: the camera is perfectly stable until it moves.

Two rules follow.

* **Always finish a move from the same side.** `kasa_ptz.goto()` steps away and
  comes back from the positive direction before returning. Round-trip
  repeatability after that is about ±25 px on a 2560 px frame.
* **The positive side is the only choice on the tilt axis.** y=363 sits within
  one minimum step of the 387 limit, so every start position for a downward
  final approach (413, 418, 425, 446…) is beyond the end stop. Tilt can only
  ever be entered from below.

Vertical shift could not be measured reliably on this scene at all — the
zero-motion control disagreed with itself by 87 px, because the frame is
dominated by horizontal plank structure that leaves vertical correlation
ambiguous. Only the horizontal numbers above are trustworthy.

## Three error codes worth knowing

The firmware distinguishes them, and that is what made the API enumerable
without moving anything:

* **`-10008` "Unsupported API call"** — no such method. Sending every candidate
  name with `{}` maps the whole method table safely, because a method that does
  not exist cannot act.
* **`-41202` "The parameter format error"** — the method exists, the argument
  *shape* is wrong.
* **`-41203` "The parameter value error"** — the shape is right and the value is
  not. Note this covers **both** an unrecognised direction word and an axis at
  its end stop, so a refused move does not distinguish a typo from a limit.
  Check the spelling before concluding the camera cannot go further.

One caveat that cost a real move here: a **non-object** parameter — `"left"`,
`0` — returns a bare `{}`, with no `err_code` at all. That is neither success
nor a documented error, and code that treats "no error code" as success will be
wrong. Only `{"err_code": 0}` means the command was accepted.

The `-41202` reply does **not** encode whether a value was out of range. Keys
`x`/`y` were rejected identically at `0` (certainly in range) and at `999999`,
so it separates key names from values not at all. There is no way to confirm
`set_move`'s argument shape without a call that may move the camera — which is
how the format was ultimately found.

## The camera's own on/off, separate from its plug

`smartlife.cam.ipcamera.switch`, and it is supported on **both** cameras:

| method | argument | reply |
|--------|----------|-------|
| `get_is_enable` | `{}` | `{"value": "on"}` |
| `set_is_enable` | `{"value": "on"｜"off"}` | `{"err_code": 0}` |

The error text names the parameter outright — `-40800 "The parameter [value]
format error."` — which is how the shape was found without guessing. The LED is
the same shape under `smartlife.cam.ipcamera.led` / `set_status`.

**"Off" is a real disable, not a recording flag.** Measured 2026-08-16:

| camera_switch | the 19443 stream |
|---------------|------------------|
| on | HTTP 200, 76 video + 84 audio parts in 5 s |
| off | **HTTP 503, 19 bytes, no parts at all** |

So it stops the video **and the microphone together**. Anything treating this
camera as a safety sensor has to check `get_is_enable` before trusting it,
because a camera switched off in the Kasa app is indistinguishable from one
that is unreachable — and both the picture and the sound go at once.

## The sky camera has no motor at all

Correcting an earlier claim here that the KC420WS "answers all of the above".
That was inferred from the model being a pan/tilt product, not tested. It is
wrong:

```
Sky camera (KC420WS)   ptz.get_position     -10008 Unsupported API call
                       ptz.get_capability   -10008 Unsupported API call
Iris cam   (KC410S)    ptz.get_position     {"x": -123, "y": 363}
```

The sky camera is fixed. Its plate solution therefore **cannot** be invalidated
by a software pan, which removes the risk the rest of this document worries
about. `PROTECTED` in `scripts/kasa_ptz.py` stays anyway: it costs nothing, and
it is the wrong thing to remove on the strength of one probe.

The switch above **is** supported on it, though, so the all-sky feed can still
be killed remotely — that one is worth guarding.

That camera carries a **plate solution** — blind-solved, 1.5 px residual, axis
4° off zenith, 104° FOV, F=1617 px/rad — and everything derived from it: any
pixel to alt/az, the foliage mask, the horizon comparison, limiting magnitude.
All of it assumes the camera points where it pointed when it was solved. A
single `set_move` silently invalidates the lot, and `get_position` gives no
absolute reference to restore from unless the starting numbers were written down
first.

If it ever does need to move: record `get_position` **before** anything else,
and re-solve afterwards rather than trusting a restored coordinate. The backlash
section above is the reason that is not excessive caution — restoring the
coordinate demonstrably did *not* restore the pointing, and on the sky camera a
121 px error is about 5° on a 104° fisheye spread over 2560 px — far larger than
the 1.5 px residual the solution was fitted to.
