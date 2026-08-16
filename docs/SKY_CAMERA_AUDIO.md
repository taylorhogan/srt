# The sky camera has a microphone, and we throw it away

Findings from 2026-08-16. Nothing here is wired up; this is a record of what the
KC420WS exposes locally so the next person does not have to re-probe it.

## The microphone works, and it is already arriving

`sky_camera.capture_burst()` opens

```
https://<host>:19443/https/stream/mixed?video=h264&audio=g711
```

and splits the multipart body. Line 167 then keeps only the parts whose
content-type contains `h264`, so every G.711 part is decoded out of the buffer
and discarded. A six-second capture yields:

```
HTTP 200  multipart/x-mixed-replace
parts seen: {'video/x-h264': 91, 'audio/g711u': 102}
audio bytes: 48960   ->  6.12 s at 8000 Hz mono
```

No new endpoint, no extra authentication, no second connection. The audio is on
the stream the 5-minute monitor already opens.

Format is **G.711 mu-law, 8-bit, 8 kHz, mono**. Decode it by hand — the stdlib
`audioop` that would normally do this was removed in Python 3.13:

```python
u = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
u = ~u & 0xFF
sign, exponent, mantissa = u & 0x80, (u >> 4) & 0x07, u & 0x0F
sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
pcm16 = np.where(sign, -sample, sample).clip(-32768, 32767).astype(np.int16)
```

**Baseline, still night, 05:30:** rms 5, peak 88 against a full scale of 32767 —
effectively silence. That is what makes it promising as a detector: the noise
floor is near zero, so anything real is unmissable.

## The speaker is NOT reachable locally

Probed and ruled out, so it is not worth retrying:

* **Only two ports are open** on the camera: 9999 and 19443. No 554 (RTSP), no
  2020 (ONVIF), no 80, no 443.
* **Ten plausible talk-back paths on 19443 all return 404** — `/https/stream/talk`,
  `/https/talk`, `/talk`, `/https/stream/talkback`, `/https/stream/audio`,
  `/https/audio`, `/https/stream/send`, `/https/stream/speaker`,
  `/https/stream/twoway`. The single 200 was `/https/stream/mixed?talk=1`, which
  is just the normal stream ignoring an unknown query parameter.
* **The stream is a read-only GET.** Multipart flows down; there is no upload
  channel on it.
* **`get_sysinfo` advertises no audio module** — 29 leaf fields, and the only
  match for mic/audio/speaker/volume/talk is `mic_mac`, which is TP-Link's name
  for the device MAC (`BC071D29970D`, identical to `mac`), not a microphone
  control.

Two-way audio in the Kasa app therefore almost certainly goes via TP-Link's
cloud. The local interface is a monitoring surface, not a control one.

That was confirmed from the other direction on 2026-08-16: the camera's pan/tilt
motor is not on the LAN either, but *is* reachable through the cloud passthrough
`hardware_control/kasa_cloud.py` already speaks. See [[KASA_CAMERA_PTZ]], which
also carries the full port map — the "only two ports are open" claim above was
made against a six-port list, and a full 65535-port sweep finds 10443, 17443 and
18443 as well.

## Note: the camera answers the legacy protocol over UDP, not TCP

Unlike the plugs, TCP 9999 accepts a connection and then returns nothing. The
same XOR-encoded `{"system":{"get_sysinfo":{}}}` sent over **UDP 9999** replies
in full. Worth knowing before concluding the camera is unresponsive.

Useful fields it does return: `alias`, `model` (`KC420WS(US)`), `sw_ver`
(`2.3.26 Build 20240510 rel.32915`), `resolution` (`2560x1440`), `rssi`,
`led_status`, `camera_switch`, `status`, `bind_status`, and its GPS position.
`rssi` in particular is a free health metric for a camera that has already
dropped off the network once.

## Why this might be worth building

The rain detector currently works from central-sky pixel motion. It is blind in
daylight, gives ~0 lead at dusk, and lost 26% of its lead time to camera dropouts
in August. **Rain is loud on a camera housing**, and an acoustic channel has none
of those failure modes — it works in daylight, at dusk, and through cloud.

More importantly it fails *differently* from a vision channel, which is the
property [[ROOF_STATE_SENSING]] argues for: two sensors that fail the same way
are one sensor.

The machinery already exists. `sentry/audio_classify.py` builds mel spectrograms
and scores them by MSE against a labelled library, for roof movement. This is the
same problem with a different microphone.

The change is small: collect the `audio/g711u` parts alongside the video in
`capture_burst()`, which already runs on rain suspicion, and write a WAV beside
the `.h264`. That costs nothing extra on the wire — the bytes are already being
received and dropped.
