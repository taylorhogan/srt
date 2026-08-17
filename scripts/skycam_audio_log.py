"""
skycam_audio_log.py
Sample the sky camera's microphone on a cadence and log what it hears.

Built to catch a daytime rain onset. The vision rain detector is blind above
the horizon — `moving_pct` reads `none` in daylight — so the only channel that
can see a daytime shower is sound. This collects the data that a daytime
detector would be trained on, with the transition included: starting before the
rain matters more than catching it once it has settled, because the onset is
where any usable lead time lives.

    python skycam_audio_log.py --seconds 15 --every 180 --hours 12

Writes `local/skycam_audio/` with a WAV per sample and one features row per
sample in `local/skycam_audio_log.jsonl`. Features are logged always and are
tiny; WAVs are capped and rotate, because they are not.

Label afterwards from the wall clock — the operator knows when it started
raining, which is the whole point of doing this on a day someone is watching.
"""
import argparse
import json
import os
import sys
import time
import wave
from datetime import datetime, timezone

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

from configs import config
from scripts.probe_kasa_camera import (legacy_tls_session, _boundary_of,
                                       _split_parts)
from sentry.sky_camera import credentials

OUT_DIR = "local/skycam_audio"
LOG_PATH = "local/skycam_audio_log.jsonl"
RATE = 8000

# Bands chosen for what separates rain from the other things an outdoor
# microphone hears. Rain on a plastic housing is broadband and weighted high;
# wind is dominated by the bottom band; road noise sits low-mid. Reporting the
# shape rather than a single loudness is what lets those be told apart later.
BANDS = [(50, 300), (300, 800), (800, 1500), (1500, 2500), (2500, 3900)]


def ulaw_to_pcm16(raw):
    u = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    u = ~u & 0xFF
    sign, exponent, mantissa = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.clip(np.where(sign, -sample, sample), -32768, 32767).astype(np.int16)


def grab_audio(host, user, pw, seconds):
    url = "https://%s:19443/https/stream/mixed?video=h264&audio=g711" % host
    sess = legacy_tls_session()
    resp = sess.get(url, auth=HTTPBasicAuth(user, pw), stream=True,
                    verify=False, timeout=(15, 12))
    if resp.status_code != 200:
        resp.close()
        return None
    delim = b"--" + (_boundary_of(resp.headers.get("Content-Type", "")) or "").encode()
    buf = bytearray()
    t0 = time.monotonic()
    try:
        for chunk in resp.iter_content(chunk_size=32 * 1024):
            buf.extend(chunk)
            if time.monotonic() - t0 > seconds:
                break
    finally:
        resp.close()
    audio = bytearray()
    for headers, body in _split_parts(bytes(buf), delim):
        ctype = headers.get("content-type", "")
        if "g711" in ctype or "audio" in ctype:
            audio.extend(body)
    return bytes(audio) or None


def features(pcm):
    """Loudness plus spectral shape, as fractions of total power.

    Fractions, not means: an earlier pass divided per-band MEANS by their sum,
    which is not a normalisation and produced 'percentages' totalling over 700.
    Power in the band over total power is the quantity that actually sums to 1.
    """
    f = pcm.astype(np.float64)
    out = {"rms": float(np.sqrt((f ** 2).mean())), "peak": int(np.abs(pcm).max())}
    n = 8192
    if len(f) >= n:
        seg = f[:n] * np.hanning(n)
        power = np.abs(np.fft.rfft(seg)) ** 2
        freq = np.fft.rfftfreq(n, 1.0 / RATE)
        total = float(power[(freq >= 50) & (freq < 3900)].sum()) + 1e-12
        for lo, hi in BANDS:
            frac = float(power[(freq >= lo) & (freq < hi)].sum()) / total
            out["b%d_%d" % (lo, hi)] = round(frac, 4)
        # Spectral centroid: one number for "how high-pitched is this", which
        # is the axis rain (hiss) and wind (rumble) differ on most.
        band = (freq >= 50) & (freq < 3900)
        out["centroid_hz"] = round(float((freq[band] * power[band]).sum()
                                         / (power[band].sum() + 1e-12)), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--every", type=float, default=180.0, help="seconds between samples")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--keep", type=int, default=400, help="rolling cap on WAVs")
    args = ap.parse_args()

    cfg = config.data()
    host = cfg["sky camera"]["host"]
    user, pw = credentials(cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    end = time.time() + args.hours * 3600
    print("logging %s every %.0fs for %.1f h -> %s"
          % (host, args.every, args.hours, OUT_DIR))

    n = 0
    while time.time() < end:
        t = datetime.now(timezone.utc)
        row = {"t": t.isoformat(timespec="seconds")}
        try:
            raw = grab_audio(host, user, pw, args.seconds)
            if raw:
                pcm = ulaw_to_pcm16(raw)
                stamp = t.strftime("%Y%m%dT%H%M%SZ")
                path = os.path.join(OUT_DIR, "aud_%s.wav" % stamp)
                with wave.open(path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(pcm.tobytes())
                row.update(features(pcm))
                row["file"] = os.path.basename(path)
                row["dur_s"] = round(len(pcm) / RATE, 2)
                n += 1
                print("  %s  rms %6.1f  peak %6d  centroid %6.1f Hz  %s"
                      % (t.astimezone().strftime("%H:%M:%S"), row["rms"],
                         row["peak"], row.get("centroid_hz", 0), row["file"]))
            else:
                row["error"] = "no audio parts"
                print("  %s  no audio" % t.astimezone().strftime("%H:%M:%S"))
        except requests.RequestException as exc:
            row["error"] = type(exc).__name__
            print("  %s  %s" % (t.astimezone().strftime("%H:%M:%S"), type(exc).__name__))

        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Rotate WAVs but never the log: the features are what a classifier
        # gets trained on and they cost almost nothing to keep.
        wavs = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".wav"))
        for old in wavs[:-args.keep]:
            try:
                os.remove(os.path.join(OUT_DIR, old))
            except OSError:
                pass

        time.sleep(max(0.0, args.every - args.seconds))
    print("done, %d samples" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
