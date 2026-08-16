"""
iriscam_record.py
Record video AND microphone from an indoor Kasa camera, in one capture.

Both arrive on the same multipart stream the camera already serves on 19443,
interleaved as `video/x-h264` and `audio/g711u` parts. `sentry/sky_camera.py`
keeps only the video and drops every audio part on the floor; this keeps both,
so a roof move can be watched and heard from a single connection.

    python iriscam_record.py --seconds 120 --label roof-open

Writes into `local/iriscam_rec/<label>_<timestamp>/`:
    audio.wav     8 kHz mono, decoded from G.711 mu-law
    video.h264    raw Annex-B elementary stream
    frame_*.jpg   stills at --frame-every seconds
    envelope.csv  RMS per bin against wall clock

Parts are timestamped as they arrive rather than after the fact, so the audio
envelope and the frames share one clock and a sound can be tied to a picture.
"""
import argparse
import csv
import json
import os
import sys
import time
import wave
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import requests
import urllib3
from requests.auth import HTTPBasicAuth

from configs import config
from scripts.probe_kasa_camera import (legacy_tls_session, _boundary_of,
                                       _split_parts)
from sentry.sky_camera import credentials

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RATE = 8000                     # G.711 is always 8 kHz mono
DEFAULT_HOST = "192.168.87.65"  # Iris cam


def ulaw_to_pcm16(raw):
    """mu-law (G.711) -> signed 16-bit PCM.

    Done by hand because the stdlib `audioop` that would normally do this was
    removed in Python 3.13.
    """
    u = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    u = ~u & 0xFF
    sign, exponent, mantissa = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.clip(np.where(sign, -sample, sample), -32768, 32767).astype(np.int16)


def capture(host, user, pw, seconds, timeout=20):
    """Buffer the mixed stream for *seconds*, keeping arrival times.

    Returns (buffer, checkpoints, t0) where checkpoints is a list of
    (byte_offset, monotonic_time) recorded as each chunk landed. Without it
    there is no way to place a part on the wall clock: raw Annex-B carries no
    timestamps, and the audio parts are only 60 ms each.
    """
    url = "https://%s:19443/https/stream/mixed?video=h264&audio=g711" % host
    session = legacy_tls_session()
    resp = session.get(url, auth=HTTPBasicAuth(user, pw), stream=True,
                       verify=False, timeout=(timeout, 15))
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    boundary = _boundary_of(ctype)
    if not boundary:
        resp.close()
        raise RuntimeError("no multipart boundary in %r" % ctype)

    buf = bytearray()
    checkpoints = []
    t0 = time.monotonic()
    deadline = t0 + seconds
    last_report = t0
    try:
        for chunk in resp.iter_content(chunk_size=32 * 1024):
            if chunk:
                buf.extend(chunk)
                checkpoints.append((len(buf), time.monotonic()))
            now = time.monotonic()
            if now - last_report >= 10:
                print("  %5.0fs  %6.1f MB" % (now - t0, len(buf) / 1e6))
                last_report = now
            if now > deadline:
                break
    except requests.RequestException as exc:
        print("  stream stopped early: %s: %s" % (type(exc).__name__, exc))
    finally:
        resp.close()
    return bytes(buf), checkpoints, t0


def _time_at(checkpoints, offset, t0):
    """Wall-clock seconds since t0 for a byte at *offset*."""
    lo, hi = 0, len(checkpoints) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if checkpoints[mid][0] < offset:
            lo = mid + 1
        else:
            hi = mid
    return checkpoints[lo][1] - t0 if checkpoints else 0.0


def split(buf, checkpoints, t0, boundary_bytes):
    """(audio bytes, video bytes, audio part times, video part count)."""
    audio, video, times = bytearray(), bytearray(), []
    n_video = 0
    offset = 0
    for headers, body in _split_parts(buf, boundary_bytes):
        # Locate this part in the buffer to date it. Searching forward from the
        # last hit keeps this linear rather than quadratic on a long capture.
        found = buf.find(body, offset) if body else -1
        if found >= 0:
            offset = found + len(body)
        kind = headers.get("content-type", "")
        if "h264" in kind:
            video.extend(body)
            n_video += 1
        elif "g711" in kind or "audio" in kind:
            times.append((len(audio), _time_at(checkpoints, max(found, 0), t0)))
            audio.extend(body)
    return bytes(audio), bytes(video), times, n_video


def envelope(pcm, times, bin_s=0.5):
    """[(t, rms, peak)] per bin, on the capture's own clock."""
    if not times:
        return []
    rows = []
    # Map sample index -> time by interpolating between the part timestamps.
    idx = np.array([t[0] for t in times], dtype=np.float64)
    sec = np.array([t[1] for t in times], dtype=np.float64)
    total = sec[-1] if len(sec) else 0.0
    n_bins = max(1, int(total / bin_s))
    for b in range(n_bins):
        t_start, t_end = b * bin_s, (b + 1) * bin_s
        i0 = int(np.interp(t_start, sec, idx))
        i1 = int(np.interp(t_end, sec, idx))
        chunk = pcm[i0:i1]
        if len(chunk) == 0:
            continue
        f = chunk.astype(np.float64)
        rows.append((t_start, float(np.sqrt((f ** 2).mean())),
                     int(np.abs(chunk).max())))
    return rows


def decode_frames(h264_path, out_dir, every_s, fps_hint):
    """Save a still every *every_s* seconds from the raw stream."""
    try:
        import cv2
    except ImportError:
        print("  OpenCV not available; no frames extracted")
        return []
    cap = cv2.VideoCapture(h264_path)
    if not cap.isOpened():
        print("  OpenCV could not open %s" % h264_path)
        return []
    saved, index, next_at = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps_hint
        if t >= next_at:
            path = os.path.join(out_dir, "frame_%06.1fs.jpg" % t)
            cv2.imwrite(path, frame)
            saved.append((t, path, float(frame.mean())))
            next_at += every_s
        index += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--label", default="rec")
    ap.add_argument("--frame-every", type=float, default=5.0)
    ap.add_argument("--outdir", default="local/iriscam_rec")
    args = ap.parse_args()

    user, pw = credentials(config.data())
    if not user:
        print("no TP-Link credentials; the 19443 stream needs them")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.outdir, "%s_%s" % (args.label, stamp))
    os.makedirs(out_dir, exist_ok=True)
    print("recording %.0fs from %s -> %s" % (args.seconds, args.host, out_dir))
    started = datetime.now()

    # Bank where the camera was pointing. A fixed pixel region only means
    # anything for one pointing, and this camera pans and tilts -- so a
    # recording that does not record its own position cannot be scored later.
    # This is a coarse check only: the encoder does NOT pin the framing (frames
    # at the identical reading have measured 121 px apart), which is why
    # roof_region_stats.py registers every frame against a reference image as
    # well. Best-effort; a cloud hiccup must not cost the recording.
    meta = {"host": args.host, "started": started.astimezone().isoformat(
        timespec="seconds"), "seconds": args.seconds, "label": args.label}
    try:
        from scripts import kasa_ptz
        dev = kasa_ptz._device("Iris cam")
        meta["ptz_position"] = list(kasa_ptz.position(dev))
        meta["ptz_capability"] = list(kasa_ptz.capability(dev))
        print("  camera at %s" % (meta["ptz_position"],))
    except Exception as exc:
        meta["ptz_error"] = "%s: %s" % (type(exc).__name__, exc)
        print("  could not read camera position (%s)" % type(exc).__name__)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    buf, checkpoints, t0 = capture(args.host, user, pw, args.seconds)
    if not buf:
        print("no bytes received")
        return 1

    # Re-read the boundary the same way capture() did.
    session = legacy_tls_session()
    resp = session.get("https://%s:19443/https/stream/mixed?video=h264&audio=g711"
                       % args.host, auth=HTTPBasicAuth(user, pw), stream=True,
                       verify=False, timeout=(20, 15))
    boundary = _boundary_of(resp.headers.get("Content-Type", ""))
    resp.close()
    delim = b"--" + boundary.encode()

    audio, video, times, n_video = split(buf, checkpoints, t0, delim)
    elapsed = checkpoints[-1][1] - t0 if checkpoints else 0.0
    print("\n  %.1f MB buffered over %.1fs" % (len(buf) / 1e6, elapsed))
    print("  audio %d bytes (%.1fs)   video %d bytes"
          % (len(audio), len(audio) / RATE, len(video)))

    if audio:
        pcm = ulaw_to_pcm16(audio)
        wav_path = os.path.join(out_dir, "audio.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())
        f = pcm.astype(np.float64)
        print("  wrote %s  rms %.0f  peak %d (full scale 32767)"
              % (wav_path, float(np.sqrt((f ** 2).mean())), int(np.abs(pcm).max())))

        rows = envelope(pcm, times)
        with open(os.path.join(out_dir, "envelope.csv"), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "rms", "peak"])
            writer.writerows(rows)
        if rows:
            quiet = float(np.median([r[1] for r in rows]))
            loud = sorted(rows, key=lambda r: -r[1])[:6]
            print("\n  median rms %.0f; loudest half-second bins:" % quiet)
            for t, rms, peak in sorted(loud):
                bar = "#" * min(50, int(rms / max(quiet, 1)))
                print("    t=%6.1fs  rms %6.0f  peak %6d  %s" % (t, rms, peak, bar))

    if video:
        h264_path = os.path.join(out_dir, "video.h264")
        with open(h264_path, "wb") as fh:
            fh.write(video)
        # No timestamps in Annex-B. One multipart part is one frame, so the
        # part count over the measured wall clock IS the rate -- counting NAL
        # start codes instead undercounts badly and stretched a 20s capture
        # into 195s of frame timestamps.
        fps = (n_video / elapsed) if elapsed else 15.0
        print("\n  wrote %s (%.2f fps from %d video parts over %.1fs)"
              % (h264_path, fps, n_video, elapsed))
        frames = decode_frames(h264_path, out_dir, args.frame_every, fps)
        print("  %d stills" % len(frames))
        for t, path, mean in frames[:40]:
            print("    t=%6.1fs  mean %5.1f  %s" % (t, mean, os.path.basename(path)))

    print("\nstarted %s, %s" % (started.strftime("%H:%M:%S"), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
