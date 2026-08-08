#!/usr/bin/env python3
"""Grab one still from the Kasa all-sky camera, locally, without the cloud.

The hard part of talking to a KC-series camera -- a TLS handshake OpenSSL 3
refuses by default, HTTP Basic against the TP-Link *cloud* account, and a
multipart body that interleaves H.264 with G.711 and has to be reassembled
before a decoder will look at it -- already lives in scripts/probe_kasa_camera.
This module deliberately calls into that rather than restating any of it: two
implementations of the same handshake would drift, and the one over there is
the one that has been shown to work against the actual camera.

Usage:  python sentry/sky_camera.py [out.jpg]
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from scripts.probe_kasa_camera import probe_kc_stream


def credentials(cfg=None):
    """(username, password) for the camera stream: private config, then env.

    Read from the top-level "sky camera auth" section rather than from
    "sky camera", and that separation is load-bearing. config.data() merges the
    two configs with a plain dict.update(), which is shallow: a "sky camera"
    key in config_private.py would REPLACE the public section outright, taking
    the camera host and every tuned detector threshold with it. Credentials
    therefore live under a key the public config never defines.
    """
    cfg = cfg or config.data()
    auth = cfg.get("sky camera auth", {})
    cam = cfg.get("sky camera", {})
    user = auth.get("username") or cam.get("username") or os.environ.get("KASA_USERNAME")
    pw = auth.get("password") or cam.get("password") or os.environ.get("KASA_PASSWORD")
    return user, pw


def capture(out_path=None, timeout=15.0):
    """Write one frame to out_path. Returns the Path, or None on failure.

    Never raises for an unreachable or unauthenticated camera: this runs on a
    timer, and a camera that is down is a thing to report, not a crash.
    """
    cfg = config.data()
    cam = cfg.get("sky camera", {})
    host = cam.get("host")
    if not host:
        print("sky camera: no host configured")
        return None

    user, pw = credentials(cfg)
    if not user or not pw:
        print("sky camera: no credentials. Set 'sky camera'"
              " username/password in configs/config_private.py, or export"
              " KASA_USERNAME / KASA_PASSWORD.")
        return None

    if out_path is None:
        root = Path(__file__).resolve().parents[1]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = root / cam.get("capture_dir", "local/sky_frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("sky_" + stamp + ".jpg")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        url = probe_kc_stream(host, user, pw, snapshot_path=str(out_path),
                              timeout=timeout)
    except Exception as exc:                      # network, TLS, decode, ...
        print("sky camera: capture raised " + type(exc).__name__ + ": " + str(exc)[:160])
        return None

    if not url or not out_path.exists() or out_path.stat().st_size == 0:
        print("sky camera: no frame captured from " + str(host))
        return None

    # grab_frame leaves the reassembled elementary stream beside the still.
    # It is only useful for debugging a decode failure, and at four frames an
    # hour it would quietly become the largest thing in local/.
    h264 = out_path.with_suffix(".h264")
    try:
        if h264.exists():
            h264.unlink()
    except OSError:
        pass
    return out_path


def capture_burst(seconds=None, out_path=None, cfg=None):
    """Record a few seconds of consecutive frames. Returns (path, n_frames).

    Kept as the RAW H.264 elementary stream rather than decoded stills: 8
    seconds is about 1.3 MB this way against roughly 71 MB as JPEGs, 55x
    smaller for identical pixels, and it decodes back on demand. Storage is the
    only reason bursts are affordable at all.

    Why a burst is worth taking during weather: at 15 fps a star drifts 0.008 px
    between adjacent frames, so anything that moves is not a star. Every frame
    also holds an entirely fresh set of raindrops, which makes a single shower
    thousands of independent drop samples even though it is only one weather
    event. Three stills five minutes apart cannot show that.
    """
    import time
    from scripts.probe_kasa_camera import (legacy_tls_session, _boundary_of,
                                           _split_parts)
    from requests.auth import HTTPBasicAuth

    cfg = cfg or config.data()
    cam = cfg.get("sky camera", {})
    seconds = float(seconds if seconds is not None else cam.get("burst_seconds", 8.0))
    host = cam.get("host")
    user, pw = credentials(cfg)
    if not host or not user or not pw:
        return None, 0

    if out_path is None:
        root = Path(__file__).resolve().parents[1]
        d = root / cam.get("burst_dir", "local/sky_bursts")
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = d / ("burst_" + stamp + ".h264")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://%s:19443/https/stream/mixed?video=h264&audio=g711" % host
    try:
        sess = legacy_tls_session()
        resp = sess.get(url, auth=HTTPBasicAuth(user, pw), verify=False,
                        stream=True, timeout=(15, 12))
        if resp.status_code != 200:
            resp.close()
            return None, 0
        delim = b"--" + (_boundary_of(resp.headers.get("Content-Type", "")) or "").encode()
        if delim == b"--":
            resp.close()
            return None, 0
        buf = bytearray()
        first = None
        deadline = None
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if chunk:
                if first is None:
                    first = time.monotonic()
                    deadline = first + seconds
                buf.extend(chunk)
            if deadline and time.monotonic() > deadline:
                break
            if len(buf) > 32 * 1024 * 1024:
                break
        resp.close()
    except Exception as exc:
        print("burst: capture raised " + type(exc).__name__ + ": " + str(exc)[:120])
        return None, 0

    video = bytearray()
    nparts = 0
    for headers, body in _split_parts(buf, delim):
        if "h264" in headers.get("content-type", ""):
            video.extend(body)
            nparts += 1
    if not video:
        return None, 0
    out_path.write_bytes(bytes(video))
    return out_path, nparts


def decode_burst(path, limit=None):
    """Decode a stored burst back to grayscale frames. For offline analysis."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(float))
        if limit and len(frames) >= limit:
            break
    cap.release()
    return frames


def prune_bursts(keep=None, cfg=None):
    """Rolling cap on stored bursts. They are small but not free."""
    cfg = cfg or config.data()
    cam = cfg.get("sky camera", {})
    keep = int(keep if keep is not None else cam.get("keep_bursts", 500))
    root = Path(__file__).resolve().parents[1]
    d = root / cam.get("burst_dir", "local/sky_bursts")
    if not d.is_dir() or keep <= 0:
        return 0
    files = sorted(d.glob("burst_*.h264"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    dropped = 0
    for old in files[keep:]:
        try:
            old.unlink()
            dropped += 1
        except OSError:
            pass
    return dropped


def prune(keep=None):
    """Hold the rolling frame archive to `keep` newest files."""
    cfg = config.data()
    cam = cfg.get("sky camera", {})
    keep = int(keep if keep is not None else cam.get("keep_frames", 400))
    root = Path(__file__).resolve().parents[1]
    d = root / cam.get("capture_dir", "local/sky_frames")
    if not d.is_dir() or keep <= 0:
        return 0
    frames = sorted(d.glob("sky_*.jpg"), key=lambda p: p.stat().st_mtime,
                    reverse=True)
    dropped = 0
    for old in frames[keep:]:
        try:
            old.unlink()
            dropped += 1
        except OSError:
            pass
    return dropped


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    got = capture(dest)
    print("captured:", got if got else "FAILED")
    sys.exit(0 if got else 1)
