#!/usr/bin/env python3
"""Capture and analyse the roof motor's current signature.

The roof move is command-triggered (``toggle_roof`` fires the relay then waits
for travel), so we don't have to *detect* motion — we sample the Shelly current
monitor at a fixed rate across the known travel window and store the resulting
``(t, current, power)`` trace as a "signature".

Good signatures are banked into a library (one folder per direction). A new
move is flagged anomalous when its scalar features fall outside the envelope of
the known-good runs (mean ± k·sigma) — e.g. a jammed or iced roof draws more
running current / energy, an early stop shortens the travel. This is the
capture-and-store + scalar-envelope half; curve-shape (DTW) comparison can be
layered on later once a few real traces exist.

CLI:
    python sentry/roof_current_signature.py capture --direction open --seconds 50
    python sentry/roof_current_signature.py features <signature.json>
    python sentry/roof_current_signature.py compare  <signature.json>
    python sentry/roof_current_signature.py label    <signature.json> --good
    python sentry/roof_current_signature.py list
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np

# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Project root
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from hardware_control import utl_shelly

logger = logging.getLogger(__name__)

# Where signatures live: sentry/roof_signatures/{good,unlabeled}/{direction}/*.json
_SIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roof_signatures")

DEFAULT_HZ = 20          # poll rate; ~30 Hz is achievable over HTTP, 20 leaves headroom
DEFAULT_SECONDS = 50     # cover the ~45s travel window in toggle_roof, plus margin
RUNNING_MARGIN_W = 5.0   # power above baseline+this is considered "motor running"
ANOMALY_SIGMA = 3.0      # flag features beyond mean ± this many sigma of good runs


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def capture_signature(direction=None, seconds=DEFAULT_SECONDS, hz=DEFAULT_HZ,
                      channel=None, stop_event=None):
    """Sample the current monitor for `seconds` at `hz`, return a signature dict.

    `direction` is "open"/"close"/None — metadata only (the relay just toggles).
    `stop_event` (threading.Event) ends sampling early when set.
    """
    period = 1.0 / hz
    t, current, power, voltage = [], [], [], []
    t0 = time.monotonic()
    deadline = t0 + seconds
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            break
        loop_start = time.monotonic()
        s = utl_shelly.read_current_monitor(channel)
        if s is not None:
            t.append(round(loop_start - t0, 4))
            current.append(s.get("current"))
            power.append(s.get("act_power"))
            voltage.append(s.get("voltage"))
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    sig = {
        "direction": direction or "unknown",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "channel": channel if channel is not None
                   else config.data()["hardware"].get("current_monitor_channel", 0),
        "hz": hz,
        "n_samples": len(t),
        "samples": {"t": t, "current": current, "power": power, "voltage": voltage},
    }
    sig["features"] = extract_features(sig)
    return sig


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(sig):
    """Reduce a raw trace to scalar, physically-interpretable features."""
    t = np.asarray(sig["samples"]["t"], dtype=float)
    p = np.asarray(sig["samples"]["power"], dtype=float)
    c = np.asarray(sig["samples"]["current"], dtype=float)
    if t.size < 3:
        return {"valid": False}

    # Baseline = idle draw before the motor runs (first ~1s of the window)
    baseline_mask = t < max(1.0, t[1])
    baseline = float(np.median(p[baseline_mask])) if baseline_mask.any() else float(p[0])

    running_mask = p > (baseline + RUNNING_MARGIN_W)
    if running_mask.any():
        run_t = t[running_mask]
        move_start = float(run_t.min())
        move_end = float(run_t.max())
        duration = move_end - move_start
        running_power = float(np.median(p[running_mask]))
        running_current = float(np.median(c[running_mask]))
    else:
        move_start = move_end = duration = 0.0
        running_power = running_current = 0.0

    # Energy over the whole capture via trapezoidal integration (W·s)
    energy_ws = float(_trapezoid(p, t)) if t.size > 1 else 0.0

    # Did current settle back to baseline by the end? (move completed in window)
    tail = p[t > (t[-1] - 1.0)]
    returned = bool(tail.size and np.median(tail) <= baseline + RUNNING_MARGIN_W)

    return {
        "valid": True,
        "baseline_w": round(baseline, 2),
        "peak_w": round(float(p.max()), 2),
        "peak_a": round(float(c.max()), 3),
        "running_w": round(running_power, 2),
        "running_a": round(running_current, 3),
        "move_duration_s": round(duration, 2),
        "energy_ws": round(energy_ws, 1),
        "returned_to_baseline": returned,
    }


# Features used for the anomaly envelope (booleans/strings excluded)
_ENVELOPE_KEYS = ("running_w", "running_a", "move_duration_s", "energy_ws", "peak_w")


# --------------------------------------------------------------------------- #
# Library storage
# --------------------------------------------------------------------------- #
def _dir_for(status, direction):
    return os.path.join(_SIG_ROOT, status, direction or "unknown")


def save_signature(sig, status="unlabeled"):
    """Write a signature to sentry/roof_signatures/{status}/{direction}/."""
    d = _dir_for(status, sig.get("direction"))
    os.makedirs(d, exist_ok=True)
    fname = f"{sig['timestamp'].replace(':', '-')}_{sig.get('direction', 'unknown')}.json"
    path = os.path.join(d, fname)
    with open(path, "w") as f:
        json.dump(sig, f, indent=2)
    logger.info("Saved roof signature: %s", path)
    return path


def load_library(direction=None, status="good"):
    """Load all known-good signatures (optionally for one direction)."""
    sigs = []
    base = os.path.join(_SIG_ROOT, status)
    if not os.path.isdir(base):
        return sigs
    dirs = [direction] if direction else os.listdir(base)
    for d in dirs:
        dpath = os.path.join(base, d)
        if not os.path.isdir(dpath):
            continue
        for fn in sorted(os.listdir(dpath)):
            if fn.endswith(".json"):
                with open(os.path.join(dpath, fn)) as f:
                    sigs.append(json.load(f))
    return sigs


# --------------------------------------------------------------------------- #
# Comparison (scalar envelope)
# --------------------------------------------------------------------------- #
def compare(sig, library=None, sigma=ANOMALY_SIGMA):
    """Compare a signature's features to the good-library envelope.

    Returns {is_anomaly, reasons[], detail{feature: {value, mean, std, z}}}.
    """
    if library is None:
        library = load_library(direction=sig.get("direction"))
    feats = sig.get("features") or extract_features(sig)

    if not feats.get("valid"):
        return {"is_anomaly": True, "reasons": ["capture invalid / too few samples"], "detail": {}}
    if len(library) < 2:
        return {"is_anomaly": False, "reasons": [],
                "detail": {}, "note": f"only {len(library)} good signature(s) — need >=2 to judge"}

    reasons, detail = [], {}
    for key in _ENVELOPE_KEYS:
        good_vals = [s["features"][key] for s in library
                     if s.get("features", {}).get("valid")]
        if len(good_vals) < 2:
            continue
        mean = float(np.mean(good_vals))
        std = float(np.std(good_vals))
        val = feats.get(key)
        # Guard against zero-variance: fall back to a 15% relative tolerance
        tol = max(std * sigma, abs(mean) * 0.15, 1e-6)
        z = (val - mean) / std if std > 1e-9 else float("inf") if abs(val - mean) > tol else 0.0
        detail[key] = {"value": val, "mean": round(mean, 2), "std": round(std, 2),
                       "z": round(z, 2) if np.isfinite(z) else None}
        if abs(val - mean) > tol:
            direction = "high" if val > mean else "low"
            reasons.append(f"{key} {val} is {direction} (good {mean:.1f}±{std:.1f})")

    if not feats.get("returned_to_baseline", True):
        reasons.append("current did not return to baseline — roof may not have completed travel")

    return {"is_anomaly": bool(reasons), "reasons": reasons, "detail": detail}


# --------------------------------------------------------------------------- #
# Background capture — for hooking into toggle_roof without blocking it
# --------------------------------------------------------------------------- #
def start_background_capture(direction=None, seconds=DEFAULT_SECONDS, hz=DEFAULT_HZ):
    """Begin sampling in a daemon thread. Returns a handle for finish_background_capture.

    Best-effort: never raises into the caller (roof safety path must not break).
    """
    handle = {"result": None, "stop": threading.Event(), "thread": None}

    def _run():
        try:
            handle["result"] = capture_signature(
                direction=direction, seconds=seconds, hz=hz, stop_event=handle["stop"])
        except Exception as e:  # noqa: BLE001 — observer must not crash the roof flow
            logger.warning("Roof signature capture failed: %s", e)

    th = threading.Thread(target=_run, name="roof-current-capture", daemon=True)
    handle["thread"] = th
    th.start()
    return handle


def finish_background_capture(handle, status="unlabeled", save=True):
    """Stop/join a background capture and optionally save it. Returns the signature or None."""
    if not handle or handle.get("thread") is None:
        return None
    try:
        handle["stop"].set()
        handle["thread"].join(timeout=10)
        sig = handle.get("result")
        if sig and save:
            save_signature(sig, status=status)
        return sig
    except Exception as e:  # noqa: BLE001
        logger.warning("Finishing roof signature capture failed: %s", e)
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_features(sig):
    f = sig.get("features", {})
    print(f"  direction={sig.get('direction')}  samples={sig.get('n_samples')}  "
          f"hz={sig.get('hz')}  @ {sig.get('timestamp')}")
    for k, v in f.items():
        print(f"    {k:22} {v}")


def main():
    ap = argparse.ArgumentParser(description="Roof motor current-signature tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="sample now and save (start this, then move the roof)")
    c.add_argument("--direction", choices=["open", "close"], default=None)
    c.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    c.add_argument("--hz", type=float, default=DEFAULT_HZ)
    c.add_argument("--status", default="unlabeled", help="library bucket to save into")

    f = sub.add_parser("features", help="print features of a saved signature")
    f.add_argument("path")

    cm = sub.add_parser("compare", help="compare a saved signature to the good library")
    cm.add_argument("path")

    lb = sub.add_parser("label", help="move a signature into the good/bad library")
    lb.add_argument("path")
    g = lb.add_mutually_exclusive_group(required=True)
    g.add_argument("--good", action="store_true")
    g.add_argument("--bad", action="store_true")

    sub.add_parser("list", help="list stored signatures")

    args = ap.parse_args()

    if args.cmd == "capture":
        print(f"Capturing {args.seconds:.0f}s @ {args.hz:.0f} Hz "
              f"(direction={args.direction or 'unknown'}) — trigger the roof now…", flush=True)
        sig = capture_signature(direction=args.direction, seconds=args.seconds, hz=args.hz)
        path = save_signature(sig, status=args.status)
        _print_features(sig)
        print(f"Saved -> {path}")

    elif args.cmd == "features":
        with open(args.path) as fh:
            _print_features(json.load(fh))

    elif args.cmd == "compare":
        with open(args.path) as fh:
            sig = json.load(fh)
        res = compare(sig)
        print(f"ANOMALY: {res['is_anomaly']}")
        for r in res.get("reasons", []):
            print(f"  - {r}")
        if res.get("note"):
            print(f"  ({res['note']})")

    elif args.cmd == "label":
        with open(args.path) as fh:
            sig = json.load(fh)
        path = save_signature(sig, status="good" if args.good else "bad")
        print(f"Labeled {'good' if args.good else 'bad'} -> {path}")

    elif args.cmd == "list":
        if not os.path.isdir(_SIG_ROOT):
            print("(no signatures yet)")
            return
        for status in sorted(os.listdir(_SIG_ROOT)):
            spath = os.path.join(_SIG_ROOT, status)
            for direction in sorted(os.listdir(spath)):
                dpath = os.path.join(spath, direction)
                if os.path.isdir(dpath):
                    n = len([x for x in os.listdir(dpath) if x.endswith(".json")])
                    print(f"  {status}/{direction}: {n}")


if __name__ == "__main__":
    main()
