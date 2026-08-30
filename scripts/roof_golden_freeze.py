#!/usr/bin/env python3
"""Freeze golden reference sets for roof audio and current signatures.

    python scripts/roof_golden_freeze.py            # seed (refuses if golden exists)
    python scripts/roof_golden_freeze.py --refresh  # re-anchor after mechanical service

Copies the OLDEST N good samples per direction into golden/ for both systems:

    sentry/roof_audio/golden/{open,close}/          (png + wav pairs)
    sentry/roof_signatures/golden/{open,close}/     (json)

Oldest, because the rolling/accumulating good libraries have been absorbing
the roof's drift (logged good peak_w mean rose 351 W -> 398 W in twelve days,
Aug 2026) -- the earliest surviving samples are the closest thing on disk to
"the roof when healthy". Golden sets are compared against by
audio_classify.classify() and roof_current_signature.judge_and_record().

Run with --refresh ONLY right after mechanical service (lubrication, wheel
replacement), when today's roof IS the new healthy reference: it replaces
golden with the NEWEST N good samples instead.
"""
import os
import shutil
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N_PER_DIRECTION = 10

AUDIO_GOOD = os.path.join(_ROOT, "sentry", "roof_audio", "good")
AUDIO_GOLDEN = os.path.join(_ROOT, "sentry", "roof_audio", "golden")
SIG_GOOD = os.path.join(_ROOT, "sentry", "roof_signatures", "good")
SIG_GOLDEN = os.path.join(_ROOT, "sentry", "roof_signatures", "golden")


def _freeze(good_root, golden_root, exts, newest):
    for direction in ("open", "close"):
        src = os.path.join(good_root, direction)
        dst = os.path.join(golden_root, direction)
        if not os.path.isdir(src):
            print(f"  {direction}: no good library at {src} — skipped")
            continue
        # Filenames start with the capture timestamp, so name order is age order.
        stems = sorted({os.path.splitext(f)[0] for f in os.listdir(src)
                        if os.path.splitext(f)[1] in exts})
        picked = stems[-N_PER_DIRECTION:] if newest else stems[:N_PER_DIRECTION]
        os.makedirs(dst, exist_ok=True)
        copied = 0
        for stem in picked:
            for ext in exts:
                s = os.path.join(src, stem + ext)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(dst, stem + ext))
                    copied += 1
        age = "newest" if newest else "oldest"
        span = f"{picked[0][:10]} .. {picked[-1][:10]}" if picked else "none"
        print(f"  {direction}: froze {len(picked)} {age} samples ({copied} files), {span}")


def main() -> int:
    refresh = "--refresh" in sys.argv
    for golden in (AUDIO_GOLDEN, SIG_GOLDEN):
        populated = any(os.path.isdir(os.path.join(golden, d)) and
                        os.listdir(os.path.join(golden, d))
                        for d in ("open", "close"))
        if populated:
            if not refresh:
                print(f"golden set already exists at {golden} — this is the frozen "
                      "healthy reference and is not touched without --refresh "
                      "(which is for right after mechanical service only)")
                return 1
            shutil.rmtree(golden)
    print("roof audio golden:")
    _freeze(AUDIO_GOOD, AUDIO_GOLDEN, {".png", ".wav"}, newest=refresh)
    print("current signature golden:")
    _freeze(SIG_GOOD, SIG_GOLDEN, {".json"}, newest=refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
