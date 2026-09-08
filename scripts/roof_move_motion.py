#!/usr/bin/env python3
"""roof_move_motion.py -- when does the panel actually move, relative to the motor?

    python scripts/roof_move_motion.py                      # newest clip
    python scripts/roof_move_motion.py --direction close    # newest close
    python scripts/roof_move_motion.py <clip.h264>          # a particular one

Every roof move since 2026-09-08 keeps its whole Kasa-camera clip and the
audio from the same stream (sentry/roof_frames_kasa/clips, written by
kasa_roof_frames.save_move_clip). This reads one and answers, in seconds
from the start of the capture:

    motor start   the sound: first 0.1 s window whose RMS is MOTOR_FACTOR x
                  the capture's quiet floor
    panel start   the picture: first frame whose motion energy inside the
                  roof aperture (kasa_state.REGION_*) exceeds MOTION_FACTOR x
                  the pre-move floor
    stall         panel start - motor start

WHY. Every close since 5 Sep draws ~500 W for exactly its first second and
takes one second longer than an August close, while the stops have not
moved. If the panel stands still for that second while the motor strains,
the stall here reads ~1 s; an August-style move reads ~0. The 3 July
gear-not-engage fault was 45 s of this.

Writes beside the clip: <stamp>_motion.png (audio envelope, motion energy,
the two onsets marked) and <stamp>_breakaway.jpg (a strip of frames from
0.5 s before the motor starts to 2 s after, so the moment can be watched).
"""
import argparse
import glob
import os
import sys
import wave

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np

MOTOR_FACTOR = 6.0      # RMS over the quiet floor that means "motor running"
MOTION_FACTOR = 6.0     # motion energy over the pre-move floor that means "panel moving"
FPS = 15.0              # the camera's nominal rate; the clip carries no timestamps


def _clips(direction=None):
    from sentry.kasa_roof_frames import CLIP_DIR
    pat = "*_%s.h264" % direction if direction else "*.h264"
    return sorted(glob.glob(os.path.join(CLIP_DIR, pat)))


def audio_onset(wav_path, factor=MOTOR_FACTOR):
    """(t_motor_s, envelope_t, envelope_rms, floor) from the capture's audio."""
    with wave.open(wav_path, "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    win = int(rate * 0.1)
    n = len(pcm) // win
    rms = np.sqrt((pcm[:n * win].reshape(n, win) ** 2).mean(axis=1))
    t = (np.arange(n) + 0.5) * 0.1
    floor = np.percentile(rms[: max(5, n // 5)], 20) + 1e-6
    above = np.where(rms > factor * floor)[0]
    return (float(t[above[0]]) if len(above) else None), t, rms, floor


def motion_energy(clip_path, region=None, fps=FPS):
    """Per-frame mean absolute difference inside *region* (y0,y1,x0,x1)."""
    import cv2
    cap = cv2.VideoCapture(clip_path)
    prev = None
    energy, frames = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if region:
            y0, y1, x0, x1 = region
            g = g[y0:y1, x0:x1]
        g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
        energy.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
        frames.append(fr)
        prev = g
    cap.release()
    t = np.arange(len(energy)) / fps
    return t, np.array(energy), frames


def motion_onset(t, energy, t_motor, factor=MOTION_FACTOR):
    """First time the aperture's motion energy leaves its pre-move floor."""
    if len(energy) < 5:
        return None, None
    pre = energy[1:][t[1:] < (t_motor if t_motor else t[-1] * 0.2)]
    floor = (np.median(pre) if len(pre) else np.median(energy[1:])) + 1e-6
    above = np.where(energy > factor * floor)[0]
    above = above[above > 0]
    return (float(t[above[0]]) if len(above) else None), floor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--direction", choices=("open", "close"))
    args = ap.parse_args()

    clip = args.clip or (_clips(args.direction) or [None])[-1]
    if not clip or not os.path.exists(clip):
        print("no clip found"); return 2
    base = clip[:-5]
    wav = base + ".wav"
    print("clip:", os.path.basename(clip))

    t_motor = None
    if os.path.exists(wav):
        t_motor, at, arms, afloor = audio_onset(wav)
        print("motor start (audio): %s" % ("%.1f s" % t_motor if t_motor is not None else "not found"))
    else:
        at = arms = afloor = None
        print("no audio beside the clip; motor start unknown")

    try:
        from sentry import kasa_state
        region = (kasa_state.REGION_Y[0], kasa_state.REGION_Y[1],
                  kasa_state.REGION_X[0], kasa_state.REGION_X[1])
    except Exception:
        region = None
    t, energy, frames = motion_energy(clip, region)
    print("frames decoded: %d (%.1f s at %.0f fps)" % (len(frames), len(frames) / FPS, FPS))
    t_panel, mfloor = motion_onset(t, energy, t_motor)
    print("panel start (video): %s" % ("%.2f s" % t_panel if t_panel is not None else "not found"))
    if t_motor is not None and t_panel is not None:
        print("STALL (panel - motor): %+.2f s" % (t_panel - t_motor))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), dpi=130, sharex=True)
    if at is not None:
        a1.plot(at, arms, color="#2a78d6", lw=1.4)
        a1.axhline(MOTOR_FACTOR * afloor, color="#9a9990", lw=0.8, ls="--")
        a1.set_ylabel("audio RMS")
    a2.plot(t, energy, color="#eb6834", lw=1.4)
    if mfloor:
        a2.axhline(MOTION_FACTOR * mfloor, color="#9a9990", lw=0.8, ls="--")
    a2.set_ylabel("aperture motion"); a2.set_xlabel("seconds from capture start")
    for ax in (a1, a2):
        if t_motor is not None:
            ax.axvline(t_motor, color="#2a78d6", lw=1, alpha=0.7)
        if t_panel is not None:
            ax.axvline(t_panel, color="#eb6834", lw=1, alpha=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    title = os.path.basename(base)
    if t_motor is not None and t_panel is not None:
        title += "   motor %.1f s -> panel %.2f s   stall %+.2f s" % (t_motor, t_panel, t_panel - t_motor)
    a1.set_title(title, loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(base + "_motion.png"); plt.close(fig)

    # a strip of frames around the breakaway, aperture region only
    if frames and (t_motor is not None or t_panel is not None):
        t0 = (t_motor if t_motor is not None else t_panel) - 0.5
        idx = [int(round((t0 + k * 0.25) * FPS)) for k in range(11)]
        idx = [i for i in idx if 0 <= i < len(frames)]
        tiles = []
        for i in idx:
            fr = frames[i]
            if region:
                y0, y1, x0, x1 = region
                fr = fr[y0:y1, x0:x1]
            fr = cv2.resize(fr, (int(fr.shape[1] * 300 / fr.shape[0]), 300))
            cv2.putText(fr, "%.2fs" % (i / FPS), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            tiles.append(fr)
        if tiles:
            cv2.imwrite(base + "_breakaway.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", base + "_motion.png", "and", base + "_breakaway.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
