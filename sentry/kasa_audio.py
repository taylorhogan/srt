"""
kasa_audio.py
A second roof-move audio library, heard through the inside Kasa cam's
microphone instead of the webcam's.

The webcam and its microphone are being retired together -- they are one USB
device, which is why they have always failed together. The Kasa cam can see the
roof and the scope; this is the part that hears them.

WHY A SEPARATE LIBRARY RATHER THAN MORE ENTRIES IN THE OLD ONE

audio_classify judges a move by mel-spectrogram MSE against known-good moves.
That comparison is only meaningful between recordings made through the same
microphone, in the same place, at the same gain and the same bandwidth. None of
those hold across these two mics, and the last one is not a matter of degree:

    eMeet webcam mic   44100 Hz  ->  content to 22 kHz
    Kasa camera mic     8000 Hz  ->  content to  4 kHz   (G.711 is always 8 k)

More than two thirds of the band the existing library was built on does not
exist in a Kasa recording. Filing Kasa captures into roof_audio/ would not give
the classifier more references, it would give it references it cannot compare,
and the self-tuning threshold (the good library's own worst pairwise similarity)
would collapse toward zero and start calling everything good. So: parallel tree,
parallel counts, same code.

    roof_audio/        eMeet mic, 44.1 kHz  -- live, still the one in the roof path
    roof_audio_kasa/   Kasa mic,   8 kHz    -- this one, being built

The classifier itself is reused unchanged. classify() compares rendered PNGs, so
it never sees a sample rate; only the rendering and the capture are rate-bound,
and both live here. Spectrograms are rendered at the same figure size and DPI as
the originals so the two trees stay pixel-comparable in shape, even though their
contents must never be mixed.

THE 4 kHz CEILING IS A REAL COST, NOT A TECHNICALITY. The failure this system
exists to catch is gear chatter, and chatter is broadband -- some of what makes
it obvious on the eMeet mic is simply absent here. Whether what remains still
separates a bad move from a good one is an open question that this library is
being built to answer, not one it assumes.

    python -m sentry.kasa_audio record open --seconds 45
    python -m sentry.kasa_audio counts
    python -m sentry.kasa_audio classify <png> open
"""
import argparse
import contextlib
import io
import os
import sys
import threading
import wave
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sentry import audio_classify


def _log(msg):
    """Log without importing the app logger at module import time."""
    try:
        audio_classify._logger.info(msg)
    except Exception:                     # noqa: BLE001
        print(msg)

_HERE = os.path.dirname(os.path.abspath(__file__))
KASA_AUDIO_ROOT = os.path.join(_HERE, "roof_audio_kasa")

RATE = 8000          # G.711 is always 8 kHz mono
# Nyquist is 4000 Hz, so the 8000 the eMeet renderer asks for is not merely
# generous here, it is unreachable -- librosa would build mel bands over a band
# that carries no signal and the top of every Kasa spectrogram would be a
# constant. Sitting just under Nyquist keeps every band real.
FMAX = 3800
HOST = "192.168.87.65"

# Matched to audio_classify so both trees render to the same pixel dimensions.
FIG_SIZE = audio_classify.FIG_SIZE
DPI = audio_classify.DPI
CMAP = audio_classify.CMAP


def generate_spectrogram(audio_np, save_path):
    """Render an 8 kHz clip the way audio_classify renders a 44.1 kHz one.

    Deliberately NOT audio_classify.generate_spectrogram, which hardcodes
    sr=44100 and fmax=8000. Handed 8 kHz samples it would place every feature
    5.5x too high and ask for a band above Nyquist -- producing a plausible
    looking image that encodes the wrong frequencies, which is the worst
    possible failure for a library built to be compared against for years.
    """
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    S = librosa.feature.melspectrogram(y=audio_np, sr=RATE, n_mels=128, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis=None, y_axis=None, sr=RATE,
                             fmax=FMAX, cmap=CMAP)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def capture_av(seconds, host=HOST):
    """Pull `seconds` off the camera as (int16 PCM, part times, H.264 bytes).

    The camera serves ONE multipart stream with audio and video interleaved,
    so the video costs nothing extra -- it is already on the wire. It used to
    be discarded here; sentry/kasa_roof_frames.py now turns it into labelled
    stills of the roof. Grabbing those from a second connection would have
    risked colliding with this capture for no gain.
    """
    from configs import config
    from scripts import iriscam_record as ir
    from sentry.sky_camera import credentials

    user, pw = credentials(config.data())
    with contextlib.redirect_stdout(io.StringIO()):
        buf, checkpoints, t0 = ir.capture(host, user, pw, seconds)
        audio, video, times, _n = ir.split(buf, checkpoints, t0, b"--data-boundary--")
    return ir.ulaw_to_pcm16(audio), times, video


def capture_pcm(seconds, host=HOST):
    """Pull `seconds` of microphone audio off the camera as int16 PCM."""
    pcm, times, _video = capture_av(seconds, host)
    return pcm, times


def record(direction, seconds=45, host=HOST, status="unlabeled"):
    """Record a roof move and file it as WAV + spectrogram.

    Returns the same shape as audio_classify.finish_background_capture so the
    existing labelling and promotion helpers work on it unchanged.
    """
    pcm, times = capture_pcm(seconds, host)
    if pcm.size == 0:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(KASA_AUDIO_ROOT, status, direction)
    os.makedirs(dest, exist_ok=True)
    base = os.path.join(dest, "%s_%s" % (stamp, direction))
    wav_path, png_path = base + ".wav", base + ".png"

    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())

    # librosa wants float in [-1, 1]; int16 straight in would be 32768x hot and
    # power_to_db(ref=np.max) would hide it by normalising, so it would look fine.
    generate_spectrogram((pcm.astype(np.float32) / 32768.0), png_path)

    f = pcm.astype(np.float64)
    return {"direction": direction, "wav": wav_path, "spectrogram": png_path,
            "seconds": round(len(pcm) / RATE, 1),
            "rms": round(float(np.sqrt((f ** 2).mean())), 1),
            "peak": int(np.abs(pcm).max())}


# A roof move is NOT a sustained step. Measured on the 2026-08-18 open, on both
# microphones independently: a hard transient at the start, then a decay to the
# noise floor within about twelve seconds -- while the motor stays powered for
# another thirty-four. The travel is over long before the relay is released.
#
#   eMeet 44.1 kHz   t=0-11 s   13315 -> 997 rms, floor by t=12
#   Kasa   8   kHz   t=0-12 s    1306 ->  10 rms, floor by t=13
#
# That agreement matters more than either number: it rules out the obvious
# suspicion about a consumer IP camera, that its noise suppression was gating
# the steady part away. There is no steady part to gate.
#
# 8.0 was set from a guess and cut the clip at 9 s, dropping the decay tail --
# which is the part of the move most likely to sound wrong when a gear is not
# engaging. 3.0 keeps the whole 12 s and still sits far above the ~1.0x ambient
# wander measured either side of the move.
# Two thresholds, not one, and the reason is a mistake worth recording. A flat
# 3.0x with "take the longest run above it" picked up 190 s of low rumble later
# in the same recording -- the mount slewing, most likely -- and preferred it to
# the roof, because at a low threshold the longest run is no longer the loudest
# event. A flat 8.0x found the roof but truncated its decay at 9 s.
#
# Neither is fixable by moving one number, because duration and amplitude are
# selecting for different events. So: FIND the move by its peak (nothing else
# in these recordings comes near 200x), then EXPAND outward while the signal
# stays above a much lower bound. Standard hysteresis, and it gets both ends
# right -- the roof is the loudest thing that happens, and its tail is the part
# that matters for hearing a gear fail to engage.
MOVE_FACTOR = 20.0    # peak must reach this x floor to count as a move at all
EXTEND_FACTOR = 3.0   # expand around that peak while above this x floor
# A roof move sounds for ~12 s. A door slam is under a second and can easily be
# louder, so peak alone picks the slam and files a 1 s clip as the reference for
# what a roof sounds like. Candidates are tried loudest-first and the first one
# that lasts plausibly long wins.
MIN_MOVE_S = 3.0
BIN_S = 0.5
PAD_S = 4.0           # kept either side, so the start-up clunk is not clipped
# Every clip is emitted at EXACTLY this length, move starting at PAD_S.
# Not cosmetic: the spectrogram renderer stretches any duration to the same
# 1000x600 image, so two normal moves clipped at 17 s and 20 s land at
# different positions and widths, and the image-MSE classifier reads the
# misalignment as dissimilarity. Measured on the first two good opens:
# similarity 0.147 between two moves the eMeet system rated good -- a
# threshold so weak (0.13) that almost anything would classify as good.
# Fixed length + fixed move position makes the images comparable at all.
CLIP_S = 24.0


def extract_move(pcm, bin_s=BIN_S, factor=MOVE_FACTOR,
                 pad_s=PAD_S, extend_factor=EXTEND_FACTOR):
    """Find the roof move inside a long capture and return (clip, detail).

    Deliberately not "record for exactly the length of the move". Each capture
    opens a fresh HTTPS stream, so back-to-back short captures leave a
    reconnect gap of a second or two between them -- and this camera already
    drops frames often enough to have eaten the rain detector's warning time.
    A gap landing inside a 20 s move would silently corrupt a library entry
    that then gets compared against for years. One uninterrupted window with
    the move found afterwards has no seam to land in.
    """
    n = int(bin_s * RATE)
    if pcm.size < n * 4:
        return None, {"why": "capture too short"}
    bins = pcm[:pcm.size // n * n].reshape(-1, n).astype(np.float64)
    rms = np.sqrt((bins ** 2).mean(axis=1))
    floor = float(np.median(rms)) or 1.0
    lim = floor * extend_factor
    order = np.argsort(rms)[::-1]
    if rms[order[0]] < floor * factor:
        return None, {"why": "loudest bin only %.1fx the floor, need %.1fx"
                             % (rms[order[0]] / floor, factor),
                      "floor_rms": round(floor, 1),
                      "peak_bin_rms": round(float(rms[order[0]]), 1)}

    # Loudest first, but a candidate has to last like a roof move. Walking out
    # from the peak is what stops a long quiet rumble elsewhere winning; the
    # duration test is what stops a short loud slam winning.
    best = None
    for peak in order:
        if rms[peak] < floor * factor:
            break
        lo = hi = int(peak)
        while lo > 0 and rms[lo - 1] > lim:
            lo -= 1
        while hi < len(rms) - 1 and rms[hi + 1] > lim:
            hi += 1
        if best is None:
            best = (lo, hi, int(peak))          # fall back to the loudest
        if (hi - lo + 1) * bin_s >= MIN_MOVE_S:
            best = (lo, hi, int(peak))
            break
    lo, hi, peak = best
    if (hi - lo + 1) * bin_s < MIN_MOVE_S:
        return None, {"why": "loudest event lasts only %.1fs, need %.1fs"
                             % ((hi - lo + 1) * bin_s, MIN_MOVE_S),
                      "floor_rms": round(floor, 1),
                      "peak_bin_rms": round(float(rms[peak]), 1)}

    # Fixed-length window: PAD_S of lead-in, then whatever of the move and
    # its aftermath fits in CLIP_S. Clamp to the recording, keeping length.
    want = int(CLIP_S * RATE)
    i0 = max(0, int((lo * bin_s - pad_s) * RATE))
    if i0 + want > pcm.size:
        i0 = max(0, pcm.size - want)
    i1 = i0 + want
    detail = {"floor_rms": round(floor, 1),
              "move_rms": round(float(rms[lo:hi + 1].mean()), 1),
              "ratio": round(float(rms[lo:hi + 1].mean() / floor), 1),
              "peak_ratio": round(float(rms[peak] / floor), 1),
              "move_s": round((hi - lo + 1) * bin_s, 1),
              "at_s": round(lo * bin_s, 1),
              # A move touching either end was probably cut off by the window.
              "clipped": bool(lo == 0 or hi >= len(rms) - 1)}
    return pcm[i0:i1], detail


def listen(direction, minutes=6.0, host=HOST, status="unlabeled"):
    """Watch one uninterrupted window, keep the roof move found inside it."""
    pcm, _times = capture_pcm(int(minutes * 60), host)
    if pcm.size == 0:
        return None
    clip, detail = extract_move(pcm)
    if clip is None:
        return {"found": False, **detail}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(KASA_AUDIO_ROOT, status, direction)
    os.makedirs(dest, exist_ok=True)
    base = os.path.join(dest, "%s_%s" % (stamp, direction))
    with wave.open(base + ".wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(clip.tobytes())
    generate_spectrogram(clip.astype(np.float32) / 32768.0, base + ".png")
    return {"found": True, "direction": direction, "wav": base + ".wav",
            "spectrogram": base + ".png",
            "seconds": round(len(clip) / RATE, 1), **detail}


# Long enough to hold the ~12 s of audible travel plus the lead-in before the
# relay fires and a margin for a slow move. Nothing waits on this, so being
# generous costs only a background thread.
CAPTURE_S = 50


def start_capture_async(direction=None, seconds=CAPTURE_S, host=HOST,
                        on_done=None):
    """Record the move on a daemon thread and file it. Returns immediately.

    Fire-and-forget ON PURPOSE. The eMeet path can stop its sounddevice stream
    the instant the motor cuts, so its finish step is free; this one cannot --
    the camera hands over a fixed-length HTTP stream and there is no way to end
    it early. A matching finish/join would therefore park the roof flow for
    however much of the window remained, right at the point where the post-move
    vision check runs. Nothing in the roof path may wait on a shadow library.

    So the thread files its own result and the caller gets a handle it can
    ignore. The cost is that the clip is not available to attach to the chat
    message the eMeet capture posts. That is acceptable while this library is
    being built and judged by nobody.

    Never raises. A capture that fails must be indistinguishable, from the roof
    flow's point of view, from one that was never started.
    """
    handle = {"direction": direction, "started": False, "thread": None}

    def _run():
        try:
            pcm, _t, video = capture_av(seconds, host)
            # Labelled stills of the roof, decoded from the video half of the
            # same stream. Filed FIRST and in its own try: it is independent of
            # whether the audio turns out to contain a usable move, and one
            # observer must not be able to cost the other its data.
            try:
                from sentry import kasa_roof_frames
                kasa_roof_frames.save_move_frames(video, direction)
            except Exception as e:        # noqa: BLE001 - shadow observer
                _log("kasa roof frames: failed (ignored): %r" % (e,))
            if pcm.size == 0:
                _log("kasa roof audio: no audio arrived")
                return
            clip, detail = extract_move(pcm)
            if clip is None:
                _log("kasa roof audio: no move found (%s)" % detail.get("why"))
                return
            stamp = datetime.now().strftime("%Y%m%dT%H-%M-%S")
            dest = os.path.join(KASA_AUDIO_ROOT, "unlabeled", direction or "unknown")
            os.makedirs(dest, exist_ok=True)
            base = os.path.join(dest, "%s_%s" % (stamp, direction or "unknown"))
            with wave.open(base + ".wav", "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
                w.writeframes(clip.tobytes())
            generate_spectrogram(clip.astype(np.float32) / 32768.0, base + ".png")
            verdict = classify(base + ".png", direction)
            _log("kasa roof audio %s: %.1fs move, %.0fx floor, verdict %s (%s)"
                 % (direction, detail["move_s"], detail["peak_ratio"],
                    verdict["verdict"], verdict["note"] or verdict["best_match"]))
            # Self-extend only once the library can actually judge, exactly as
            # the eMeet path does. Until then everything waits in unlabeled/.
            if verdict["verdict"] == "good":
                promote_to_good({"direction": direction, "wav": base + ".wav",
                                 "spectrogram": base + ".png"})
            if on_done:
                on_done(base, detail, verdict)
        except Exception as e:            # noqa: BLE001 - shadow observer
            _log("kasa roof audio capture failed: %r" % (e,))

    try:
        th = threading.Thread(target=_run, name="kasa-roof-audio", daemon=True)
        th.start()
        handle["started"] = True
        handle["thread"] = th
    except Exception as e:                # noqa: BLE001
        _log("kasa roof audio thread failed to start: %r" % (e,))
    return handle


# The library helpers, bound to this tree. Same code, same self-tuning
# threshold, same MIN_GOOD_REFS -- only the root differs.
def classify(png_path, direction):
    return audio_classify.classify(png_path, direction, root=KASA_AUDIO_ROOT)


def label(direction, verdict, name=None):
    return audio_classify.label(direction, verdict, name=name, root=KASA_AUDIO_ROOT)


def promote_to_good(result):
    return audio_classify.promote_to_good(result, root=KASA_AUDIO_ROOT)


def list_unlabeled(direction=None):
    return audio_classify.list_unlabeled(direction, root=KASA_AUDIO_ROOT)


def counts():
    return audio_classify.library_counts(root=KASA_AUDIO_ROOT)


def readiness():
    """How far off is this library from being able to judge a move?"""
    have = counts().get("good", {})
    need = audio_classify.MIN_GOOD_REFS
    return {d: {"have": have.get(d, 0), "need": need,
                "ready": have.get(d, 0) >= need}
            for d in ("open", "close")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record a roof move into unlabeled/")
    r.add_argument("direction", choices=("open", "close"))
    r.add_argument("--seconds", type=int, default=45)
    r.add_argument("--good", action="store_true",
                   help="file straight into good/ (only when the move was normal)")

    c = sub.add_parser("classify", help="judge a spectrogram against this library")
    c.add_argument("png")
    c.add_argument("direction", choices=("open", "close"))

    li = sub.add_parser("listen", help="watch a window and keep the move inside it")
    li.add_argument("direction", choices=("open", "close"))
    li.add_argument("--minutes", type=float, default=6.0)
    li.add_argument("--good", action="store_true")

    sub.add_parser("counts", help="library contents and readiness")

    la = sub.add_parser("label", help="file the newest unlabeled capture")
    la.add_argument("direction", choices=("open", "close"))
    la.add_argument("verdict", choices=("good", "bad"))

    args = ap.parse_args()

    if args.cmd == "record":
        print("recording %ds from the Kasa mic..." % args.seconds)
        res = record(args.direction, args.seconds,
                     status="good" if args.good else "unlabeled")
        if not res:
            print("no audio arrived from the camera")
            return 1
        print("  %.1fs  rms %.1f  peak %d" % (res["seconds"], res["rms"], res["peak"]))
        print("  %s" % res["spectrogram"])
        print("  %s" % res["wav"])
        verdict = classify(res["spectrogram"], args.direction)
        print("  classify: %s (%s)" % (verdict["verdict"], verdict["note"] or
                                       "best %.4f vs threshold %.4f"
                                       % (verdict["best_score"] or 0,
                                          verdict["threshold"] or 0)))
    elif args.cmd == "listen":
        print("listening %.1f min for the %s..." % (args.minutes, args.direction))
        res = listen(args.direction, args.minutes,
                     status="good" if args.good else "unlabeled")
        if not res:
            print("no audio arrived from the camera")
            return 1
        if not res.get("found"):
            print("  no move detected: %s (floor rms %s, loudest bin %s)"
                  % (res.get("why"), res.get("floor_rms"), res.get("peak_bin_rms")))
            return 2
        print("  move at t=%ss, %ss long, %sx the floor%s"
              % (res["at_s"], res["move_s"], res["ratio"],
                 "  *** CLIPPED by the window ***" if res["clipped"] else ""))
        print("  kept %.1fs -> %s" % (res["seconds"], res["spectrogram"]))
        v = classify(res["spectrogram"], args.direction)
        print("  classify: %s (%s)" % (v["verdict"], v["note"] or "best %.4f"
                                       % (v["best_score"] or 0)))
    elif args.cmd == "classify":
        print(classify(args.png, args.direction))
    elif args.cmd == "counts":
        print("library:", counts() or "(empty)")
        for d, s in readiness().items():
            print("  %-6s %d/%d  %s" % (d, s["have"], s["need"],
                                        "ready" if s["ready"] else "NOT ready"))
    elif args.cmd == "label":
        print(label(args.direction, args.verdict) or "nothing unlabeled to file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
