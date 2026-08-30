import argparse
import shutil
import threading
import time
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
import os
import sys
import glob
from datetime import datetime
from PIL import Image
import numpy as np
from skimage import img_as_float
from scipy.io.wavfile import write

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils import utils

# ----------------------------- CONFIGURATION -----------------------------
# Audio I/O uses sounddevice (not pyaudio). On the Windows observatory PC the
# normal Windows audio host APIs (MME/DirectSound/WASAPI) present **no** devices
# to a disconnected/headless remote session — only kernel-streaming WDM-KS sees
# the mics. pyaudio's bundled PortAudio fails WDM-KS capture with a -9999 host
# error; sounddevice's PortAudio handles it. WDM-KS also requires int16 (float32
# is what produced the -9999) and only supports PortAudio's CALLBACK interface
# ("Blocking API not supported yet"), so capture is callback-driven. WDM-KS
# bypasses the per-session audio engine, so it keeps working when no one is
# logged in — exactly what unattended operation needs. See memory:
# project_audio_detector_windows_migration.
#
# This module is command-triggered: the roof move starts a recording that runs
# for the move's duration (parallel to the current-signature capture), so there
# is deliberately NO ambient-noise trigger / RMS threshold here.
CHANNELS = 1
RATE = 44100                    # Common sample rate
CHUNK = 1024                    # Audio chunk size
DEVICE_NAME = "eMeet"           # Substring of the input device to use (eMeet C960 webcam mic)

# Paths anchored to this script's directory so capture works regardless of the
# current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
# Roof-move spectrograms, organized by status + direction (parallels roof_signatures/):
#   roof_audio/{unlabeled,good,bad}/{open,close}/<timestamp>_<direction>.{png,wav}
# good/ is the reference library that classify() judges new moves against. It is
# self-extending: a move classify() judges "good" is auto-filed here by the roof
# flow (capped at MAX_GOOD_REFS, rolling). Captures that come back "bad"/"unknown"
# stay in unlabeled/ for a human to file with `audio <open|close> <good|bad>`.
ROOF_AUDIO_ROOT = os.path.join(_HERE, "roof_audio")
FIG_SIZE = (10, 6)              # Fixed figure size for consistent image dimensions
DPI = 100                       # Fixed DPI → consistent pixel size (1000×600 here)
CMAP = 'magma'                  # Consistent colormap (common for spectrograms)

# classify(): a new move counts as "good" if its best similarity to a known-good
# spectrogram reaches GOOD_MARGIN × the good library's own worst pairwise
# similarity — i.e. it must look at least about as normal as good moves look to
# each other. Self-tuning, so no absolute threshold to hand-tune as the library
# grows. MIN_GOOD_REFS mirrors roof_current_signature.compare()'s ≥2 rule.
GOOD_MARGIN = 0.9
MIN_GOOD_REFS = 2
# Cap the auto-filed good library. classify() does an O(n^2) pairwise image-MSE
# over the whole good library on every roof move, so an unbounded library would
# slowly make each move more expensive. Once we hold this many good references
# per direction the library is "mature"; the newest capture retires the oldest
# (rolling window), keeping cost bounded and the envelope current.
MAX_GOOD_REFS = 40
# -------------------------------------------------------------------------

_logger = utils.set_logger()


def find_input_device(name_substr):
    """Return the device index of the first input-capable device whose name
    contains `name_substr`. WDM-KS entries are preferred because that is the
    only host API that enumerates mics in a headless session on this machine.
    Returns None if no match (caller falls back to the default input device)."""
    matches = []
    for idx, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_substr.lower() in d["name"].lower():
            api = sd.query_hostapis()[d["hostapi"]]["name"]
            matches.append((idx, d["name"], api))
    if not matches:
        return None
    # Prefer WDM-KS so headless capture works; otherwise take the first match.
    for idx, name, api in matches:
        if "WDM-KS" in api:
            return idx
    return matches[0][0]


def list_input_devices():
    print("Input-capable devices:")
    for idx, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis()[d["hostapi"]]["name"]
            print(f"  idx {idx:>2} [{api:18}] {d['name']}  "
                  f"({d['max_input_channels']}ch @ {int(d['default_samplerate'])}Hz)")


def generate_spectrogram(audio_np, save_path):
    """Generate and save a spectrogram image from numpy audio array."""
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    S = librosa.feature.melspectrogram(y=audio_np, sr=RATE, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis=None, y_axis=None, sr=RATE, fmax=8000, cmap=CMAP)
    plt.axis('off')  # Hide axes for cleaner comparison
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def _img_array(path, size=None):
    """Load a spectrogram PNG as a float RGB array, resized to `size` if given.

    All spectrograms are rendered at the same figure size/DPI so sizes should
    already match; the resize is a guard against library entries from before a
    rendering change.
    """
    img = Image.open(path).convert('RGB')
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img_as_float(np.array(img))


def _similarity(a, b):
    """MSE-based similarity between two same-shaped image arrays, in (0, 1]."""
    mse = np.mean((a - b) ** 2)
    return float(1 / (1 + mse * 100))


def classify(png_path, direction, root=None):
    """Judge a roof-move spectrogram against the known-good library for `direction`.

    Returns {"verdict": "good"|"bad"|"unknown", "best_match", "best_score",
    "threshold", "n_refs", "note"}. The threshold is self-tuning: the good
    library's worst pairwise similarity × GOOD_MARGIN (see constants above).
    Never raises — this runs in the roof safety path, so any failure comes back
    as an "unknown" verdict instead.
    """
    root = root or ROOF_AUDIO_ROOT
    result = {"verdict": "unknown", "best_match": None, "best_score": None,
              "threshold": None, "n_refs": 0, "note": ""}
    try:
        lib_dir = os.path.join(root, "good", direction or "unknown")
        ref_paths = sorted(glob.glob(os.path.join(lib_dir, "*.png")))
        result["n_refs"] = len(ref_paths)
        if len(ref_paths) < MIN_GOOD_REFS:
            result["note"] = (f"only {len(ref_paths)} known-good {direction or 'unknown'} "
                              f"spectrogram(s) — need >= {MIN_GOOD_REFS} to judge")
            return result

        new_img = Image.open(png_path).convert('RGB')
        size = new_img.size
        new_arr = img_as_float(np.array(new_img))
        refs = [(os.path.basename(p), _img_array(p, size)) for p in ref_paths]

        for name, arr in refs:
            score = _similarity(new_arr, arr)
            if result["best_score"] is None or score > result["best_score"]:
                result["best_score"] = score
                result["best_match"] = name

        pairwise = [_similarity(refs[i][1], refs[j][1])
                    for i in range(len(refs)) for j in range(i + 1, len(refs))]
        result["threshold"] = min(pairwise) * GOOD_MARGIN
        result["verdict"] = "good" if result["best_score"] >= result["threshold"] else "bad"

        # Golden distance. good/ ROLLS (newest 40), so its threshold follows
        # the roof: a slowly degrading mechanism stays "good" against last
        # month forever. golden/ is frozen at a known-healthy era (seeded by
        # scripts/roof_golden_freeze.py, re-anchored only after mechanical
        # service) and answers the other question: how far from HEALTHY.
        gpaths = sorted(glob.glob(os.path.join(root, "golden",
                                               direction or "unknown", "*.png")))
        if len(gpaths) >= MIN_GOOD_REFS:
            grefs = [_img_array(p, size) for p in gpaths]
            result["golden_score"] = max(_similarity(new_arr, a) for a in grefs)
            gpair = [_similarity(grefs[i], grefs[j])
                     for i in range(len(grefs)) for j in range(i + 1, len(grefs))]
            result["golden_threshold"] = min(gpair) * GOOD_MARGIN
            result["golden_ok"] = result["golden_score"] >= result["golden_threshold"]
            _record_drift({"kind": "audio", "direction": direction,
                           "rolling_ok": result["verdict"] == "good",
                           "golden_ok": result["golden_ok"],
                           "summary": "similarity %.3f vs golden floor %.3f"
                                      % (result["golden_score"],
                                         result["golden_threshold"])})
    except Exception as e:  # noqa: BLE001 — classifier must not crash the roof flow
        _logger.warning("Audio classify failed for %s: %s", png_path, e)
        result["note"] = f"classification failed: {e}"
    return result


def _record_drift(entry):
    """Append one drift record to local/roof_drift.jsonl. Never raises."""
    try:
        import json
        from datetime import datetime as _dt
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "local", "roof_drift.jsonl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        entry.setdefault("t", _dt.now().astimezone().isoformat(timespec="seconds"))
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001 — drift logging must not touch the roof flow
        _logger.exception("roof drift record failed (ignored)")


# --------------------------------------------------------------------------- #
# Labeling — file unlabeled captures into the good/bad library
# --------------------------------------------------------------------------- #
def list_unlabeled(direction=None, root=None):
    """Return unlabeled captures as [{"base", "direction", "png", "wav"}], newest first.

    Basenames start with an ISO timestamp, so lexicographic order is time order.
    """
    root = root or ROOF_AUDIO_ROOT
    entries = []
    base_dir = os.path.join(root, "unlabeled")
    dirs = [direction] if direction else \
        (sorted(os.listdir(base_dir)) if os.path.isdir(base_dir) else [])
    for d in dirs:
        for png in glob.glob(os.path.join(base_dir, d, "*.png")):
            base = os.path.splitext(os.path.basename(png))[0]
            wav = os.path.splitext(png)[0] + ".wav"
            entries.append({"base": base, "direction": d, "png": png,
                            "wav": wav if os.path.exists(wav) else None})
    entries.sort(key=lambda e: e["base"], reverse=True)
    return entries


def label(direction, verdict, name=None, root=None):
    """Move an unlabeled capture (PNG + WAV) into the good/bad library.

    Targets the most recent unlabeled capture for `direction`, or the one whose
    basename contains `name`. Returns {"base", "direction", "moved": [paths]}
    or None if there is nothing to label / no match.
    """
    if verdict not in ("good", "bad"):
        raise ValueError(f"verdict must be 'good' or 'bad', not {verdict!r}")
    root = root or ROOF_AUDIO_ROOT
    candidates = list_unlabeled(direction, root=root)
    if name:
        candidates = [c for c in candidates if name in c["base"]]
    if not candidates:
        return None
    target = candidates[0]
    dest_dir = os.path.join(root, verdict, direction)
    os.makedirs(dest_dir, exist_ok=True)
    moved = []
    for path in (target["png"], target["wav"]):
        if path:
            dest = os.path.join(dest_dir, os.path.basename(path))
            shutil.move(path, dest)
            moved.append(dest)
    _logger.info("Labeled roof audio %s as %s: %s", target["base"], verdict, moved)
    return {"base": target["base"], "direction": direction, "moved": moved}


def _prune_good_library(direction, root=None):
    """Keep only the newest MAX_GOOD_REFS good captures for `direction`.

    Filenames are timestamp-prefixed, so lexicographic order is chronological;
    anything past the cap (oldest first) has its PNG + WAV removed. Best-effort —
    a failed delete is logged, not raised. Returns the retired basenames.
    """
    root = root or ROOF_AUDIO_ROOT
    good_dir = os.path.join(root, "good", direction or "unknown")
    pngs = sorted(glob.glob(os.path.join(good_dir, "*.png")))
    retired = []
    surplus = pngs[:-MAX_GOOD_REFS] if len(pngs) > MAX_GOOD_REFS else []
    for png in surplus:
        for path in (png, os.path.splitext(png)[0] + ".wav"):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                _logger.warning("Could not retire old good ref %s: %s", path, e)
        retired.append(os.path.basename(png))
    if retired:
        _logger.info("Retired %d oldest good %s ref(s): %s",
                     len(retired), direction, retired)
    return retired


def promote_to_good(result, root=None):
    """Auto-file a just-captured roof-move audio clip into the good library.

    `result` is the dict returned by finish_background_capture (spectrogram + wav
    + direction), which lives in roof_audio/unlabeled/<direction>/. Moves the
    PNG + WAV into roof_audio/good/<direction>/ and prunes to MAX_GOOD_REFS.
    Returns {"moved": [new_paths], "retired": [basenames]} or None. Never raises
    — this runs in the roof safety path, so a filing failure must not break it.
    """
    root = root or ROOF_AUDIO_ROOT
    try:
        direction = result.get("direction") or "unknown"
        dest_dir = os.path.join(root, "good", direction)
        os.makedirs(dest_dir, exist_ok=True)
        moved = []
        for path in (result.get("spectrogram"), result.get("wav")):
            if path and os.path.exists(path):
                dest = os.path.join(dest_dir, os.path.basename(path))
                shutil.move(path, dest)
                moved.append(dest)
        if not moved:
            return None
        _logger.info("Auto-filed roof %s audio as good: %s",
                     direction, [os.path.basename(m) for m in moved])
        retired = _prune_good_library(direction, root=root)
        return {"moved": moved, "retired": retired}
    except Exception as e:  # noqa: BLE001 — must not crash the roof flow
        _logger.warning("Auto-file to good failed: %s", e)
        return None


def library_counts(root=None):
    """Return {status: {direction: n_spectrograms}} for the roof audio library."""
    root = root or ROOF_AUDIO_ROOT
    counts = {}
    if not os.path.isdir(root):
        return counts
    for status in sorted(os.listdir(root)):
        spath = os.path.join(root, status)
        if not os.path.isdir(spath):
            continue
        for d in sorted(os.listdir(spath)):
            n = len(glob.glob(os.path.join(spath, d, "*.png")))
            if n:
                counts.setdefault(status, {})[d] = n
    return counts


# --------------------------------------------------------------------------- #
# Background capture — for hooking into toggle_roof without blocking it
# (parallels sentry/roof_current_signature.py's start/finish helpers)
# --------------------------------------------------------------------------- #
def start_background_capture(direction=None, device_name=DEVICE_NAME):
    """Begin recording roof-move audio in a callback stream.

    Recording runs until finish_background_capture stops it, so the clip spans
    the whole move (the caller controls the duration — there is no noise
    trigger). Returns a handle for finish_background_capture. Best-effort: never
    raises into the caller (the roof safety path must not break). `direction` is
    metadata only — it labels where the spectrogram is filed.
    """
    handle = {"frames": [], "stream": None, "direction": direction,
              "lock": threading.Lock()}
    try:
        device_index = find_input_device(device_name)
        frames = handle["frames"]
        lock = handle["lock"]

        def _callback(indata, n_frames, time_info, status):
            if status:
                _logger.warning("roof audio input status: %s", status)
            # callback runs on the PortAudio thread; guard the shared list
            with lock:
                frames.append(indata[:, 0].copy())

        stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="int16",
                                blocksize=CHUNK, device=device_index, callback=_callback)
        stream.start()
        handle["stream"] = stream
    except Exception as e:  # noqa: BLE001 — observer must not crash the roof flow
        _logger.warning("Roof audio capture failed to start: %s", e)
    return handle


def finish_background_capture(handle, status="unlabeled", save=True):
    """Stop an audio capture and, if saving, write a spectrogram PNG + WAV.

    Files are organized by direction under roof_audio/{status}/{direction}/.
    Returns {"spectrogram", "wav", "direction"} or None. Never raises.
    """
    if not handle or handle.get("stream") is None:
        return None
    try:
        stream = handle["stream"]
        stream.stop()
        stream.close()
        if not save:
            return None
        with handle["lock"]:
            frames = list(handle["frames"])
        if not frames:
            _logger.warning("Roof audio capture produced no frames")
            return None

        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0
        direction = handle.get("direction") or "unknown"
        out_dir = os.path.join(ROOF_AUDIO_ROOT, status, direction)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
        base = f"{ts}_{direction}"
        png_path = os.path.join(out_dir, base + ".png")
        wav_path = os.path.join(out_dir, base + ".wav")

        generate_spectrogram(audio_np, png_path)
        write(wav_path, RATE, np.int16(audio_np * 32767))  # float[-1,1] -> int16
        _logger.info("Saved roof audio spectrogram: %s", png_path)
        return {"spectrogram": png_path, "wav": wav_path, "direction": direction}
    except Exception as e:  # noqa: BLE001
        _logger.warning("Finishing roof audio capture failed: %s", e)
        return None


def record_wav(seconds, out_path, device_name=DEVICE_NAME, spectrogram_path=None):
    """Record `seconds` of mono audio and write it as a WAV to `out_path`.

    A lighter-weight sibling of finish_background_capture: no library filing —
    just the raw clip, for attaching to a status report. If `spectrogram_path`
    is given, a mel spectrogram PNG of the same clip is written there too (same
    style as the roof-move spectrograms). Returns `out_path` on success or None
    on failure. Never raises (a status command must still succeed if the mic is
    unavailable); a failed spectrogram does not fail the recording.
    """
    handle = start_background_capture(device_name=device_name)
    if handle.get("stream") is None:
        return None
    try:
        time.sleep(seconds)
        stream = handle["stream"]
        stream.stop()
        stream.close()
        with handle["lock"]:
            frames = list(handle["frames"])
        if not frames:
            _logger.warning("record_wav produced no frames")
            return None
        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        write(out_path, RATE, np.int16(audio_np * 32767))  # float[-1,1] -> int16
        _logger.info("Saved status audio clip: %s", out_path)
        if spectrogram_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(spectrogram_path)), exist_ok=True)
                generate_spectrogram(audio_np, spectrogram_path)
            except Exception as e:  # noqa: BLE001 — spectrogram is best-effort
                _logger.warning("status spectrogram failed: %s", e)
        return out_path
    except Exception as e:  # noqa: BLE001
        _logger.warning("record_wav failed: %s", e)
        return None


def capture_test(direction, seconds, device_name):
    """Record a fixed-length test clip and save its spectrogram (manual check)."""
    print(f"Recording {seconds:.0f}s from a {device_name!r} mic "
          f"(direction={direction or 'unknown'})...")
    handle = start_background_capture(direction=direction, device_name=device_name)
    time.sleep(seconds)
    res = finish_background_capture(handle, status="unlabeled")
    if res:
        print(f"Saved spectrogram: {res['spectrogram']}")
        print(f"Saved wav:         {res['wav']}")
    else:
        print("Capture failed (no audio captured).")


def main():
    ap = argparse.ArgumentParser(description="Roof-move audio capture utility")
    ap.add_argument("--device", default=DEVICE_NAME,
                    help=f"input device name substring (default: {DEVICE_NAME!r})")
    ap.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    ap.add_argument("--capture", action="store_true",
                    help="record a fixed-length test clip and save its spectrogram")
    ap.add_argument("--direction", choices=["open", "close"], default=None,
                    help="direction for --capture / --classify / --label")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="length of a --capture clip (default: 5)")
    ap.add_argument("--classify", metavar="PNG",
                    help="classify a spectrogram PNG against the known-good library "
                         "(requires --direction)")
    ap.add_argument("--label", choices=["good", "bad"],
                    help="file the latest unlabeled capture for --direction as good/bad")
    ap.add_argument("--name", default=None,
                    help="with --label: target the capture whose name contains this")
    ap.add_argument("--list-unlabeled", action="store_true",
                    help="list unlabeled captures and library counts")
    args = ap.parse_args()

    if args.list_devices:
        list_input_devices()
    elif args.capture:
        capture_test(args.direction, args.seconds, args.device)
    elif args.classify:
        if not args.direction:
            ap.error("--classify requires --direction")
        res = classify(args.classify, args.direction)
        print(f"verdict:    {res['verdict']}")
        print(f"best match: {res['best_match']}  (score {res['best_score']})")
        print(f"threshold:  {res['threshold']}  ({res['n_refs']} good refs)")
        if res["note"]:
            print(f"note:       {res['note']}")
    elif args.label:
        if not args.direction:
            ap.error("--label requires --direction")
        res = label(args.direction, args.label, name=args.name)
        if res:
            print(f"Labeled {res['base']} as {args.label}:")
            for p in res["moved"]:
                print(f"  {p}")
        else:
            print(f"No unlabeled {args.direction} capture"
                  + (f" matching {args.name!r}" if args.name else "") + " found.")
    elif args.list_unlabeled:
        entries = list_unlabeled()
        print(f"{len(entries)} unlabeled capture(s):")
        for e in entries:
            wav = "+wav" if e["wav"] else "no wav"
            print(f"  {e['base']}  [{e['direction']}, {wav}]")
        print("Library counts:")
        for status, dirs in library_counts().items():
            for d, n in dirs.items():
                print(f"  {status}/{d}: {n}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
