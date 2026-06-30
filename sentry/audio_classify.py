import argparse
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
LIBRARY_DIR = os.path.join(_HERE, "library_spectrograms")    # pre-saved reference spectrogram PNGs
# Roof-move spectrograms, organized by direction (parallels roof_signatures/):
#   roof_audio/{status}/{open,close}/<timestamp>_<direction>.{png,wav}
ROOF_AUDIO_ROOT = os.path.join(_HERE, "roof_audio")
FIG_SIZE = (10, 6)              # Fixed figure size for consistent image dimensions
DPI = 100                       # Fixed DPI → consistent pixel size (1000×600 here)
CMAP = 'magma'                  # Consistent colormap (common for spectrograms)

# Ensure the reference library directory exists
os.makedirs(LIBRARY_DIR, exist_ok=True)
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


def compare_to_library(new_img_path):
    """Compare a spectrogram image to the reference library using MSE similarity.

    Returns (best_match, best_score, sorted_results). The library is currently
    empty until the reference spectrograms are copied over; the future plan is to
    classify each roof-move spectrogram against it.
    """
    new_img = Image.open(new_img_path).convert('RGB')
    new_array = img_as_float(np.array(new_img))

    best_score = -1
    best_match = None
    results = []

    for lib_path in glob.glob(os.path.join(LIBRARY_DIR, "*.png")):
        lib_img = Image.open(lib_path).convert('RGB')
        lib_array = img_as_float(np.array(lib_img))

        # Ensure same size (should be identical if generated with same params)
        if new_array.shape != lib_array.shape:
            lib_img = lib_img.resize(new_img.size, Image.LANCZOS)
            lib_array = img_as_float(np.array(lib_img))

        # Calculate MSE and convert to similarity score
        mse = np.mean((new_array - lib_array) ** 2)
        score = 1 / (1 + mse * 100)

        results.append((os.path.basename(lib_path), score))
        if score > best_score:
            best_score = score
            best_match = os.path.basename(lib_path)

    # Sort results for full ranking
    results.sort(key=lambda x: x[1], reverse=True)
    return best_match, best_score, results


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
                    help="label for a --capture clip")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="length of a --capture clip (default: 5)")
    args = ap.parse_args()

    if args.list_devices:
        list_input_devices()
    elif args.capture:
        capture_test(args.direction, args.seconds, args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
