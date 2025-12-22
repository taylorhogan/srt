import pyaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import numpy as np  # Already imported, but for clarity
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float

# ----------------------------- CONFIGURATION -----------------------------
FORMAT = pyaudio.paFloat32      # Easier for RMS calculation
CHANNELS = 1
RATE = 44100                    # Common sample rate
CHUNK = 1024                    # Audio chunk size
THRESHOLD = 0.005               # RMS threshold – tune this based on your mic/environment (0.01–0.05 typical)
RECORD_SECONDS = 10             # Length of audio clip to capture when sound is detected
LIBRARY_DIR = "library_spectrograms/"   # Folder with your pre-saved spectrogram PNGs
DETECTED_DIR = "detected_spectrograms/" # Where detected spectrograms will be saved (optional, for review)
FIG_SIZE = (10, 6)              # Fixed figure size for consistent image dimensions
DPI = 100                       # Fixed DPI → results in consistent pixel size (1000×600 here)
CMAP = 'magma'                  # Consistent colormap (common for spectrograms)

# Ensure directories exist
os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(DETECTED_DIR, exist_ok=True)
# -------------------------------------------------------------------------

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
    """Compare the new spectrogram image to all in the library using SSIM (Structural Similarity)."""
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

        score = ssim(new_array, lib_array, multichannel=True, channel_axis=-1,data_range=1.0)
        results.append((os.path.basename(lib_path), score))
        if score > best_score:
            best_score = score
            best_match = os.path.basename(lib_path)

    # Sort results for full ranking
    results.sort(key=lambda x: x[1], reverse=True)
    return best_match, best_score, results

# ----------------------------- MAIN LOOP -----------------------------
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening... (Ctrl+C to stop)")

buffer = []
triggered = False
frames_since_trigger = 0

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.float32)
        rms = np.sqrt(np.mean(audio_chunk**2))

        if rms > THRESHOLD:
            print (f"Sound detected (RMS: {rms:.4f})")
            if not triggered:
                print(f"Sound detected (RMS: {rms:.4f}) – starting capture...")
                triggered = True
                buffer = []
                frames_since_trigger = 0
            buffer.append(audio_chunk)
            frames_since_trigger = 0
        else:
            if triggered:
                frames_since_trigger += 1
                buffer.append(audio_chunk)  # Keep buffering even during short silences

                # Reset if silence lasts too long (~1 second)
                if frames_since_trigger > int(RATE / CHUNK):
                    triggered = False
                    buffer = []

        # If we have enough audio when triggered
        if triggered and len(buffer) * CHUNK >= RATE * RECORD_SECONDS:
            print("Capture complete – processing spectrogram...")
            audio_np = np.concatenate(buffer)[:int(RATE * RECORD_SECONDS)]  # Trim to exact length

            # Save detected spectrogram
            timestamp = len(glob.glob(os.path.join(DETECTED_DIR, "*.png")))
            new_path = os.path.join(DETECTED_DIR, f"detected_{timestamp}.png")
            generate_spectrogram(audio_np, new_path)

            # Compare to library
            best_match, best_score, all_results = compare_to_library(new_path)

            print(f"\nClosest match: {best_match}")
            print(f"Similarity score: {best_score:.4f} (1.0 = identical, >0.8 usually very similar)")
            print("Top 3 matches:")
            for name, score in all_results[:3]:
                print(f"  - {name}: {score:.4f}")
            print("-" * 50)

            # Reset for next detection
            triggered = False
            buffer = []

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
