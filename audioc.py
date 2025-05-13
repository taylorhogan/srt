import librosa
import numpy as np
from scipy.signal import correlate


def compare_audio_files(file1_path, file2_path):
    try:
        # Load the audio files
        y1, sr1 = librosa.load(file1_path, sr=None)
        y2, sr2 = librosa.load(file2_path, sr=None)

        # Resample to the same sampling rate if they differ
        if sr1 != sr2:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
            sr = sr1
        else:
            sr = sr1

        # Trim to the same length (use the shorter file's length)
        min_length = min(len(y1), len(y2))
        y1 = y1[:min_length]
        y2 = y2[:min_length]

        # Normalize the audio signals to have zero mean and unit variance
        y1 = (y1 - np.mean(y1)) / np.std(y1)
        y2 = (y2 - np.mean(y2)) / np.std(y2)

        # Compute cross-correlation
        corr = correlate(y1, y2, mode='full')
        max_corr = np.max(corr)

        # Normalize correlation to get a similarity score
        # Divide by the auto-correlation of y1 to scale the result
        auto_corr = np.sqrt(np.sum(y1 ** 2) * np.sum(y2 ** 2))
        if auto_corr == 0:
            return 0.0  # Avoid division by zero
        similarity = max_corr / auto_corr

        # Convert to percentage (clip to [0, 1] range)
        similarity_percentage = np.clip(similarity * 100, 0, 100)

        return similarity_percentage

    except Exception as e:
        print(f"Error processing audio files: {e}")
        return 0.0


# Example usage
if __name__ == "__main__":
    file1 = "rec2.wav"  # Replace with your first audio file path
    file2 = "rec1.wav"  # Replace with your second audio file path

    similarity = compare_audio_files(file1, file2)
    print(f"Similarity between the two audio files: {similarity:.2f}%")