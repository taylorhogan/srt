import librosa
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def compare_audio_files(file1_path: str, file2_path: str) -> float:
    """
    Compare two audio files and return similarity percentage.
    
    Args:
        file1_path (str): Path to first audio file
        file2_path (str): Path to second audio file
        
    Returns:
        float: Similarity percentage between 0 and 100
    """
    try:
        # Load audio files
        y1, sr1 = librosa.load(file1_path)
        y2, sr2 = librosa.load(file2_path)
        
        # Compute mel spectrograms
        mel_spect1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
        mel_spect2 = librosa.feature.melspectrogram(y=y2, sr=sr2)
        
        # Convert to log scale
        mel_spect1_db = librosa.power_to_db(mel_spect1, ref=np.max)
        mel_spect2_db = librosa.power_to_db(mel_spect2, ref=np.max)
        
        # Flatten the spectrograms
        flat_spect1 = mel_spect1_db.flatten()
        flat_spect2 = mel_spect2_db.flatten()
        
        # If spectrograms have different lengths, truncate to shorter one
        min_length = min(len(flat_spect1), len(flat_spect2))
        flat_spect1 = flat_spect1[:min_length]
        flat_spect2 = flat_spect2[:min_length]
        
        # Calculate cosine similarity
        similarity = 1 - cosine(flat_spect1, flat_spect2)
        
        # Convert to percentage and ensure it's between 0 and 100
        similarity_percentage = max(0, min(100, similarity * 100))
        
        return similarity_percentage
        
    except Exception as e:
        print(f"Error comparing audio files: {str(e)}")
        return 0.0


def plot_audio_file(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Create a figure with two subplots
    plt.figure(figsize=(12, 8))

    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example paths - replace with actual file paths
    file1 = "rec2.wav"
    file2 = "rec4.wav"
    
    similarity = compare_audio_files(file1, file2)
    print(f"Audio files are {similarity:.2f}% similar")
    plot_audio_file(file1)
    plot_audio_file(file2)
