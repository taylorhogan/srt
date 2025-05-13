import pyaudio
import wave
import sys


def record_audio(output_filename, duration=5, sample_rate=44100, channels=1, chunk=1024):
    """
    Record audio from the microphone and save it to a WAV file.

    Parameters:
    - output_filename: Name of the output WAV file (e.g., 'output.wav')
    - duration: Recording duration in seconds (default: 5)
    - sample_rate: Sampling rate in Hz (default: 44100)
    - channels: Number of audio channels (default: 1 for mono)
    - chunk: Buffer size for audio frames (default: 1024)
    """
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    try:
        # Set up the audio stream
        stream = audio.open(format=pyaudio.paInt16,  # 16-bit resolution
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk)

        print("Recording...")

        # Record audio in chunks
        frames = []
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        print("Recording finished.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Save the recorded audio to a WAV file
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        print(f"Audio saved to {output_filename}")

    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        # Terminate PyAudio
        audio.terminate()


if __name__ == "__main__":
    # Example usage
    output_file = "rec4.wav"  # Output file name
    record_duration = 20  # Record for 5 seconds

    record_audio(output_file, duration=record_duration)