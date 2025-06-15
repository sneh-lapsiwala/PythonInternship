# Feature extraction utils
import numpy as np
import librosa

def extract_features_from_audio(audio_file, sr=22050, duration=None, offset=0.0):
    """
    Extracts a comprehensive set of audio features (MFCCs, Chroma, ZCR)
    from an audio file, including robust preprocessing.

    Parameters:
    - audio_file: path to the audio file (string or file-like object)
    - sr: target sample rate (default = 22050)
    - duration: maximum duration to load (in seconds). None for full duration.
    - offset: start loading audio at this many seconds into the file.

    Returns:
    - feature_vector: 1D numpy array of combined features, or None if invalid.
    """
    try:
        # Load audio with consistent sample rate and optional duration/offset
        y, sr = librosa.load(audio_file, sr=sr, duration=duration, offset=offset)

        # Trim leading and trailing silence
        # This is generally a good idea, as silence doesn't contribute much to features
        y, _ = librosa.effects.trim(y)

        # Normalize volume
        # Helps ensure features aren't skewed by varying audio loudness
        # Avoid division by zero if y is entirely zero (e.g., after trimming)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        else:
            # If after trimming and normalization attempt, y is still all zeros,
            # it indicates a silent or empty audio segment.
            print(f"⚠️ Skipped file {audio_file}: audio became entirely silent after processing.")
            return None

        # Skip if audio is too quiet or too short after preprocessing
        # A threshold of 0.001 is often more appropriate for normalized audio
        # Minimum duration check (e.g., at least 0.5 seconds of audio)
        if np.max(np.abs(y)) < 0.001 or len(y) < sr * 0.5:
            print(f"⚠️ Skipped file {audio_file}: too quiet or too short after preprocessing")
            return None

        # Extract a richer set of features
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Chroma features (Perceptual representation of pitch content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Zero-Crossing Rate (Rate at which the signal changes sign, useful for percussive sounds)
        zcr = librosa.feature.zero_crossing_rate(y)

        # Compute the mean of each feature over time and concatenate
        # Taking the mean provides a single summary vector for the entire audio segment
        features = np.hstack([
            np.mean(mfccs.T, axis=0),   # (40,)
            np.mean(chroma.T, axis=0),  # (12,)
            np.mean(zcr.T, axis=0)      # (1,)
        ])

        return features

    except Exception as e:
        print(f"❌ Error extracting features from {audio_file}: {e}")
        return None

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Create a dummy audio file for testing (requires soundfile library)
    try:
        import soundfile as sf
        dummy_audio_path = "dummy_audio.wav"
        # Generate a 5-second sine wave
        sr_test = 22050
        duration_test = 5
        frequency_test = 440  # A4 note
        t = np.linspace(0, duration_test, int(sr_test * duration_test), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency_test * t)
        sf.write(dummy_audio_path, audio_data, sr_test)
        print(f"Created dummy audio file: {dummy_audio_path}")

        # Test the updated function
        print("\nTesting with full audio:")
        features_full = extract_features_from_audio_updated(dummy_audio_path)
        if features_full is not None:
            print(f"Extracted features shape: {features_full.shape}") # Should be (53,)
            print(f"First 5 features: {features_full[:5]}")
        else:
            print("Feature extraction failed.")

        print("\nTesting with specific duration/offset:")
        features_segment = extract_features_from_audio_updated(dummy_audio_path, duration=3, offset=1)
        if features_segment is not None:
            print(f"Extracted features shape (segment): {features_segment.shape}")
        else:
            print("Feature extraction failed for segment.")

        # Test with a path that doesn't exist
        print("\nTesting with non-existent file:")
        features_non_existent = extract_features_from_audio_updated("non_existent_audio.wav")
        if features_non_existent is None:
            print("Correctly handled non-existent file.")

        # Test with a very short "audio" (simulating an issue)
        print("\nTesting with very short audio (simulated):")
        with open("short_audio.wav", "wb") as f:
            f.write(b'\x00' * 100) # Write some dummy bytes, not a real WAV
        features_short = extract_features_from_audio_updated("short_audio.wav")
        if features_short is None:
            print("Correctly handled short/invalid audio.")
        import os
        os.remove("short_audio.wav")

        # Test with completely silent audio (simulating an issue)
        print("\nTesting with silent audio (simulated):")
        silent_audio_path = "silent_audio.wav"
        sf.write(silent_audio_path, np.zeros(sr_test), sr_test)
        features_silent = extract_features_from_audio_updated(silent_audio_path)
        if features_silent is None:
            print("Correctly handled silent audio.")


    except ImportError:
        print("\nSkipping example usage: 'soundfile' library not installed. Install with 'pip install soundfile'")
    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")
    finally:
        # Clean up dummy files
        if 'dummy_audio_path' in locals() and os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
        if 'silent_audio_path' in locals() and os.path.exists(silent_audio_path):
            os.remove(silent_audio_path)