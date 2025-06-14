# Feature extraction utils
import numpy as np
import librosa

def extract_features_from_audio(audio_file, sr=22050):
    """
    Extracts robust MFCC features from a .wav file, including preprocessing.
    
    Parameters:
    - audio_file: path to the .wav file (string or file-like object)
    - sr: target sample rate (default = 22050)

    Returns:
    - feature_vector: 1D numpy array of MFCCs (length = 40) or None if invalid
    """
    try:
        # Load audio with consistent sample rate
        y, sr = librosa.load(audio_file, sr=sr)

        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y)

        # Normalize volume
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

        # Skip if audio is too quiet or too short
        if np.max(np.abs(y)) < 0.01 or len(y) < sr * 1.5:
            print(f"⚠️ Skipped file {audio_file}: too quiet or too short")
            return None

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None
