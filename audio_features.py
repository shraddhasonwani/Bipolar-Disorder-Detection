import librosa
import numpy as np

def extract_audio_features(audio_path):
    # ⛔ LIMIT TO FIRST 30 SECONDS
    y, sr = librosa.load(audio_path, sr=16000, duration=30)

    if len(y) == 0:
        raise ValueError("Empty audio file")

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Energy
    energy = float(np.mean(librosa.feature.rms(y=y)))

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0

    features = np.hstack([mfcc_mean, energy, pitch])

    return features
