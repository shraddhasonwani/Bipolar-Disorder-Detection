import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
try:
    import opensmile
except ImportError:
    print(" OpenSMILE not found. Install using: pip install opensmile")
    opensmile = None


AUDIO_FOLDER = "BP_Organized_Dataset/audio"

OUTPUT_AUDIO_FEATURES = "Processed_Features/audio"
OUTPUT_SPECTROGRAMS = "Processed_Features/spectrograms"

SAMPLE_RATE = 16000
MAX_DURATION = 30          # seconds
N_MELS = 128
TRIM_DB = 25               # silence removal threshold

os.makedirs(OUTPUT_AUDIO_FEATURES, exist_ok=True)
os.makedirs(OUTPUT_SPECTROGRAMS, exist_ok=True)

smile = None
if opensmile:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPS,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print(" OpenSMILE initialized with eGeMAPS feature set")


def load_and_clean_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    return y, sr

def extract_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)
    return mel_db

all_opensmile_features = []

print("\n Starting Audio Preprocessing...\n")

for file in tqdm(sorted(os.listdir(AUDIO_FOLDER))):
    if not file.lower().endswith(".wav"):
        continue

    participant_id = os.path.splitext(file)[0]
    file_path = os.path.join(AUDIO_FOLDER, file)

    try:
        y, sr = load_and_clean_audio(file_path)

        if smile:
            df_smile = smile.process_file(file_path)
            feature_dict = df_smile.iloc[0].to_dict()
            feature_dict["participant_id"] = participant_id
            all_opensmile_features.append(feature_dict)

        mel_spec = extract_mel_spectrogram(y, sr)

        spec_filename = f"{participant_id}_mel.npy"
        np.save(os.path.join(OUTPUT_SPECTROGRAMS, spec_filename), mel_spec)

    except Exception as e:
        print(f"\n Error processing {file}: {e}")
        continue

if all_opensmile_features:
    df_features = pd.DataFrame(all_opensmile_features)

    cols = ["participant_id"] + [c for c in df_features.columns if c != "participant_id"]
    df_features = df_features[cols]

    csv_path = os.path.join(OUTPUT_AUDIO_FEATURES, "audio_eGeMAPS_features.csv")
    df_features.to_csv(csv_path, index=False)

    df_features["participant_id"].to_csv(
        os.path.join(OUTPUT_AUDIO_FEATURES, "audio_participants.txt"),
        index=False
    )

    print("\n Audio preprocessing completed successfully!")
    print(f" ML Features CSV: {csv_path}")
    print(f" DL Spectrograms: {OUTPUT_SPECTROGRAMS}")

else:
    print("\n⚠ No OpenSMILE features extracted. Check audio files or OpenSMILE installation.")
