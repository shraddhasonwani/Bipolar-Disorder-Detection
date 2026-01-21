import pandas as pd

# ================= PATHS =================
AUDIO_PATH = "Processed_Features/audio/audio_eGeMAPS_features.csv"
VIDEO_PATH = "Processed_Features/video/video_features_functionals.csv"
TEXT_PATH  = "Processed_Features/transcript/transcript_advanced_features.csv"
LABELS_PATH = "BP_Organized_Dataset/labels/labels.csv"

OUTPUT_PATH = "Processed_Features/fusion_output/multimodal_features.csv"

# ================= LOAD DATA =================
print("Loading feature files...")

audio = pd.read_csv(AUDIO_PATH)
video = pd.read_csv(VIDEO_PATH)
text  = pd.read_csv(TEXT_PATH)
labels = pd.read_csv(LABELS_PATH)

print("Audio shape:", audio.shape)
print("Video shape:", video.shape)
print("Text shape :", text.shape)
print("Labels shape:", labels.shape)

# ---------- FIX participant_id TYPES ----------
# Extract numeric part of participant_id and convert to int
for df in [audio, video, text, labels]:
    if "participant_id" not in df.columns:
        raise ValueError("participant_id column missing in one of the files!")
    df["participant_id"] = df["participant_id"].astype(str).str.extract(r'(\d+)').astype(int)

# ================= MERGE =================
# Use inner merge to keep only participants present in all modalities
merged = audio.merge(video, on="participant_id", how="inner")
merged = merged.merge(text, on="participant_id", how="inner")
merged = merged.merge(labels, on="participant_id", how="inner")

print("Final merged shape:", merged.shape)

# ================= HANDLE MISSING VALUES =================
# Optional: fill missing values if any
merged.fillna(0, inplace=True)

# ================= SAVE =================
merged.to_csv(OUTPUT_PATH, index=False)
print("Multimodal features saved to:", OUTPUT_PATH)
