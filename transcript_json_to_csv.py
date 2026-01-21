import os
import json
import pandas as pd

# ===============================
# Paths
# ===============================
TRANSCRIPT_DIR = "Processed_Features/transcript"
OUTPUT_DIR = os.path.join(TRANSCRIPT_DIR)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "transcript_advanced_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Process JSON files
# ===============================
rows = []

for file in os.listdir(TRANSCRIPT_DIR):
    if file.endswith("_TRANSCRIPT_features.json"):
        pid = file.split("_")[0]  # extract participant ID

        file_path = os.path.join(TRANSCRIPT_DIR, file)
        with open(file_path, "r") as f:
            features = json.load(f)

        # Flatten if nested dictionary (one level)
        flat_features = {}
        for k, v in features.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat_features[f"{k}_{subk}"] = subv
            else:
                flat_features[k] = v

        # Add participant_id
        flat_features["participant_id"] = pid

        rows.append(flat_features)

# ===============================
# Create DataFrame & Save CSV
# ===============================
if rows:
    df = pd.DataFrame(rows)

    # Ensure participant_id is first column
    cols = ["participant_id"] + [c for c in df.columns if c != "participant_id"]
    df = df[cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(" Transcript CSV created:", OUTPUT_CSV)
else:
    print(" No transcript JSON files found in:", TRANSCRIPT_DIR)
