# ===============================
# Script: data_preprocessing_script.py
# ===============================

import os
import re
import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===============================
# CONFIG
# ===============================
FEAT_DIR = "Processed_Features"
OUTPUT_DIR = os.path.join(FEAT_DIR, "normalized")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS_FILE = "BP_Organized_Dataset/labels/labels.csv"
ID_COL = "participant_id"

FEATURE_FILES = {
    "audio": os.path.join(FEAT_DIR, "audio", "audio_eGeMAPS_features.csv"),
    "video": os.path.join(FEAT_DIR, "video", "video_features_functionals.csv"),
    "text": os.path.join(FEAT_DIR, "transcript", "transcript_advanced_features.csv"),
}

# ===============================
# Helper: clean participant ID
# ===============================
def clean_pid(pid):
    """Extract numeric participant ID and return as integer."""
    match = re.search(r"\d+", str(pid))
    return int(match.group()) if match else None

# ===============================
# Helper: convert stringified lists to numeric columns
# ===============================
def expand_list_columns(df):
    for col in df.columns:
        if df[col].dtype == object and df[col].astype(str).str.startswith('[').any():
            # Convert string list to actual list safely
            def safe_eval(x):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return [float(x)] if x.replace('.', '', 1).isdigit() else [0]
            df[col] = df[col].apply(safe_eval)

            # Expand list into multiple columns
            df_expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            df_expanded.columns = [f"{col}_{i}" for i in range(df_expanded.shape[1])]
            df = df.drop(columns=[col]).join(df_expanded)
    return df

# ===============================
def apply_scaler(df, train_ids):
    df_features = df.drop(columns=[ID_COL], errors="ignore")
    df_features = expand_list_columns(df_features)

    # Fit scaler on train IDs only
    train_df = df[df[ID_COL].isin(train_ids)].drop(columns=[ID_COL], errors="ignore")
    train_df = expand_list_columns(train_df)

    if train_df.empty:
        raise ValueError(" No matching TRAIN samples after ID alignment.")

    scaler = StandardScaler()
    scaler.fit(train_df)

    scaled = scaler.transform(df_features)
    scaled_df = pd.DataFrame(scaled, columns=df_features.columns)
    scaled_df.insert(0, ID_COL, df[ID_COL].values)

    return scaled_df

# ===============================
def main():

    # ---- Load labels ----
    labels_df = pd.read_csv(LABELS_FILE)
    labels_df[ID_COL] = labels_df[ID_COL].apply(clean_pid)

    # Ensure label column exists
    if "label" not in labels_df.columns:
        raise ValueError("Label column missing in labels CSV!")

    # Drop rows with missing labels
    labels_df = labels_df.dropna(subset=["label"])

    # Split train/test IDs
    train_ids, _ = train_test_split(
        labels_df[ID_COL].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=labels_df["label"]
    )

    print(" Starting Z-score normalization...")

    for modality, path in FEATURE_FILES.items():
        if not os.path.exists(path):
            print(f" Missing {modality} file → skipped")
            continue

        print(f"\n Processing {modality} features...")
        df = pd.read_csv(path)

        # Clean IDs
        df[ID_COL] = df[ID_COL].apply(clean_pid)

        # Keep only participants present in labels
        df = df[df[ID_COL].isin(labels_df[ID_COL])]

        if df.empty:
            print(f" No matching IDs for {modality}. Skipping.")
            continue

        df_norm = apply_scaler(df, train_ids)

        out_path = os.path.join(OUTPUT_DIR, f"{modality}_normalized.csv")
        df_norm.to_csv(out_path, index=False)

        print(f" Saved → {out_path}")

    print("\n Normalization completed successfully!")

# ===============================
if __name__ == "__main__":
    main()
