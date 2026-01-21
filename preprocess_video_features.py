import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import warnings

# Suppress the FutureWarning about to_numeric
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
OPENFACE_OUTPUT_FOLDER = "BP_Organized_Dataset/openface_output"
OUTPUT_FOLDER = "Processed_Features/video"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Updated to match the short names found in your DAIC-WOZ files
CRITICAL_COLS = [
    # Head Pose
    "Tx", "Ty", "Tz", "Rx", "Ry", "Rz",
    # Gaze 
    "x_0", "y_0", "z_0", "x_1", "y_1", "z_1",
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    # Action Units (including intensity and presence formats)
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU09", "AU10", "AU12", 
    "AU14", "AU15", "AU17", "AU20", "AU25", "AU26", "AU45",
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU09_r", "AU10_r", 
    "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU25_r", "AU26_r", "AU45_r"
]

def calculate_functionals(df, col):
    """
    Convert a frame-level signal into statistical descriptors
    """
    series = df[col].replace([np.inf, -np.inf], np.nan)
    mean_val = series.mean()
    series = series.fillna(mean_val if pd.notnull(mean_val) else 0.0)

    if series.isnull().all() or len(series) < 2:
        return {f"{col}_{func}": 0.0 for func in ["mean", "std", "max", "min", "range", "median", "skew", "kurtosis", "25p", "75p", "delta_mean"]}

    values = series.values
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    delta = np.abs(np.diff(values)).mean() if len(values) > 1 else 0.0

    return {
        f"{col}_mean": series.mean(),
        f"{col}_std": series.std(),
        f"{col}_max": series.max(),
        f"{col}_min": series.min(),
        f"{col}_range": series.max() - series.min(),
        f"{col}_median": series.median(),
        f"{col}_skew": skew(values, bias=False) if len(values) > 2 else 0.0,
        f"{col}_kurtosis": kurtosis(values, bias=False) if len(values) > 3 else 0.0,
        f"{col}_25p": p25,
        f"{col}_75p": p75,
        f"{col}_delta_mean": delta,
    }

print("\n--- Starting DAIC-WOZ Video Feature Preprocessing ---\n")

all_features = []
files = sorted([f for f in os.listdir(OPENFACE_OUTPUT_FOLDER) if f.endswith(".txt")])

if not files:
    print(f" Warning: No .txt files found in {OPENFACE_OUTPUT_FOLDER}")

for file_name in tqdm(files, desc="Processing Videos"):
    pid_full = os.path.splitext(file_name)[0]
    file_path = os.path.join(OPENFACE_OUTPUT_FOLDER, file_name)

    try:
        # Detect delimiter automatically
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.columns = [str(c).strip() for c in df.columns]

        # Normalization: Map short names to standard names for the final CSV
        mapping = {
            "x_0": "gaze_0_x", "y_0": "gaze_0_y", "z_0": "gaze_0_z",
            "Tx": "pose_Tx", "Ty": "pose_Ty", "Tz": "pose_Tz",
            "Rx": "pose_Rx", "Ry": "pose_Ry", "Rz": "pose_Rz"
        }
        df.rename(columns=mapping, inplace=True)
        
        df = df.apply(pd.to_numeric, errors='coerce')
    except Exception:
        continue

    # Update available columns based on the potentially renamed columns
    current_critical = [mapping.get(c, c) for c in CRITICAL_COLS]
    available_cols = [c for c in current_critical if c in df.columns]

    if not available_cols:
        # Only print if it's not a landmarks-only file (x0, x1...)
        if 'x0' not in df.columns:
            print(f"\n {pid_full} skipped. No target columns found.")
        continue

    session_features = {"participant_id": pid_full}
    
    for col in available_cols:
        session_features.update(calculate_functionals(df, col))

    session_features["tracking_success_ratio"] = df["success"].mean() if "success" in df.columns else 1.0
    session_features["frame_count"] = len(df)
    all_features.append(session_features)

# Finalizing output
if all_features:
    final_df = pd.DataFrame(all_features)
    
    # Extract numeric ID (the part before the first underscore)
    final_df['participant_id'] = final_df['participant_id'].astype(str).str.split('_').str[0]
    
    # Group by participant and average the features from different files (AUs, Gaze, Pose)
    final_df = final_df.groupby('participant_id').mean().reset_index()

    output_path = os.path.join(OUTPUT_FOLDER, "video_features_functionals.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\n Merged Success! Saved {len(final_df)} participants to: {output_path}")
    print(f"Total features per participant: {final_df.shape[1] - 1}")
else:
    print("\n No features were extracted. Please check file headers.")