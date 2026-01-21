import pandas as pd
import os

# ---------------- PATHS ----------------
LABEL_DIR = "BP_Organized_Dataset/labels"

TRAIN_CSV = os.path.join(LABEL_DIR, "train_split_Depression_AVEC2017.csv")
DEV_CSV   = os.path.join(LABEL_DIR, "dev_split_Depression_AVEC2017.csv")

OUTPUT_CSV = os.path.join(LABEL_DIR, "labels.csv")

# ---------------- LOAD SPLITS ----------------
train_df = pd.read_csv(TRAIN_CSV)
dev_df = pd.read_csv(DEV_CSV)

# ---------------- SELECT REQUIRED COLUMNS ----------------
train_df = train_df[["Participant_ID", "PHQ8_Binary"]]
dev_df = dev_df[["Participant_ID", "PHQ8_Binary"]]

# ---------------- MERGE TRAIN + DEV ----------------
labels_df = pd.concat([train_df, dev_df], ignore_index=True)

# ---------------- RENAME COLUMNS ----------------
labels_df = labels_df.rename(columns={
    "Participant_ID": "participant_id",
    "PHQ8_Binary": "label"
})

# ---------------- SORT & SAVE ----------------
labels_df = labels_df.sort_values("participant_id").reset_index(drop=True)
labels_df.to_csv(OUTPUT_CSV, index=False)

# ---------------- REPORT ----------------
print(" REAL labels.csv created successfully!")
print("Saved at:", OUTPUT_CSV)
print("\nTotal participants:", len(labels_df))
print("\nLabel distribution:")
print(labels_df["label"].value_counts())
print("\nSample:")
print(labels_df.head())
