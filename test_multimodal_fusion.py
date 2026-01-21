import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================== PATHS =====================
BASE_DIR = "BP_Organized_Dataset"
NORMALIZED_FEAT_DIR = os.path.join("Processed_Features", "normalized")

LABELS_PATH = os.path.join(BASE_DIR, "labels", "labels.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "multimodal_model.pth")
OUTPUT_CSV  = os.path.join(BASE_DIR, "results", "test_predictions.csv")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ===================== DEVICE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== LOAD LABELS =====================
labels_df = pd.read_csv(LABELS_PATH)
labels_df["participant_id"] = labels_df["participant_id"].apply(
    lambda x: int(''.join(filter(str.isdigit, str(x))))
)
labels_df["label"] = labels_df["label"].astype(int)
print("Labels loaded:", labels_df.shape)

# ===================== LOAD FEATURES =====================
feature_files = {
    "audio": os.path.join(NORMALIZED_FEAT_DIR, "audio_normalized.csv"),
    "video": os.path.join(NORMALIZED_FEAT_DIR, "video_normalized.csv"),
    "text":  os.path.join(NORMALIZED_FEAT_DIR, "text_normalized.csv"),
}

feature_dfs = {}

for modality, path in feature_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["participant_id"] = df["participant_id"].apply(
            lambda x: int(''.join(filter(str.isdigit, str(x))))
        )
        feature_dfs[modality] = df
        print(f"{modality} features loaded:", df.shape)
    else:
        print(f"WARNING: {modality} features not found")

# ===================== ALIGN PARTICIPANTS =====================
all_pids = labels_df["participant_id"].values

for modality, df in feature_dfs.items():
    missing = set(all_pids) - set(df["participant_id"].values)
    if missing:
        zero_rows = pd.DataFrame(0, index=range(len(missing)), columns=df.columns)
        zero_rows["participant_id"] = list(missing)
        df = pd.concat([df, zero_rows], ignore_index=True)
    feature_dfs[modality] = df.sort_values("participant_id").reset_index(drop=True)

# ===================== MERGE FEATURES =====================
merged_df = labels_df.copy()

for df in feature_dfs.values():
    merged_df = merged_df.merge(df, on="participant_id", how="left")

merged_df.fillna(0, inplace=True)
print("Final merged shape:", merged_df.shape)

# ===================== PREPARE DATA =====================
X = merged_df.drop(columns=["participant_id", "label"]).values
y_true = merged_df["label"].values

# ===================== SCALE + PCA (SAME AS TRAINING) =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = min(200, X_scaled.shape[0], X_scaled.shape[1])
pca = PCA(n_components=n_components, svd_solver="full")
X_pca = pca.fit_transform(X_scaled)

# ===================== LOAD MODEL =====================
class MultimodalFusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

model = MultimodalFusionNet(X_pca.shape[1]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===================== PREDICTION =====================
X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs = model(X_tensor)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

# ===================== SAVE RESULTS =====================
results_df = pd.DataFrame({
    "participant_id": merged_df["participant_id"],
    "true_label": y_true,
    "predicted_label": y_pred
})

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n Predictions saved to: {OUTPUT_CSV}")

# ===================== METRICS =====================
print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
