import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# ===================== PATHS =====================
BASE_DIR = "BP_Organized_Dataset"
NORMALIZED_FEAT_DIR = os.path.join("Processed_Features", "normalized")
LABELS_PATH = os.path.join(BASE_DIR, "labels", "labels.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "multimodal_model.pth")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs("scalers", exist_ok=True)

# ===================== DEVICE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== LOAD LABELS =====================
labels_df = pd.read_csv(LABELS_PATH)

labels_df["participant_id"] = labels_df["participant_id"].apply(
    lambda x: int("".join(filter(str.isdigit, str(x))))
)

labels_df = labels_df.dropna(subset=["label"])
labels_df["label"] = labels_df["label"].astype(int)

print("Labels loaded:", labels_df.shape)

# ===================== LOAD NORMALIZED FEATURES =====================
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
            lambda x: int("".join(filter(str.isdigit, str(x))))
        )
        feature_dfs[modality] = df
        print(f"{modality} features loaded:", df.shape)
    else:
        print(f"⚠ Warning: {modality} file not found at {path}")

# ===================== ALIGN PARTICIPANTS =====================
all_pids = labels_df["participant_id"].unique()

for modality, df in feature_dfs.items():
    missing_pids = set(all_pids) - set(df["participant_id"].values)
    if missing_pids:
        zero_rows = pd.DataFrame(0, index=range(len(missing_pids)), columns=df.columns)
        zero_rows["participant_id"] = list(missing_pids)
        df = pd.concat([df, zero_rows], ignore_index=True)

    feature_dfs[modality] = df.sort_values("participant_id").reset_index(drop=True)

# ===================== MERGE FEATURES =====================
merged_df = labels_df.copy()

for modality, df in feature_dfs.items():
    merged_df = merged_df.merge(df, on="participant_id", how="left")

merged_df.fillna(0, inplace=True)

print("All features merged:", merged_df.shape)

# ===================== SPLIT FEATURES / LABELS =====================
X = merged_df.drop(columns=["participant_id", "label"]).values
y = merged_df["label"].values

# ===================== SCALE FEATURES =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===================== PCA =====================
desired_components = 200
max_components = min(X_scaled.shape[0], X_scaled.shape[1])
n_components = min(desired_components, max_components)

print(f"Applying PCA with n_components = {n_components}")

pca = PCA(n_components=n_components, svd_solver="full")
X_pca = pca.fit_transform(X_scaled)

print("PCA output shape:", X_pca.shape)

# ===================== TRAIN / VAL SPLIT =====================
train_ids, val_ids = train_test_split(
    merged_df["participant_id"].values,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_mask = np.isin(merged_df["participant_id"], train_ids)
val_mask   = np.isin(merged_df["participant_id"], val_ids)

X_train, y_train = X_pca[train_mask], y[train_mask]
X_val,   y_val   = X_pca[val_mask],   y[val_mask]

# ===================== DATASET =====================
class MultimodalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MultimodalDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(MultimodalDataset(X_val, y_val), batch_size=32, shuffle=False)

# ===================== MODEL =====================
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

model = MultimodalFusionNet(
    input_dim=X_train.shape[1],
    num_classes=len(np.unique(y))
).to(device)

# ===================== TRAINING =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()

    train_acc = correct / len(train_loader.dataset)

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_outputs = model(X_val_batch)
            val_correct += (val_outputs.argmax(1) == y_val_batch).sum().item()

    val_acc = val_correct / len(val_loader.dataset)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {total_loss/len(train_loader):.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

# ===================== SAVE MODEL + PREPROCESSORS =====================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
joblib.dump(scaler, "scalers/fusion_scaler.pkl")
joblib.dump(pca, "scalers/fusion_pca.pkl")

print("\n✅ Model, scaler, and PCA saved successfully")
