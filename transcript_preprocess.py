import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ================= CONFIG =================
TRANSCRIPT_DIR = "Processed_Features/transcript"
OUTPUT_DIR = "Processed_Features/transcript"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"  # FAST
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODEL =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print(f"\nModel loaded on {device}")
print("Starting Transcript Preprocessing...\n")

# ================= PROCESS =================
files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".txt")]

for file in tqdm(files):
    pid = file.split(".")[0]
    path = os.path.join(TRANSCRIPT_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if len(text) == 0:
        continue

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    features = {
        "participant_id": pid,
        "distilbert_embedding": embedding.tolist()
    }

    out_path = os.path.join(OUTPUT_DIR, f"{pid}_TRANSCRIPT_features.json")
    with open(out_path, "w") as f:
        json.dump(features, f)

print("\n✅ Transcript preprocessing COMPLETED successfully!")
