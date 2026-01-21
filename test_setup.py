"""
Quick test script to verify all dependencies and files are properly loaded
"""
import os
import sys

print("=" * 60)
print("TESTING PROJECT SETUP")
print("=" * 60)

# Test 1: Check critical files
print("\n1. Checking critical files...")
files_to_check = [
    "BP_Organized_Dataset/models/multimodal_model.pth",
    "scalers/audio_scaler.pkl",
    "scalers/video_scaler.pkl",
    "templates/index.html",
    "templates/dashboard.html",
    "templates/voice_prediction.html",
    "templates/navbar.html",
    "templates/footer.html"
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - MISSING!")

# Test 2: Check imports
print("\n2. Checking Python dependencies...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except:
    print("   ✗ PyTorch - NOT INSTALLED")

try:
    import librosa
    print(f"   ✓ librosa")
except:
    print("   ✗ librosa - NOT INSTALLED")

try:
    import soundfile
    print(f"   ✓ soundfile")
except:
    print("   ✗ soundfile - NOT INSTALLED")

try:
    import cv2
    print(f"   ✓ OpenCV")
except:
    print("   ✗ OpenCV - NOT INSTALLED")

try:
    from flask import Flask
    print(f"   ✓ Flask")
except:
    print("   ✗ Flask - NOT INSTALLED")

try:
    from transformers import DistilBertTokenizer
    print(f"   ✓ Transformers")
except:
    print("   ✗ Transformers - NOT INSTALLED")

try:
    import joblib
    print(f"   ✓ joblib")
except:
    print("   ✗ joblib - NOT INSTALLED")

# Test 3: Try loading model
print("\n3. Testing model loading...")
try:
    import torch
    model_path = "BP_Organized_Dataset/models/multimodal_model.pth"
    state_dict = torch.load(model_path, map_location="cpu")
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Model has {len(state_dict)} layers")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")

# Test 4: Try loading scalers
print("\n4. Testing scaler loading...")
try:
    import joblib
    audio_scaler = joblib.load("scalers/audio_scaler.pkl")
    print(f"   ✓ Audio scaler loaded")
except Exception as e:
    print(f"   ✗ Audio scaler failed: {e}")

try:
    video_scaler = joblib.load("scalers/video_scaler.pkl")
    print(f"   ✓ Video scaler loaded")
except Exception as e:
    print(f"   ✗ Video scaler failed: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nIf all tests passed, run: python app.py")
print("Then open: http://127.0.0.1:5000")
