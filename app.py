import os
import io
import torch
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import traceback
import librosa
import soundfile as sf
import cv2
from transformers import DistilBertTokenizer, DistilBertModel

# ================= CONFIG =================
MODEL_PATH = "BP_Organized_Dataset/models/multimodal_model.pth"
AUDIO_SCALER_PATH = "scalers/audio_scaler.pkl"
VIDEO_SCALER_PATH = "scalers/video_scaler.pkl"

SAMPLE_RATE = 16000
MAX_DURATION = 30
N_MELS = 128
TRIM_DB = 25
FEATURE_SIZE = 142

# ================= FLASK =================
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app)

# ================= MONGODB =================
try:
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    db = client['bipolar_detection']
    users_collection = db['users']
    sessions_collection = db['sessions']
    
    # Test connection
    client.admin.command('ping')
    print("✓ MongoDB connected")
    MONGODB_AVAILABLE = True
except Exception as e:
    print(f"⚠ MongoDB not available: {e}")
    print("⚠ Running without database - authentication disabled")
    client = None
    db = None
    users_collection = None
    sessions_collection = None
    MONGODB_AVAILABLE = False

# ================= MODEL =================
class MultimodalFusion(torch.nn.Module):
    def __init__(self, input_size=142, hidden1=128, hidden2=128, output_size=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

model = MultimodalFusion(input_size=FEATURE_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ================= SCALERS =================
audio_scaler = joblib.load(AUDIO_SCALER_PATH) if os.path.exists(AUDIO_SCALER_PATH) else None
video_scaler = joblib.load(VIDEO_SCALER_PATH) if os.path.exists(VIDEO_SCALER_PATH) else None

print(f"[OK] Model loaded from {MODEL_PATH}")
print(f"[OK] Audio scaler: {'Loaded' if audio_scaler else 'Not found'}")
print(f"[OK] Video scaler: {'Loaded' if video_scaler else 'Not found'}")

# ================= TEXT MODEL =================
try:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    bert.eval()
    print("[OK] BERT model loaded")
except Exception as e:
    print(f"[WARNING] BERT model failed to load: {e}")
    tokenizer = None
    bert = None

# ================= PREPROCESS =================
def preprocess_audio_bytes(audio_bytes):
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
        y = y[:SAMPLE_RATE * MAX_DURATION]
        
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
        feat = np.mean(librosa.power_to_db(mel, ref=np.max).T, axis=0)
        
        if audio_scaler:
            expected_features = audio_scaler.n_features_in_
            if len(feat) < expected_features:
                feat = np.pad(feat, (0, expected_features - len(feat)))
            else:
                feat = feat[:expected_features]
            feat = audio_scaler.transform([feat])[0]
        else:
            feat = np.pad(feat, (0, FEATURE_SIZE - len(feat)))[:FEATURE_SIZE]
        
        if len(feat) < FEATURE_SIZE:
            feat = np.pad(feat, (0, FEATURE_SIZE - len(feat)))
        else:
            feat = feat[:FEATURE_SIZE]
        
        return torch.tensor(feat, dtype=torch.float32), y, SAMPLE_RATE
    except Exception as e:
        print(f"[ERROR] Audio processing error: {e}")
        raise

def preprocess_text(text):
    if not tokenizer or not bert:
        raise Exception("BERT model not loaded")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        cls = bert(**inputs).last_hidden_state[:, 0, :].numpy().flatten()
    
    cls = np.pad(cls[:FEATURE_SIZE], (0, max(0, FEATURE_SIZE - len(cls))))
    cls = cls[:FEATURE_SIZE]
    
    return torch.tensor(cls, dtype=torch.float32)

import pandas as pd

# ================= DATASET INTEGRATION =================
LABELS_PATH = "BP_Organized_Dataset/labels/labels.csv"
OPENFACE_PATH = "BP_Organized_Dataset/openface_output/"

# Load dataset labels
try:
    labels_df = pd.read_csv(LABELS_PATH)
    print(f"[OK] Dataset labels loaded: {len(labels_df)} participants")
except Exception as e:
    print(f"[WARNING] Labels file not found: {e}")
    labels_df = None

def extract_openface_features(frame):
    """Extract OpenFace-style features from frame for dataset compatibility"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhanced cascade detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    # Initialize OpenFace-style features
    features = {
        'face_detected': len(faces) > 0,
        'confidence': 0.0,
        'success': len(faces) > 0,
        'gaze_angle_x': 0.0,
        'gaze_angle_y': 0.0,
        'pose_Rx': 0.0,
        'pose_Ry': 0.0,
        'pose_Rz': 0.0,
        'AU01_r': 0.0,  # Inner Brow Raiser
        'AU02_r': 0.0,  # Outer Brow Raiser
        'AU04_r': 0.0,  # Brow Lowerer
        'AU05_r': 0.0,  # Upper Lid Raiser
        'AU06_r': 0.0,  # Cheek Raiser
        'AU07_r': 0.0,  # Lid Tightener
        'AU09_r': 0.0,  # Nose Wrinkler
        'AU10_r': 0.0,  # Upper Lip Raiser
        'AU12_r': 0.0,  # Lip Corner Puller
        'AU14_r': 0.0,  # Dimpler
        'AU15_r': 0.0,  # Lip Corner Depressor
        'AU17_r': 0.0,  # Chin Raiser
        'AU20_r': 0.0,  # Lip Stretcher
        'AU23_r': 0.0,  # Lip Tightener
        'AU25_r': 0.0,  # Lips Part
        'AU26_r': 0.0,  # Jaw Drop
        'AU45_r': 0.0,  # Blink
        'eyes_detected': 0,
        'smile_detected': False,
        'face_area_ratio': 0.0,
        'brightness': float(np.mean(gray)),
        'contrast': float(np.std(gray))
    }
    
    if len(faces) > 0:
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        
        # Calculate basic features
        features['face_area_ratio'] = (w * h) / (frame.shape[0] * frame.shape[1])
        features['brightness'] = float(np.mean(face_roi))
        features['contrast'] = float(np.std(face_roi))
        features['confidence'] = min(1.0, features['face_area_ratio'] * 10)
        
        # Eye detection
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10))
        features['eyes_detected'] = len(eyes)
        
        # Smile detection
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.3, minNeighbors=15, minSize=(15, 15))
        features['smile_detected'] = len(smiles) > 0
        
        # Estimate Action Units based on facial features
        if features['smile_detected']:
            features['AU12_r'] = 2.0  # Lip Corner Puller
            features['AU06_r'] = 1.5  # Cheek Raiser
        
        if features['eyes_detected'] >= 2:
            features['AU05_r'] = 1.0  # Upper Lid Raiser
        elif features['eyes_detected'] == 1:
            features['AU07_r'] = 1.0  # Lid Tightener
        
        # Estimate emotional state based on brightness and contrast
        if features['brightness'] < 80:
            features['AU01_r'] = 1.0  # Inner Brow Raiser (concern)
            features['AU04_r'] = 1.5  # Brow Lowerer (sadness)
        
        if features['contrast'] > 40:
            features['AU04_r'] = 2.0  # Brow Lowerer (tension)
            features['AU07_r'] = 1.5  # Lid Tightener (stress)
    
    return features

def analyze_mood_from_features(features, prediction, confidence):
    """Advanced mood analysis using OpenFace-style Action Units"""
    
    if not features['face_detected']:
        return {
            'emotion_label': 'No Face Detected',
            'description': 'Unable to analyze - face not visible in frame',
            'recommendations': ['Position face clearly in camera', 'Ensure adequate lighting', 'Move closer to camera'],
            'bipolar_risk': 'Unknown',
            'symptoms': ['Cannot assess without facial data'],
            'action_units': {},
            'detailed_analysis': 'No facial data available for analysis'
        }
    
    # Extract Action Unit intensities for detailed analysis
    au_data = {
        'AU01': {'intensity': features['AU01_r'], 'present': features['AU01_c'], 'name': 'Inner Brow Raiser'},
        'AU02': {'intensity': features['AU02_r'], 'present': features['AU02_c'], 'name': 'Outer Brow Raiser'},
        'AU04': {'intensity': features['AU04_r'], 'present': features['AU04_c'], 'name': 'Brow Lowerer'},
        'AU05': {'intensity': features['AU05_r'], 'present': features['AU05_c'], 'name': 'Upper Lid Raiser'},
        'AU06': {'intensity': features['AU06_r'], 'present': features['AU06_c'], 'name': 'Cheek Raiser'},
        'AU07': {'intensity': features['AU07_r'], 'present': features['AU07_c'], 'name': 'Lid Tightener'},
        'AU09': {'intensity': features['AU09_r'], 'present': features['AU09_c'], 'name': 'Nose Wrinkler'},
        'AU12': {'intensity': features['AU12_r'], 'present': features['AU12_c'], 'name': 'Lip Corner Puller'},
        'AU15': {'intensity': features['AU15_r'], 'present': features['AU15_c'], 'name': 'Lip Corner Depressor'},
        'AU25': {'intensity': features['AU25_r'], 'present': features['AU25_c'], 'name': 'Lips Part'},
        'AU26': {'intensity': features['AU26_r'], 'present': features['AU26_c'], 'name': 'Jaw Drop'}
    }
    
    # Detailed analysis based on Action Units
    detailed_analysis = []
    active_aus = [au for au, data in au_data.items() if data['present'] == 1]
    
    # Happiness Detection (AU6 + AU12)
    if features['AU12_c'] == 1 and features['AU06_c'] == 1:
        happiness_intensity = (features['AU12_r'] + features['AU06_r']) / 2
        if happiness_intensity > 3.0:
            detailed_analysis.append(f"Strong genuine smile detected (Duchenne smile) - AU12: {features['AU12_r']:.1f}, AU06: {features['AU06_r']:.1f}")
            return {
                'emotion_label': 'Genuine Happiness/Joy',
                'description': f'Authentic positive emotion detected with high intensity ({happiness_intensity:.1f}/5.0). Both lip corner pulling and cheek raising indicate genuine happiness.',
                'recommendations': ['Maintain positive activities', 'Continue current routine', 'Share positivity with others'],
                'bipolar_risk': 'Low',
                'symptoms': ['Genuine smile', 'Elevated cheeks', 'Positive facial expression', 'Good emotional state'],
                'action_units': au_data,
                'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
            }
        else:
            detailed_analysis.append(f"Mild happiness detected - AU12: {features['AU12_r']:.1f}")
            return {
                'emotion_label': 'Mild Happiness/Content',
                'description': f'Moderate positive emotion detected ({happiness_intensity:.1f}/5.0). Lip corner pulling suggests contentment.',
                'recommendations': ['Continue positive activities', 'Maintain social connections'],
                'bipolar_risk': 'Low',
                'symptoms': ['Slight smile', 'Content expression', 'Mild positive mood'],
                'action_units': au_data,
                'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
            }
    
    # Sadness Detection (AU15 + AU04 + AU01)
    elif features['AU15_c'] == 1 or (features['AU04_c'] == 1 and features['AU01_c'] == 1):
        sadness_intensity = features['AU15_r'] + features['AU04_r'] + features['AU01_r']
        
        if features['AU15_c'] == 1:
            detailed_analysis.append(f"Lip corner depression detected - AU15: {features['AU15_r']:.1f}")
        if features['AU04_c'] == 1:
            detailed_analysis.append(f"Brow lowering detected - AU04: {features['AU04_r']:.1f}")
        if features['AU01_c'] == 1:
            detailed_analysis.append(f"Inner brow raising detected - AU01: {features['AU01_r']:.1f}")
        
        if sadness_intensity > 4.0:
            return {
                'emotion_label': 'Severe Sadness/Depression',
                'description': f'Strong indicators of depressive mood state detected ({sadness_intensity:.1f}/15.0). Multiple facial markers suggest significant emotional distress.',
                'recommendations': ['Seek professional counseling immediately', 'Contact mental health services', 'Reach out to support system', 'Consider crisis intervention'],
                'bipolar_risk': 'High',
                'symptoms': ['Severe sad expression', 'Downward mouth', 'Furrowed brow', 'Distressed appearance'],
                'action_units': au_data,
                'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
            }
        elif sadness_intensity > 2.0:
            return {
                'emotion_label': 'Moderate Sadness/Low Mood',
                'description': f'Clear signs of sadness detected ({sadness_intensity:.1f}/15.0). Facial expressions indicate emotional distress.',
                'recommendations': ['Monitor mood patterns', 'Consider professional counseling', 'Practice self-care', 'Engage support system'],
                'bipolar_risk': 'Moderate',
                'symptoms': ['Sad facial expression', 'Downward mouth curve', 'Lowered brows', 'Emotional distress'],
                'action_units': au_data,
                'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
            }
        else:
            return {
                'emotion_label': 'Mild Sadness/Subdued Mood',
                'description': f'Subtle signs of sadness detected ({sadness_intensity:.1f}/15.0). Minor facial indicators of low mood.',
                'recommendations': ['Monitor mood changes', 'Engage in positive activities', 'Consider talking to someone'],
                'bipolar_risk': 'Low-Moderate',
                'symptoms': ['Mild sadness indicators', 'Subdued expression', 'Minor mood concerns'],
                'action_units': au_data,
                'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
            }
    
    # Surprise Detection (AU01 + AU02 + AU05 + AU26)
    elif features['AU01_c'] == 1 and features['AU02_c'] == 1 and (features['AU05_c'] == 1 or features['AU26_c'] == 1):
        surprise_intensity = features['AU01_r'] + features['AU02_r'] + features['AU05_r'] + features['AU26_r']
        detailed_analysis.append(f"Surprise expression detected - AU01: {features['AU01_r']:.1f}, AU02: {features['AU02_r']:.1f}")
        
        if features['AU05_c'] == 1:
            detailed_analysis.append(f"Wide eyes detected - AU05: {features['AU05_r']:.1f}")
        if features['AU26_c'] == 1:
            detailed_analysis.append(f"Jaw drop detected - AU26: {features['AU26_r']:.1f}")
        
        return {
            'emotion_label': 'Surprise/Shock',
            'description': f'Surprise expression detected ({surprise_intensity:.1f}/20.0). Raised eyebrows and wide eyes indicate unexpected reaction.',
            'recommendations': ['Process the surprising information', 'Take time to adjust', 'Seek clarification if needed'],
            'bipolar_risk': 'Low',
            'symptoms': ['Raised eyebrows', 'Wide eyes', 'Open mouth', 'Surprised expression'],
            'action_units': au_data,
            'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
        }
    
    # Anger/Irritation Detection (AU04 + AU07 + AU23)
    elif features['AU04_c'] == 1 and (features['AU07_c'] == 1 or features['AU23_c'] == 1):
        anger_intensity = features['AU04_r'] + features['AU07_r'] + features['AU23_r']
        detailed_analysis.append(f"Anger indicators detected - AU04: {features['AU04_r']:.1f}")
        
        if features['AU07_c'] == 1:
            detailed_analysis.append(f"Eye tightening detected - AU07: {features['AU07_r']:.1f}")
        
        return {
            'emotion_label': 'Anger/Irritation',
            'description': f'Signs of anger or irritation detected ({anger_intensity:.1f}/15.0). Furrowed brow and tightened features suggest negative emotional state.',
            'recommendations': ['Practice anger management techniques', 'Take deep breaths', 'Remove yourself from stressful situation', 'Consider professional help if persistent'],
            'bipolar_risk': 'Moderate-High',
            'symptoms': ['Furrowed brow', 'Tightened eyes', 'Tense facial expression', 'Irritated appearance'],
            'action_units': au_data,
            'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
        }
    
    # Disgust Detection (AU09 + AU15)
    elif features['AU09_c'] == 1:
        detailed_analysis.append(f"Nose wrinkling detected - AU09: {features['AU09_r']:.1f}")
        
        return {
            'emotion_label': 'Disgust/Distaste',
            'description': f'Disgust expression detected ({features["AU09_r"]:.1f}/5.0). Nose wrinkling indicates aversion or distaste.',
            'recommendations': ['Identify source of discomfort', 'Remove yourself from unpleasant situation', 'Practice coping strategies'],
            'bipolar_risk': 'Low-Moderate',
            'symptoms': ['Wrinkled nose', 'Disgusted expression', 'Aversion reaction'],
            'action_units': au_data,
            'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
        }
    
    # Elevated/Manic state (based on ML prediction + high AU activity)
    elif prediction == 1 and features['emotion_intensity'] > 2.0:
        detailed_analysis.append(f"High emotional intensity detected: {features['emotion_intensity']:.1f}")
        detailed_analysis.append(f"Multiple facial action units active: {len(active_aus)}")
        
        return {
            'emotion_label': 'Elevated/Manic State',
            'description': f'High emotional intensity and elevated mood detected. Multiple active facial expressions suggest possible manic episode.',
            'recommendations': ['Seek immediate psychiatric consultation', 'Monitor sleep patterns', 'Avoid major decisions', 'Contact healthcare provider'],
            'bipolar_risk': 'Very High',
            'symptoms': ['Elevated mood', 'High emotional intensity', 'Multiple active expressions', 'Possible mania'],
            'action_units': au_data,
            'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus)}'
        }
    
    # Neutral/Stable state
    else:
        if len(active_aus) == 0:
            detailed_analysis.append("No significant facial action units detected")
        else:
            detailed_analysis.append(f"Minimal facial activity detected")
        
        return {
            'emotion_label': 'Neutral/Stable',
            'description': f'Balanced emotional state with minimal facial expression activity. Confidence: {features["confidence"]:.2f}',
            'recommendations': ['Maintain current stability', 'Continue healthy routine'],
            'bipolar_risk': 'Low',
            'symptoms': ['Neutral expression', 'Stable mood', 'Balanced state'],
            'action_units': au_data,
            'detailed_analysis': '; '.join(detailed_analysis) + f'; Active AUs: {", ".join(active_aus) if active_aus else "None"}'
        }

def extract_facial_features(frame):
    """Enhanced facial feature extraction using OpenFace-style analysis"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    # Initialize comprehensive OpenFace-style features
    features = {
        'face_detected': len(faces) > 0,
        'confidence': 0.0,
        'success': len(faces) > 0,
        
        # Gaze features
        'gaze_angle_x': 0.0,
        'gaze_angle_y': 0.0,
        'gaze_direction_x': 0.0,
        'gaze_direction_y': 0.0,
        
        # Head pose
        'pose_Rx': 0.0,  # Pitch
        'pose_Ry': 0.0,  # Yaw
        'pose_Rz': 0.0,  # Roll
        
        # Facial Action Units (AU) - Intensity
        'AU01_r': 0.0,  # Inner Brow Raiser
        'AU02_r': 0.0,  # Outer Brow Raiser
        'AU04_r': 0.0,  # Brow Lowerer
        'AU05_r': 0.0,  # Upper Lid Raiser
        'AU06_r': 0.0,  # Cheek Raiser
        'AU07_r': 0.0,  # Lid Tightener
        'AU09_r': 0.0,  # Nose Wrinkler
        'AU10_r': 0.0,  # Upper Lip Raiser
        'AU12_r': 0.0,  # Lip Corner Puller (Smile)
        'AU14_r': 0.0,  # Dimpler
        'AU15_r': 0.0,  # Lip Corner Depressor
        'AU17_r': 0.0,  # Chin Raiser
        'AU20_r': 0.0,  # Lip Stretcher
        'AU23_r': 0.0,  # Lip Tightener
        'AU25_r': 0.0,  # Lips Part
        'AU26_r': 0.0,  # Jaw Drop
        'AU28_r': 0.0,  # Lip Suck
        'AU45_r': 0.0,  # Blink
        
        # Facial Action Units (AU) - Presence
        'AU01_c': 0,  # Inner Brow Raiser
        'AU02_c': 0,  # Outer Brow Raiser
        'AU04_c': 0,  # Brow Lowerer
        'AU05_c': 0,  # Upper Lid Raiser
        'AU06_c': 0,  # Cheek Raiser
        'AU07_c': 0,  # Lid Tightener
        'AU09_c': 0,  # Nose Wrinkler
        'AU10_c': 0,  # Upper Lip Raiser
        'AU12_c': 0,  # Lip Corner Puller
        'AU14_c': 0,  # Dimpler
        'AU15_c': 0,  # Lip Corner Depressor
        'AU17_c': 0,  # Chin Raiser
        'AU20_c': 0,  # Lip Stretcher
        'AU23_c': 0,  # Lip Tightener
        'AU25_c': 0,  # Lips Part
        'AU26_c': 0,  # Jaw Drop
        'AU28_c': 0,  # Lip Suck
        'AU45_c': 0,  # Blink
        
        # Basic features
        'eyes_detected': 0,
        'smile_detected': False,
        'face_area_ratio': 0.0,
        'brightness': float(np.mean(gray)),
        'contrast': float(np.std(gray)),
        'eye_openness': 0.0,
        'mouth_curve': 0.0,
        'eyebrow_position': 0.0,
        'face_symmetry': 0.0,
        'emotion_intensity': 0.0
    }
    
    if len(faces) > 0:
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        
        # Basic measurements
        features['face_area_ratio'] = (w * h) / (frame.shape[0] * frame.shape[1])
        features['brightness'] = float(np.mean(face_roi))
        features['contrast'] = float(np.std(face_roi))
        features['confidence'] = min(1.0, features['face_area_ratio'] * 10)
        
        # Eye region analysis (upper 40% of face)
        eye_region = face_roi[:int(h*0.4), :]
        eyes = eye_cascade.detectMultiScale(eye_region, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15))
        features['eyes_detected'] = len(eyes)
        
        # Calculate eye openness and related AUs
        if len(eyes) >= 2:
            eye_areas = [eye[2] * eye[3] for eye in eyes[:2]]
            features['eye_openness'] = float(np.mean(eye_areas)) / (w * h) * 100
            
            # AU05 - Upper Lid Raiser (wide eyes)
            if features['eye_openness'] > 8:
                features['AU05_r'] = min(5.0, features['eye_openness'] / 2)
                features['AU05_c'] = 1
            
            # AU07 - Lid Tightener (squinting)
            elif features['eye_openness'] < 3:
                features['AU07_r'] = 3.0 - features['eye_openness']
                features['AU07_c'] = 1
        
        # Eyebrow region analysis (upper 25% of face)
        eyebrow_region = face_roi[:int(h*0.25), :]
        eyebrow_mean = np.mean(eyebrow_region)
        face_mean = np.mean(face_roi)
        eyebrow_contrast = np.std(eyebrow_region)
        
        features['eyebrow_position'] = (eyebrow_mean - face_mean) / face_mean
        
        # AU01 - Inner Brow Raiser (surprise, concern)
        if eyebrow_contrast > 35 and features['eyebrow_position'] > 0.05:
            features['AU01_r'] = min(5.0, eyebrow_contrast / 10)
            features['AU01_c'] = 1
        
        # AU02 - Outer Brow Raiser (surprise)
        if features['eyebrow_position'] > 0.1:
            features['AU02_r'] = min(5.0, features['eyebrow_position'] * 50)
            features['AU02_c'] = 1
        
        # AU04 - Brow Lowerer (anger, concentration, sadness)
        if features['eyebrow_position'] < -0.05 or eyebrow_contrast > 40:
            features['AU04_r'] = min(5.0, abs(features['eyebrow_position']) * 50 + eyebrow_contrast / 15)
            features['AU04_c'] = 1
        
        # Mouth region analysis (bottom 40% of face)
        mouth_region = face_roi[int(h*0.6):, :]
        
        # Enhanced smile detection
        smiles = smile_cascade.detectMultiScale(mouth_region, scaleFactor=1.8, minNeighbors=22, minSize=(25, 15))
        
        smile_confidence = 0
        if len(smiles) > 0:
            for (sx, sy, sw, sh) in smiles:
                if sy < mouth_region.shape[0] * 0.7:  # Smile in correct position
                    smile_confidence += 1
        
        features['smile_detected'] = smile_confidence > 0
        
        # Detailed mouth analysis
        if mouth_region.size > 0:
            mouth_mean = np.mean(mouth_region)
            mouth_std = np.std(mouth_region)
            
            # Detect mouth curve using edge analysis
            mouth_edges = cv2.Canny(mouth_region, 30, 100)
            h_mouth, w_mouth = mouth_region.shape
            
            # Divide mouth into regions
            left_region = mouth_edges[:, :w_mouth//3]
            center_region = mouth_edges[:, w_mouth//3:2*w_mouth//3]
            right_region = mouth_edges[:, 2*w_mouth//3:]
            
            left_edges = np.sum(left_region)
            center_edges = np.sum(center_region)
            right_edges = np.sum(right_region)
            
            # AU12 - Lip Corner Puller (smile)
            if features['smile_detected'] and (left_edges + right_edges) > center_edges * 1.2:
                features['AU12_r'] = min(5.0, (left_edges + right_edges) / (center_edges + 1) * 2)
                features['AU12_c'] = 1
                features['mouth_curve'] = 1.0
                
                # AU06 - Cheek Raiser (accompanies genuine smile)
                if features['AU12_r'] > 3.0:
                    features['AU06_r'] = features['AU12_r'] - 1.0
                    features['AU06_c'] = 1
            
            # AU15 - Lip Corner Depressor (sadness)
            elif center_edges > (left_edges + right_edges) * 0.8 and mouth_mean < face_mean * 0.95:
                features['AU15_r'] = min(5.0, center_edges / (left_edges + right_edges + 1) * 3)
                features['AU15_c'] = 1
                features['mouth_curve'] = -0.8
            
            # AU25 - Lips Part (surprise, speaking)
            if mouth_std > 25:
                features['AU25_r'] = min(5.0, mouth_std / 8)
                features['AU25_c'] = 1
            
            # AU26 - Jaw Drop (surprise, shock)
            if mouth_region.shape[0] > h * 0.15:  # Large mouth region
                features['AU26_r'] = min(5.0, (mouth_region.shape[0] / h) * 20)
                features['AU26_c'] = 1
        
        # Nose region analysis (middle 30% of face)
        nose_region = face_roi[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.7)]
        if nose_region.size > 0:
            nose_std = np.std(nose_region)
            
            # AU09 - Nose Wrinkler (disgust)
            if nose_std > 30:
                features['AU09_r'] = min(5.0, nose_std / 10)
                features['AU09_c'] = 1
        
        # Face symmetry analysis
        left_half = face_roi[:, :w//2]
        right_half = cv2.flip(face_roi[:, w//2:], 1)
        
        if left_half.shape == right_half.shape:
            symmetry = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            features['face_symmetry'] = float(symmetry)
        
        # Calculate overall emotion intensity
        au_intensities = [features[f'AU{au:02d}_r'] for au in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]]
        features['emotion_intensity'] = sum(au_intensities) / len(au_intensities)
        
        # Estimate head pose (simplified)
        face_center_x = x + w//2
        face_center_y = y + h//2
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        
        # Yaw (left-right head turn)
        features['pose_Ry'] = (face_center_x - frame_center_x) / frame_center_x * 30
        
        # Pitch (up-down head tilt)
        features['pose_Rx'] = (face_center_y - frame_center_y) / frame_center_y * 20
        
        # Estimate gaze direction based on eye position
        if features['eyes_detected'] >= 2:
            features['gaze_direction_x'] = features['pose_Ry'] / 30
            features['gaze_direction_y'] = features['pose_Rx'] / 20
            features['gaze_angle_x'] = features['pose_Ry']
            features['gaze_angle_y'] = features['pose_Rx']
    
    return features

# ================= PAGES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if not MONGODB_AVAILABLE:
            flash("Authentication system unavailable - MongoDB not running", "error")
            return render_template("login.html")
            
        email = request.form.get("email")
        password = request.form.get("password")
        
        if not email or not password:
            flash("Please fill in all fields", "error")
            return render_template("login.html")
        
        try:
            user = users_collection.find_one({"email": email})
            if user and check_password_hash(user["password"], password):
                session["user_id"] = str(user["_id"])
                session["username"] = user["username"]
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid email or password", "error")
        except Exception as e:
            print(f"[ERROR] Login database error: {e}")
            flash("Database error occurred", "error")
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        if not MONGODB_AVAILABLE:
            flash("Authentication system unavailable - MongoDB not running", "error")
            return render_template("sign_up.html")
            
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if not all([username, email, password, confirm_password]):
            flash("Please fill in all fields", "error")
            return render_template("sign_up.html")
        
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template("sign_up.html")
        
        if len(password) < 6:
            flash("Password must be at least 6 characters", "error")
            return render_template("sign_up.html")
        
        try:
            if users_collection.find_one({"email": email}):
                flash("Email already registered", "error")
                return render_template("sign_up.html")
            
            user_data = {
                "username": username,
                "email": email,
                "password": generate_password_hash(password),
                "created_at": datetime.utcnow()
            }
            
            result = users_collection.insert_one(user_data)
            session["user_id"] = str(result.inserted_id)
            session["username"] = username
            flash("Account created successfully!", "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            print(f"[ERROR] Signup database error: {e}")
            flash("Database error occurred", "error")
    
    return render_template("sign_up.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/voice")
def voice_page():
    return render_template("voice_prediction_simple.html")

@app.route("/text")
def text_page():
    return render_template("text_analysis.html")

@app.route("/video")
def video_page():
    return render_template("camera_emotion.html")

@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")

@app.route("/report")
def report():
    return render_template("report.html")

# ================= APIs =================
@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    try:
        if "video_frame" not in request.files:
            return jsonify({"error": "No frame uploaded"}), 400
        
        frame_file = request.files["video_frame"]
        frame_bytes = frame_file.read()
        
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        features = extract_facial_features(frame)
        
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized / 255.0
        feat = frame_normalized.flatten()
        
        if video_scaler:
            expected_features = video_scaler.n_features_in_
            if len(feat) < expected_features:
                feat = np.pad(feat, (0, expected_features - len(feat)))
            else:
                feat = feat[:expected_features]
            feat = video_scaler.transform([feat])[0]
        
        if len(feat) < FEATURE_SIZE:
            feat = np.pad(feat, (0, FEATURE_SIZE - len(feat)))
        else:
            feat = feat[:FEATURE_SIZE]
        
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(feat_tensor)
        
        prediction = int(out.argmax())
        probabilities = out.numpy()[0].tolist()
        confidence = max(probabilities) * 100
        
        mood_analysis = analyze_mood_from_features(features, prediction, confidence)
        
        return jsonify({
            "emotion_label": mood_analysis['emotion_label'],
            "description": mood_analysis['description'],
            "recommendations": mood_analysis['recommendations'],
            "bipolar_risk": mood_analysis['bipolar_risk'],
            "symptoms": mood_analysis['symptoms'],
            "action_units": mood_analysis['action_units'],
            "detailed_analysis": mood_analysis['detailed_analysis'],
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "facial_features": {
                "face_detected": features['face_detected'],
                "eyes_detected": features['eyes_detected'],
                "smile_detected": features['smile_detected'],
                "face_area_ratio": round(features['face_area_ratio'], 4),
                "brightness": round(features['brightness'], 2),
                "contrast": round(features['contrast'], 2),
                "eye_openness": round(features['eye_openness'], 2),
                "emotion_intensity": round(features['emotion_intensity'], 2),
                "head_pose": {
                    "pitch": round(features['pose_Rx'], 2),
                    "yaw": round(features['pose_Ry'], 2),
                    "roll": round(features['pose_Rz'], 2)
                },
                "gaze": {
                    "direction_x": round(features['gaze_direction_x'], 2),
                    "direction_y": round(features['gaze_direction_y'], 2),
                    "angle_x": round(features['gaze_angle_x'], 2),
                    "angle_y": round(features['gaze_angle_y'], 2)
                }
            },
            "probabilities": {
                "normal_euthymic": round(probabilities[0] * 100, 2),
                "manic_elevated": round(probabilities[1] * 100, 2)
            }
        })
        
    except Exception as e:
        print(f"[ERROR] Frame processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    try:
        file = request.files.get("audio") or request.files.get("file")
        if not file:
            return jsonify({"error": "No audio uploaded"}), 400

        audio_bytes = file.read()
        feat, y, sr = preprocess_audio_bytes(audio_bytes)
        feat = feat.unsqueeze(0)
        
        with torch.no_grad():
            out = model(feat)
        
        prediction = int(out.argmax())
        probabilities = out.numpy()[0].tolist()
        confidence = max(probabilities) * 100
        
        energy = float(np.mean(np.abs(y)))
        
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            avg_pitch = float(np.mean(pitch_values)) if pitch_values else 150.0
        except:
            avg_pitch = 150.0
        
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
        
        if prediction == 0:
            emotion = "Calm/Stable"
            mood_state = "Normal/Euthymic"
        else:
            emotion = "Elevated/Manic"
            mood_state = "Manic Episode"
        
        return jsonify({
            "prediction": prediction,
            "emotion": emotion,
            "mood_state": mood_state,
            "confidence": confidence,

            "symptoms": [
                "High vocal energy" if energy > 0.05 else "Low vocal energy",
                "Elevated pitch" if avg_pitch > 180 else "Normal pitch"
            ],

            "recommendations": [
                "Maintain regular sleep schedule",
                "Avoid stimulants",
                "Consult psychiatrist if symptoms persist"
            ],

            "probabilities": {
                "calm": probabilities[0] * 100,
                "manic": probabilities[1] * 100
            }
        })

        
    except Exception as e:
        print(f"[ERROR] Audio prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_text", methods=["POST"])
def predict_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text or len(text) < 10:
            return jsonify({"error": "Text too short (minimum 10 characters)"}), 400
        
        feat = preprocess_text(text)
        feat = feat.unsqueeze(0)
        
        with torch.no_grad():
            output = model(feat)
        
        prediction = int(output.argmax())
        probabilities = output.numpy()[0].tolist()
        confidence = max(probabilities) * 100
        
        if prediction == 0:
            emotion = "Calm/Stable"
            mood_state = "Normal Text Pattern"
        else:
            emotion = "Elevated/Manic"
            mood_state = "Elevated Text Pattern"
        
        return jsonify({
    "prediction": prediction,
    "emotion": emotion,
    "mood_state": mood_state,
    "confidence": confidence,

    "symptoms": [
        "Racing thoughts detected" if prediction == 1 else "Stable thought pattern"
    ],

    "recommendations": [
        "Practice mindfulness",
        "Maintain journaling",
        "Seek professional help if patterns persist"
    ],

    "probabilities": {
        "calm": probabilities[0] * 100,
        "elevated": probabilities[1] * 100
    }
})

        
    except Exception as e:
        print(f"[ERROR] Text prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip().lower()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Crisis detection keywords
        crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'want to die', 'hurt myself',
            'self harm', 'cutting', 'overdose', 'jump off', 'hang myself',
            'no point living', 'better off dead', 'can\'t go on'
        ]
        
        crisis_alert = any(keyword in user_message for keyword in crisis_keywords)
        
        # Generate response based on message content
        response = generate_chatbot_response(user_message, crisis_alert)
        
        return jsonify({
            "response": response,
            "crisis_alert": crisis_alert
        })
        
    except Exception as e:
        print(f"[ERROR] Chatbot error: {e}")
        return jsonify({"error": str(e)}), 500

def generate_chatbot_response(message, is_crisis):
    """Generate contextual responses for bipolar disorder support"""
    
    if is_crisis:
        return """
        I'm very concerned about what you're going through. Your life has value and there are people who want to help.
        <br><br>
        <strong>Please reach out for immediate support:</strong><br>
        • Call 988 (Suicide & Crisis Lifeline)<br>
        • Text HOME to 741741 (Crisis Text Line)<br>
        • Go to your nearest emergency room<br>
        • Call a trusted friend or family member<br>
        <br>
        You don't have to face this alone. Professional help is available 24/7.
        """
    
    # Mood-related responses
    if any(word in message for word in ['feeling great', 'feeling amazing', 'euphoric', 'on top of world', 'invincible']):
        return """
        It's wonderful that you're feeling good! However, if this elevated mood feels unusually intense or you're experiencing:
        <br><br>
        • Decreased need for sleep<br>
        • Racing thoughts<br>
        • Increased energy/activity<br>
        • Impulsive decisions<br>
        <br>
        This might be a hypomanic/manic episode. Consider tracking your mood and speaking with your healthcare provider.
        """
    
    elif any(word in message for word in ['feeling terrible', 'feeling awful', 'depressed', 'hopeless', 'empty']):
        return """
        I'm sorry you're going through a difficult time. Depression can feel overwhelming, but it's treatable. Here are some strategies:
        <br><br>
        • Maintain a regular sleep schedule<br>
        • Try gentle exercise (even a short walk)<br>
        • Connect with supportive people<br>
        • Practice mindfulness or meditation<br>
        • Consider professional therapy<br>
        <br>
        Remember: This feeling is temporary. You've gotten through difficult times before.
        """
    
    elif any(word in message for word in ['feeling low', 'feeling down', 'sad', 'blue']):
        return """
        I understand you're feeling low. It's important to acknowledge these feelings. Some gentle suggestions:
        <br><br>
        • Try to maintain your daily routine<br>
        • Reach out to a friend or family member<br>
        • Engage in a small activity you usually enjoy<br>
        • Practice self-compassion<br>
        <br>
        If these feelings persist or worsen, please consider speaking with a mental health professional.
        """
    
    # Medication-related
    elif any(word in message for word in ['medication', 'meds', 'pills', 'lithium', 'mood stabilizer']):
        return """
        Medication is often an important part of bipolar disorder treatment. Key points:
        <br><br>
        • Take medications as prescribed, even when feeling well<br>
        • Don't stop suddenly without medical supervision<br>
        • Track side effects and discuss with your doctor<br>
        • Be patient - finding the right medication can take time<br>
        <br>
        Always consult your healthcare provider about medication concerns.
        """
    
    # Sleep-related
    elif any(word in message for word in ['sleep', 'insomnia', 'tired', 'exhausted', 'can\'t sleep']):
        return """
        Sleep is crucial for mood stability in bipolar disorder. Sleep tips:
        <br><br>
        • Maintain consistent sleep/wake times<br>
        • Create a relaxing bedtime routine<br>
        • Avoid screens 1 hour before bed<br>
        • Keep bedroom cool and dark<br>
        • Limit caffeine after 2 PM<br>
        <br>
        If sleep problems persist, discuss with your doctor as this can trigger mood episodes.
        """
    
    # Stress and triggers
    elif any(word in message for word in ['stress', 'overwhelmed', 'anxious', 'panic', 'worried']):
        return """
        Stress can trigger bipolar episodes, so managing it is important:
        <br><br>
        • Practice deep breathing exercises<br>
        • Try progressive muscle relaxation<br>
        • Break large tasks into smaller steps<br>
        • Set realistic expectations<br>
        • Use grounding techniques (5-4-3-2-1 method)<br>
        <br>
        Consider learning stress management techniques through therapy or support groups.
        """
    
    # Relationships
    elif any(word in message for word in ['relationship', 'family', 'friends', 'partner', 'lonely']):
        return """
        Relationships can be challenging with bipolar disorder, but they're also vital for support:
        <br><br>
        • Educate loved ones about bipolar disorder<br>
        • Communicate openly about your needs<br>
        • Set healthy boundaries<br>
        • Join support groups to connect with others<br>
        • Consider family therapy<br>
        <br>
        Remember: The right people will support you through your journey.
        """
    
    # General bipolar information
    elif any(word in message for word in ['bipolar', 'manic', 'depression', 'mood swings', 'episodes']):
        return """
        Bipolar disorder involves mood episodes that can include:
        <br><br>
        <strong>Manic/Hypomanic Episodes:</strong><br>
        • Elevated or irritable mood<br>
        • Increased energy/activity<br>
        • Decreased need for sleep<br>
        • Racing thoughts<br>
        <br>
        <strong>Depressive Episodes:</strong><br>
        • Persistent sadness<br>
        • Loss of interest<br>
        • Fatigue<br>
        • Difficulty concentrating<br>
        <br>
        Treatment typically includes medication, therapy, and lifestyle management.
        """
    
    # Default supportive response
    else:
        return """
        Thank you for sharing with me. Living with bipolar disorder can be challenging, but you're not alone.
        <br><br>
        Some general tips for managing bipolar disorder:
        <br>
        • Track your moods daily<br>
        • Maintain regular routines<br>
        • Stay connected with your support system<br>
        • Take medications as prescribed<br>
        • Practice self-care<br>
        <br>
        Is there something specific about bipolar disorder you'd like to know more about?
        """

if __name__ == "__main__":
    print("\n[OK] Server starting...")
    print("[INFO] Open: http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)