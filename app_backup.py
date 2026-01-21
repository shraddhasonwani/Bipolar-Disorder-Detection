import os
import io
import torch
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
app.secret_key = 'your-secret-key-here'
CORS(app)

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

print(f"✓ Model loaded from {MODEL_PATH}")
print(f"✓ Audio scaler: {'Loaded' if audio_scaler else 'Not found'}")
print(f"✓ Video scaler: {'Loaded' if video_scaler else 'Not found'}")

# ================= TEXT MODEL =================
try:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    bert.eval()
    print("✓ BERT model loaded")
except Exception as e:
    print(f"⚠ BERT model failed to load: {e}")
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
        print(f"❌ Audio processing error: {e}")
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

# ================= PAGES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("sign_up.html")

@app.route("/logout")
def logout():
    return render_template("logout.html")

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

@app.route("/report")
def report():
    return render_template("report.html")

# ================= APIs =================
@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    try:
        file = request.files.get("video_frame")
        if not file:
            return jsonify({"error": "No frame uploaded"}), 400
        
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Advanced facial analysis with better detection parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced cascade classifiers with better parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # More sensitive face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        # Facial expression indicators
        expression_features = {
            'face_detected': False,
            'eyes_detected': 0,
            'smile_detected': False,
            'face_area_ratio': 0,
            'brightness': 0,
            'contrast': 0
        }
        
        if len(faces) > 0:
            expression_features['face_detected'] = True
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_roi = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # Calculate face area ratio (larger faces might indicate closer/more engaged)
            expression_features['face_area_ratio'] = (w * h) / (frame.shape[0] * frame.shape[1])
            
            # Detect eyes in face region with better parameters
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10))
            expression_features['eyes_detected'] = len(eyes)
            
            # Detect smile with more sensitive parameters
            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.3, minNeighbors=15, minSize=(15, 15))
            expression_features['smile_detected'] = len(smiles) > 0
            
            # Calculate brightness and contrast
            expression_features['brightness'] = np.mean(face_gray)
            expression_features['contrast'] = np.std(face_gray)
            
            face_resized = cv2.resize(face_roi, (64, 64))
        else:
            # No face detected, use full frame
            face_resized = cv2.resize(frame, (64, 64))
        
        # Enhanced feature extraction with expression analysis
        frame_normalized = face_resized.astype(np.float32) / 255.0
        
        # 1. Raw pixel features
        pixel_feat = frame_normalized.flatten()
        
        # 2. Color analysis for mood indicators
        hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        color_feat = np.concatenate([hist_h, hist_s, hist_v]) / 255.0
        
        # 3. Texture features for expression analysis
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        # Local Binary Pattern approximation
        lbp_feat = []
        for i in range(1, gray_face.shape[0]-1, 4):
            for j in range(1, gray_face.shape[1]-1, 4):
                center = gray_face[i, j]
                pattern = 0
                pattern += (gray_face[i-1, j-1] >= center) << 7
                pattern += (gray_face[i-1, j] >= center) << 6
                pattern += (gray_face[i-1, j+1] >= center) << 5
                pattern += (gray_face[i, j+1] >= center) << 4
                pattern += (gray_face[i+1, j+1] >= center) << 3
                pattern += (gray_face[i+1, j] >= center) << 2
                pattern += (gray_face[i+1, j-1] >= center) << 1
                pattern += (gray_face[i, j-1] >= center) << 0
                lbp_feat.append(pattern / 255.0)
        
        # 4. Expression-specific features
        expression_vector = [
            float(expression_features['face_detected']),
            expression_features['eyes_detected'] / 10.0,  # Normalize
            float(expression_features['smile_detected']),
            expression_features['face_area_ratio'] * 100,  # Scale up
            expression_features['brightness'] / 255.0,
            expression_features['contrast'] / 100.0
        ]
        
        # Combine all features
        combined_feat = np.concatenate([
            pixel_feat, 
            color_feat, 
            np.array(lbp_feat[:64]),  # Limit LBP features
            np.array(expression_vector)
        ])
        
        # Handle video scaler feature size mismatch
        if video_scaler:
            expected_features = video_scaler.n_features_in_
            if len(combined_feat) < expected_features:
                feat = np.pad(combined_feat, (0, expected_features - len(combined_feat)))
            else:
                feat = combined_feat[:expected_features]
            feat = video_scaler.transform([feat])[0]
        else:
            feat = combined_feat
        
        # Final padding/trimming to FEATURE_SIZE for model
        if len(feat) < FEATURE_SIZE:
            feat = np.pad(feat, (0, FEATURE_SIZE - len(feat)))
        else:
            feat = feat[:FEATURE_SIZE]
        
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = model(feat_tensor)
        
        prediction = int(output.argmax())
        probabilities = output.numpy()[0].tolist()
        confidence = max(probabilities) * 100
        
        # Advanced mood analysis based on facial features and model prediction
        mood_analysis = analyze_mood_from_features(expression_features, prediction, confidence)
        
        return jsonify({
            "emotion_label": mood_analysis['emotion_label'],
            "mood_description": mood_analysis['description'],
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "recommendations": mood_analysis.get('recommendations', []),
            "facial_features": {
                "face_detected": expression_features['face_detected'],
                "eyes_detected": expression_features['eyes_detected'],
                "smile_detected": expression_features['smile_detected'],
                "engagement_level": "High" if expression_features['face_area_ratio'] > 0.1 else "Low"
            },
            "probabilities": {
                "calm": round(probabilities[0] * 100, 2),
                "elevated": round(probabilities[1] * 100, 2)
            }
        })
        
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    try:
        if "video_frame" not in request.files:
            return jsonify({"error": "No frame uploaded"}), 400
        
        frame_file = request.files["video_frame"]
        frame_bytes = frame_file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Extract facial features using OpenCV
        features = extract_facial_features(frame)
        
        # Preprocess for model
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized / 255.0
        feat = frame_normalized.flatten()
        
        # Handle scaler compatibility
        if video_scaler:
            expected_features = video_scaler.n_features_in_
            if len(feat) < expected_features:
                feat = np.pad(feat, (0, expected_features - len(feat)))
            else:
                feat = feat[:expected_features]
            feat = video_scaler.transform([feat])[0]
        
        # Final padding for model
        if len(feat) < FEATURE_SIZE:
            feat = np.pad(feat, (0, FEATURE_SIZE - len(feat)))
        else:
            feat = feat[:FEATURE_SIZE]
        
        # Predict using model
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(feat_tensor)
        
        prediction = int(out.argmax())
        probabilities = out.numpy()[0].tolist()
        confidence = max(probabilities) * 100
        
        # Analyze mood based on facial features + model prediction
        mood_analysis = analyze_mood_from_features(features, prediction, confidence)
        
        return jsonify({
            "emotion_label": mood_analysis['emotion_label'],
            "description": mood_analysis['description'],
            "recommendations": mood_analysis['recommendations'],
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "facial_features": features,
            "probabilities": {
                "calm": round(probabilities[0] * 100, 2),
                "elevated": round(probabilities[1] * 100, 2)
            }
        })
        
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        return jsonify({"error": str(e)}), 500

def extract_facial_features(frame):
    """Extract facial features using OpenCV"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    features = {
        'face_detected': len(faces) > 0,
        'eyes_detected': 0,
        'smile_detected': False,
        'face_area_ratio': 0,
        'brightness': float(np.mean(gray)),
        'contrast': float(np.std(gray))
    }
    
    if len(faces) > 0:
        # Get largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        features['face_area_ratio'] = (w * h) / (frame.shape[0] * frame.shape[1])
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in face region
        eyes = eye_cascade.detectMultiScale(face_roi)
        features['eyes_detected'] = len(eyes)
        
        # Detect smile in face region
        smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)
        features['smile_detected'] = len(smiles) > 0
    
    return features
    """Analyze mood based on facial features with direct expression analysis"""
    
    # Direct facial expression analysis - prioritize actual facial features
    if not features['face_detected']:
        return {
            'emotion_label': 'No Face Detected',
            'description': 'Please position your face clearly in the camera',
            'recommendations': ['Adjust camera position', 'Ensure good lighting', 'Move closer to camera']
        }
    
    # Smile detection takes highest priority
    if features['smile_detected']:
        if features['eyes_detected'] >= 2:
            return {
                'emotion_label': 'Happy/Joyful',
                'description': 'Genuine happiness detected - smile with engaged eyes',
                'recommendations': ['Continue positive activities', 'Share your joy with others', 'Maintain healthy routines']
            }
        else:
            return {
                'emotion_label': 'Smiling',
                'description': 'Positive expression with smile detected',
                'recommendations': ['Keep up the positive mood', 'Engage in social activities', 'Practice gratitude']
            }
    
    # Eye engagement analysis
    if features['eyes_detected'] >= 2:
        # High engagement with good lighting
        if features['brightness'] > 120 and features['contrast'] > 30:
            if features['face_area_ratio'] > 0.12:  # Close to camera, engaged
                return {
                    'emotion_label': 'Alert/Engaged',
                    'description': 'High attention and engagement detected',
                    'recommendations': ['Channel this energy productively', 'Take breaks to avoid burnout', 'Stay hydrated']
                }
            else:
                return {
                    'emotion_label': 'Calm/Focused',
                    'description': 'Relaxed and attentive state',
                    'recommendations': ['Maintain this balanced state', 'Continue current activities', 'Practice mindfulness']
                }
        
        # Normal lighting, good eye contact
        elif features['brightness'] > 80:
            if confidence > 70 and prediction == 1:  # Elevated with high confidence
                return {
                    'emotion_label': 'Intense/Concentrated',
                    'description': 'High focus or possible stress detected',
                    'recommendations': ['Take regular breaks', 'Practice deep breathing', 'Consider stress management techniques']
                }
            else:
                return {
                    'emotion_label': 'Neutral/Attentive',
                    'description': 'Balanced emotional state with good attention',
                    'recommendations': ['Maintain current routine', 'Stay engaged with activities', 'Monitor mood changes']
                }
        
        # Low lighting but eyes visible
        else:
            return {
                'emotion_label': 'Subdued/Tired',
                'description': 'Low energy or fatigue indicators',
                'recommendations': ['Get adequate rest', 'Improve lighting conditions', 'Consider energy-boosting activities']
            }
    
    # Single eye or poor eye detection
    elif features['eyes_detected'] == 1:
        if features['brightness'] < 60:
            return {
                'emotion_label': 'Withdrawn/Sad',
                'description': 'Possible withdrawal or low mood indicators',
                'recommendations': ['Seek social support', 'Consider professional help', 'Engage in mood-lifting activities']
            }
        else:
            return {
                'emotion_label': 'Contemplative',
                'description': 'Thoughtful or introspective state',
                'recommendations': ['Take time for reflection', 'Journal your thoughts', 'Practice meditation']
            }
    
    # No eyes detected but face present
    else:
        if features['brightness'] < 50:
            return {
                'emotion_label': 'Very Low Energy',
                'description': 'Extremely low engagement or possible fatigue',
                'recommendations': ['Prioritize rest and sleep', 'Consult healthcare provider', 'Improve environment lighting']
            }
        elif features['face_area_ratio'] < 0.05:  # Very small face
            return {
                'emotion_label': 'Distant/Disengaged',
                'description': 'Low engagement - please move closer to camera',
                'recommendations': ['Adjust camera distance', 'Improve positioning', 'Ensure clear visibility']
            }
        else:
            return {
                'emotion_label': 'Expressionless',
                'description': 'Minimal facial expression detected',
                'recommendations': ['Try gentle facial exercises', 'Engage in stimulating activities', 'Monitor emotional state']
            }
    
    # Fallback
    return {
        'emotion_label': 'Neutral',
        'description': 'Unable to determine specific expression',
        'recommendations': ['Ensure good lighting', 'Position face clearly', 'Try different expressions']
    }

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
            "probabilities": {
                "calm": probabilities[0] * 100,
                "manic": probabilities[1] * 100
            },
            "audio_features": {
                "energy": energy,
                "avg_pitch": avg_pitch,
                "speech_rate": zcr
            },
            "duration": len(y) / sr
        })
        
    except Exception as e:
        print(f"❌ Audio prediction error: {e}")
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
            "probabilities": {
                "calm": probabilities[0] * 100,
                "elevated": probabilities[1] * 100
            }
        })
        
    except Exception as e:
        print(f"❌ Text prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n✅ Server starting...")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)