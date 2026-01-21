import requests
import os

BASE_URL = "http://127.0.0.1:5000"
AUDIO_PATH = "BP_Organized_Dataset/audio/sample.wav"
VIDEO_PATH = "BP_Organized_Dataset/video/sample.mp4"
TEXT_INPUT = "I am feeling happy today!"

def predict_audio(audio_path):
    with open(audio_path, "rb") as f:
        response = requests.post(f"{BASE_URL}/predict_audio", files={"file": f})
    print("\n🎵 Audio Prediction:", response.json())

def predict_text(text):
    response = requests.post(f"{BASE_URL}/predict_text", json={"text": text})
    print("\n📝 Text Prediction:", response.json())

def predict_video(video_path):
    with open(video_path, "rb") as f:
        response = requests.post(f"{BASE_URL}/predict_video", files={"file": f})
    print("\n🎬 Video Prediction:", response.json())

def predict_multimodal(audio_path=None, video_path=None, text=None):
    files_payload = {}
    if audio_path: files_payload["audio"] = open(audio_path, "rb")
    if video_path: files_payload["video"] = open(video_path, "rb")
    json_payload = {"text": text} if text else {}
    response = requests.post(f"{BASE_URL}/predict_multimodal", files=files_payload if files_payload else None, json=json_payload)
    for f in files_payload.values(): f.close()
    print("\n🤖 Multimodal Prediction:", response.json())

if __name__ == "__main__":
    choice = input("Select type: 1-Audio, 2-Text, 3-Video, 4-Multimodal: ").strip()
    if choice == "1": predict_audio(AUDIO_PATH)
    elif choice == "2": predict_text(TEXT_INPUT)
    elif choice == "3": predict_video(VIDEO_PATH)
    elif choice == "4": predict_multimodal(AUDIO_PATH, VIDEO_PATH, TEXT_INPUT)
    else: print("❌ Invalid choice")
