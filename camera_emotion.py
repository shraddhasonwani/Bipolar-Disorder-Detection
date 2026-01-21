import cv2
import numpy as np

# ---------------- Robust FER Import ----------------
try:
    from fer import FER
    DetectorClass = FER
    print("Using fer.FER")
except ImportError:
    try:
        from fer.fer import FER
        DetectorClass = FER
        print("Using fer.fer.FER")
    except ImportError:
        print("FER not found, using dummy detector")

        class DummyDetector:
            def detect_emotions(self, frame):
                return []

        DetectorClass = DummyDetector

# Initialize detector
emotion_detector = DetectorClass(mtcnn=False)

def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("❌ Camera not accessible")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # ✅ Convert BGR → RGB (VERY IMPORTANT)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = emotion_detector.detect_emotions(rgb_frame)

        if results:
            for res in results:
                (x, y, w, h) = res["box"]
                emotions = res["emotions"]

                # Sort emotions by confidence
                sorted_emotions = sorted(
                    emotions.items(),
                    key=lambda item: item[1],
                    reverse=True
                )

                top_emotion, top_conf = sorted_emotions[0]
                second_emotion, second_conf = sorted_emotions[1]

                # 🔁 Neutral override logic
                if top_emotion == "neutral" and second_conf > 0.25:
                    display_emotion = second_emotion
                    confidence = second_conf
                else:
                    display_emotion = top_emotion
                    confidence = top_conf

                # 🎨 Color coding
                color = (0, 255, 0)
                if display_emotion == "sad":
                    color = (255, 0, 0)      # Blue
                elif display_emotion in ["happy", "surprise"]:
                    color = (0, 165, 255)   # Orange
                elif display_emotion == "angry":
                    color = (0, 0, 255)     # Red
                elif display_emotion == "fear":
                    color = (128, 0, 128)   # Purple

                # Draw box + label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{display_emotion.upper()} ({confidence:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    camera.release()
