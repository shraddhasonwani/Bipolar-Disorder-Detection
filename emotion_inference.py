def infer_emotion(features):
    energy = features[-2]
    pitch = features[-1]

    if energy < 0.015 and pitch < 120:
        return "Sad"
    elif energy > 0.03 and pitch > 180:
        return "Happy"
    elif pitch > 220:
        return "Angry"
    else:
        return "Neutral"
