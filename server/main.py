from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

emotion_emojis = {
    "surprise": "😮",
    "happy": "😄",
    "disgust": "🤢",
    "anger": "😠",
    "sadness": "😢",
    "fear": "😨",
    "contempt": "😒"
}

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def extract_features(landmarks):
    landmarks = np.array(landmarks)
    mouth_left = landmarks[61, :2]
    mouth_right = landmarks[291, :2]
    mouth_top = landmarks[13, :2]
    mouth_bottom = landmarks[14, :2]
    mouth_width = euclidean(mouth_left, mouth_right)
    mouth_height = euclidean(mouth_top, mouth_bottom)
    mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
    
    left_eye_top = landmarks[159, :2]
    left_eye_bottom = landmarks[145, :2]
    right_eye_top = landmarks[386, :2]
    right_eye_bottom = landmarks[374, :2]
    left_eye_height = euclidean(left_eye_top, left_eye_bottom)
    right_eye_height = euclidean(right_eye_top, right_eye_bottom)
    eye_openness = (left_eye_height + right_eye_height) / 2
    
    left_eye_outer = landmarks[33, :2]
    left_eye_inner = landmarks[133, :2]
    right_eye_inner = landmarks[362, :2]
    right_eye_outer = landmarks[263, :2]
    left_eye_width = euclidean(left_eye_outer, left_eye_inner)
    right_eye_width = euclidean(right_eye_inner, right_eye_outer)
    eye_aspect_ratio = eye_openness / ((left_eye_width + right_eye_width)/2 + 1e-6)
    
    left_eyebrow_outer = landmarks[70, :2]
    left_eyebrow_inner = landmarks[55, :2]
    right_eyebrow_inner = landmarks[285, :2]
    right_eyebrow_outer = landmarks[300, :2]
    eyebrow_raise_left = euclidean(left_eyebrow_outer, left_eye_top)
    eyebrow_raise_right = euclidean(right_eyebrow_outer, right_eye_top)
    eyebrow_slant_left = euclidean(left_eyebrow_outer, left_eyebrow_inner)
    eyebrow_slant_right = euclidean(right_eyebrow_inner, right_eyebrow_outer)
    
    nose_tip = landmarks[1, :2]
    chin = landmarks[152, :2]
    nose_base = landmarks[168, :2]
    jaw_drop = euclidean(chin, nose_base)
    nose_mouth_dist = euclidean(nose_tip, (mouth_top + mouth_bottom)/2)
    
    mouth_corner_diff = abs(mouth_left[1] - mouth_right[1])
    
    features = np.array([
        mouth_width, mouth_height, mouth_aspect_ratio,
        left_eye_height, right_eye_height, eye_openness, eye_aspect_ratio,
        eyebrow_raise_left, eyebrow_raise_right,
        eyebrow_slant_left, eyebrow_slant_right,
        jaw_drop, nose_mouth_dist, mouth_corner_diff
    ], dtype=np.float32)
    
    face_scale = euclidean(landmarks[33, :2], landmarks[263, :2])
    features /= (face_scale + 1e-6)
    return features

app = FastAPI(title="Real-time Emotion Recognition API")

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return JSONResponse({"emotion": None, "emoji": None, "confidence": None, "message": "No face detected"})

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark], dtype=np.float32)
        features = extract_features(landmarks).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)
        predicted_index = np.argmax(pred)
        predicted_label = encoder.inverse_transform([predicted_index])[0]
        confidence = float(np.max(pred))

        emoji_icon = emotion_emojis.get(predicted_label, "😐")

        return JSONResponse({
            "emotion": predicted_label,
            "emoji": emoji_icon,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
