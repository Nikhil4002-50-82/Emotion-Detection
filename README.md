# Facial Emotion Recognition using MediaPipe and CK+ Dataset

## Overview

This project performs **facial emotion recognition** using **MediaPipe Face Mesh landmarks** and a **Keras-based neural network**.
It extracts geometric facial features (like mouth aspect ratio, eyebrow raise, eye openness, etc.) and classifies emotions using a lightweight neural model trained on the **CK+ dataset**.

## Project Workflow

### **Step 1: Setup and Import Dependencies**

Install and import all required libraries:

```python
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import matplotlib.pyplot as plt
from google.colab import files
from kaggle.api.kaggle_api_extended import KaggleApi
```

### **Step 2: Kaggle API Setup and Dataset Download**

Upload your `kaggle.json` file directly during runtime:

```python
uploaded = files.upload()
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "wb") as f:
    f.write(uploaded["kaggle.json"])
os.chmod("/root/.kaggle/kaggle.json", 600)
```

Then download and extract the **CK+ dataset**:

```python
dataset_name = "shuvoalok/ck-dataset"
extract_path = "/content/ckplus_data"

api = KaggleApi()
api.authenticate()
api.dataset_download_files(dataset_name, path=extract_path, unzip=False)

# Extract zip
import zipfile
zip_files = [f for f in os.listdir(extract_path) if f.endswith(".zip")]
zip_path = os.path.join(extract_path, zip_files[0])
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
os.remove(zip_path)
```

**Dataset structure:**

```
ckplus_data/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── sadness/
└── surprise/
```

### **Step 3: MediaPipe Landmark Extraction**

Each image is processed with **MediaPipe FaceMesh** to extract 3D face landmarks.

```python
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:
    results = face_mesh.process(rgb_image)
```

### **Step 4: Feature Extraction**

From landmarks, the following features are calculated:

* Mouth width, height, and aspect ratio
* Eye openness and aspect ratio
* Eyebrow raise and slant
* Jaw drop
* Nose–mouth distance
* Mouth corner difference
* Face scaling normalization

Example:

```python
def extract_features(landmarks):
    # Compute distances between key landmark points
    # Normalize using face width for scale invariance
    features = np.array([...], dtype=np.float32)
    return features
```

A CSV (`features.csv`) is created containing extracted features and emotion labels.

### **Step 5: Model Training**

A small **Keras Neural Network** is trained on extracted features:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

Training achieved an accuracy of **~84.26%** on the validation set.

### **Step 6: Saving the Model**

```python
model.save("emotion_model.h5")
```

### **Step 7: Emotion Prediction on New Image**

Upload a new image and extract its MediaPipe landmarks.
Then use the trained model to predict emotion:

```python
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

features = extract_features(landmarks)
prediction = model.predict(features.reshape(1, -1))
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
```

### **Step 8: Visualization**

Instead of annotated facial landmarks, the output can visualize the image with an **emoji or label overlay** corresponding to the predicted emotion.

## Example Output

* **Input:** Image of a person
* **Extracted:** 478 landmarks via MediaPipe
* **Predicted Emotion:** “Happy”
* **Model Accuracy:** ~84.26%

## Future Enhancements

* Use CNN-based hybrid model combining pixel and landmark features.
* Real-time webcam emotion detection using OpenCV.
* Integrate attention-based feature weighting for improved accuracy.

## Requirements

* Python 3.8+
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy, Pandas, Matplotlib
* Kaggle API credentials

---
