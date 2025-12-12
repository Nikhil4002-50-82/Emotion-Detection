# Facial Emotion Recognition using MediaPipe and CK+ Dataset

## Overview

This project performs **facial emotion recognition** using **MediaPipe Face Mesh landmarks** and a **Keras-based neural network**.
It extracts geometric facial features (like mouth aspect ratio, eyebrow raise, eye openness, etc.) and classifies emotions using a lightweight neural model trained on the **CK+ dataset**.

## Project Workflow

### **Step 1: Setup and Import Dependencies**
### **Step 2: Kaggle API Setup and Dataset Download**
### **Step 3: MediaPipe Landmark Extraction**
### **Step 4: Feature Extraction**
### **Step 5: Model Training**
### **Step 6: Saving the Model**
### **Step 7: Emotion Prediction on New Image**
### **Step 8: Visualization**

## Example Output

* **Input:** Image of a person
* **Extracted:** 478 landmarks via MediaPipe
* **Predicted Emotion:** “Happy”
* **Model Accuracy:** ~88.54%

## requirements.txt

* fastapi==0.111.1
* uvicorn==0.25.0
* numpy==1.26.0
* opencv-python==4.8.1.78
* mediapipe==0.11.9
* joblib==1.3.2
* tensorflow==2.13.1
* scikit-learn==1.8.0
* python-multipart==0.0.6

## **FastAPI Server Setup and Running**

### **Step 1: Create and Activate Virtual Environment (Git Bash)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/Scripts/activate
```

### **Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3: Run the FastAPI Server**

```bash
uvicorn main:app --reload
```

* The server will start at: `http://127.0.0.1:8000`

### **Step 4: Test the API**

1. Open your browser and go to:

   ```
   http://127.0.0.1:8000/docs
   ```
2. Find the `POST /predict` endpoint.
3. Click **“Try it out”**, upload an image file, and execute the request.
4. You will get a JSON response with the predicted emotion and confidence.

---