"""
Heart Disease Detection Project - Production Version
Includes:
- Training
- Evaluation
- Model Saving
- Prediction
- FastAPI API
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from fastapi import FastAPI
from pydantic import BaseModel


# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "heart_disease_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")


# =========================
# LOAD DATASET
# =========================

def load_data():
    print("Loading dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}\n"
            f"Make sure 'heart_disease_dataset.csv' is in the same folder."
        )

    df = pd.read_csv(DATASET_PATH)

    if "heart_disease" not in df.columns:
        raise ValueError(
            "Column 'heart_disease' not found in dataset.\n"
            "Check your target column name."
        )

    X = df.drop("heart_disease", axis=1)
    y = df["heart_disease"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# TRAIN MODELS
# =========================

def train_models():

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    best_model = None
    best_score = 0

    print("\nTraining Models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc:.4f}")
        print("-" * 40)

        if roc > best_score:
            best_score = roc
            best_model = model

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved as {MODEL_PATH}")


# =========================
# PREDICTION FUNCTION
# =========================

def predict(data):

    if not os.path.exists(MODEL_PATH):
        train_models()

    model = joblib.load(MODEL_PATH)

    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return prediction, probability


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="Heart Disease Detection API")


class PatientData(BaseModel):
    age: float
    sex: float
    chest_pain_type: float
    resting_bp: float
    cholesterol: float
    fasting_blood_sugar: float
    resting_ecg: float
    max_heart_rate: float
    exercise_induced_angina: float
    st_depression: float
    st_slope: float
    num_major_vessels: float
    thalassemia: float


@app.get("/")
def home():
    return {"message": "Heart Disease Detection API Running 🚀"}


@app.post("/predict")
def predict_api(data: PatientData):

    values = list(data.dict().values())

    prediction, probability = predict(values)

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
