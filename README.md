# Heart Disease Detection Using Machine Learning

## About
This project explores machine learning classification models to predict the likelihood of heart disease from patient health features. It compares multiple models and provides a prediction API using FastAPI.

## Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- FastAPI
- Pydantic

## Project Features
- Dataset loading and validation
- Training and evaluation of classification models
- Model saving using Joblib
- Prediction function
- FastAPI endpoints for API access

## Project Files
- `heart_disease_project.py` — main application code
- `heart_disease_dataset.csv` — dataset used for model training
- `README.md` — project documentation
- `requirements.txt` — project dependencies

## Setup and Usage

### 1. Clone the repository
```bash
git clone YOUR_REPOSITORY_URL
cd heart-disease-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models
```bash
python heart_disease_project.py
```

### 4. Start the API
```bash
uvicorn heart_disease_project:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API using Swagger UI.

## Disclaimer
This is an educational machine learning project. Its predictions are not medically validated and must not be used for diagnosis or treatment decisions.
