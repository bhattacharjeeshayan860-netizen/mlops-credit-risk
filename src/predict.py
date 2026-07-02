"""Prediction utilities shared by the FastAPI service.

This module will load the trained model and preprocessing artifacts from
`artifacts/`, apply inference-time preprocessing, impute nullable fields using
training medians, and return prediction probabilities.
"""
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT =Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH= ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
DEFAULT_THRESHOLD = 0.5

def load_model():
    """Load the trained model and preprocessing artifacts from artifacts."""
    model= joblib.load(MODEL_PATH)
    if not model:
        raise ValueError("Model could not be loaded.")
    return model

def load_preprocessor():
    """Load the preprocessing artifacts from artifacts."""
    preprocessor= joblib.load(PREPROCESSOR_PATH)
    if not preprocessor:
        raise ValueError("Preprocessor could not be loaded.")
    return preprocessor
def prepare_input(input_data: dict)-> pd.DataFrame:
    """prepare input data for prediction."""
    df=pd.DataFrame([input_data])
    if df.empty:
        raise ValueError("Input data is empty.")
    return df



def make_prediction(input_data: dict) -> dict:
    """Return credit-risk prediction output for one request payload."""
    model= load_model()
    preprocessor= load_preprocessor()
    df= prepare_input(input_data)

    X= preprocessor.transform(df)
    default_probability= model.predict_proba(X)[:, 1]
    default_probability= float(default_probability[0])
    
    if default_probability > DEFAULT_THRESHOLD:
        prediction = 1
        risk_label= "High Risk"
    else:
        prediction = 0
        risk_label= "Low Risk"
    return {"prediction": int(prediction), "default_probability": default_probability, "risk_label": risk_label} 
