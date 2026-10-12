import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Module-level configuration and cold-start artifact loading
MODEL_DIR = Path("model")

# Load artifacts once at import
_model = joblib.load(MODEL_DIR / "model.pkl")
_scaler = joblib.load(MODEL_DIR / "scaler.pkl")
_encoder = joblib.load(MODEL_DIR / "encoder.pkl")

with open(MODEL_DIR / "feature_order.json", "r") as f:
    _feature_order = json.load(f)

# Constants matching training (src/train.py)
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
BINARY_COLS = ['SeniorCitizen']
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

def predict(record: dict) -> dict:
    """
    Performs inference for a single customer record.

    Args:
        record: Single customer dict with raw input fields.

    Returns:
        Dict containing churn probability and predicted class.
    """
    # Build one-row DataFrame from record
    df = pd.DataFrame([record])

    # 1. Scale numeric columns
    scaled_num = _scaler.transform(df[NUMERIC_COLS])

    # 2. Pass through binary columns
    binary = df[BINARY_COLS].values

    # 3. Encode categorical columns
    encoded_cat = _encoder.transform(df[CATEGORICAL_COLS])

    # Stack components into a single 2D array
    X = np.hstack([scaled_num, binary, encoded_cat])

    # Sanity check: the feature order must match the model's expected input
    # Since we constructed X in the same order as _feature_order,
    # the shapes should already match.
    if X.shape[1] != len(_feature_order):
        raise ValueError(f"Feature mismatch: model expects {len(_feature_order)} features, got {X.shape[1]}")

    # Get probability for class 1 (Churn)
    proba = _model.predict_proba(X)[:, 1][0]

    return {
        "churn_probability": float(proba),
        "churn_class": int(proba >= 0.5)
    }

def lambda_handler(event, context):
    """
    AWS Lambda entry point for churn inference.
    """
    try:
        # Normalize event to a record dict
        if isinstance(event, str):
            record = json.loads(event)
        elif isinstance(event, dict) and "body" in event:
            body = event["body"]
            record = json.loads(body) if isinstance(body, str) else body
        else:
            record = event

        # Run prediction
        prediction = predict(record)

        return {
            "statusCode": 200,
            "body": json.dumps(prediction)
        }

    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    # Sample realistic Telco customer record
    sample = {
        "tenure": 12,
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
        "SeniorCitizen": 0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }
    print(json.dumps(lambda_handler(sample, None), indent=2))
