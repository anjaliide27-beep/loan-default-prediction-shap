"""
prediction.py
-------------
Inference, prediction, input validation, and risk categorization module
for Explainable Loan Default Prediction.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from src.data_processing import (
    format_user_input,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

MODELS_DIR = "models"
DEFAULT_MODEL_NAME = "Random Forest"

# Configurable risk thresholds
DEFAULT_LOW_THRESHOLD = 0.20   # < 20% = Low Risk
DEFAULT_MED_THRESHOLD = 0.40   # 20% - 40% = Medium Risk, >= 40% = High Risk


def load_model_artifacts(models_dir: str = MODELS_DIR):
    """
    Loads saved models, preprocessor, and metadata from disk.
    Raises FileNotFoundError with clear instructions if models have not yet been trained.
    """
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    models_path = os.path.join(models_dir, "models.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    metadata_path = os.path.join(models_dir, "metadata.json")

    missing_files = [p for p in [preprocessor_path, models_path, metrics_path, metadata_path] if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            f"Required model files missing: {missing_files}. "
            "Please run 'python -m src.train_model' to train and save the models."
        )

    preprocessor = joblib.load(preprocessor_path)
    models = joblib.load(models_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return preprocessor, models, metrics, metadata


def validate_user_inputs(inputs: dict) -> list[str]:
    """
    Validates user form inputs and returns a list of error strings.
    If valid, returns an empty list.
    """
    errors = []

    # Helper to parse floats safely
    def to_float(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val).replace('%', '').strip()
            return float(s)
        except Exception:
            return None

    # Numerical checks
    loan_amnt = to_float(inputs.get('loan_amnt'))
    if loan_amnt is None or loan_amnt <= 0:
        errors.append("Loan Amount must be a positive number greater than $0.")
    elif loan_amnt > 100000:
        errors.append("Loan Amount exceeds the supported maximum of $100,000.")

    annual_inc = to_float(inputs.get('annual_inc'))
    if annual_inc is None or annual_inc <= 0:
        errors.append("Annual Income must be a positive number greater than $0.")
    elif annual_inc < 1000:
        errors.append("Annual Income is unrealistically low (minimum $1,000 required).")

    int_rate = to_float(inputs.get('int_rate'))
    if int_rate is None or int_rate <= 0:
        errors.append("Interest Rate must be greater than 0%.")
    elif int_rate > 50:
        errors.append("Interest Rate cannot exceed 50%.")

    installment = to_float(inputs.get('installment'))
    if installment is None or installment <= 0:
        errors.append("Installment amount must be a positive number greater than $0.")

    dti = to_float(inputs.get('dti'))
    if dti is None or dti < 0:
        errors.append("Debt-to-Income (DTI) ratio cannot be negative.")
    elif dti > 100:
        errors.append("Debt-to-Income (DTI) ratio is capped at 100%.")

    term = str(inputs.get('term', '')).strip()
    if term not in ['36 months', '60 months']:
        errors.append("Loan term must be either '36 months' or '60 months'.")

    # Credit history indicators
    for col, name in [
        ('delinq_2yrs', 'Delinquencies in Past 2 Years'),
        ('inq_last_6mths', 'Inquiries in Past 6 Months'),
        ('open_acc', 'Open Credit Accounts'),
        ('pub_rec', 'Public Derogatory Records'),
        ('revol_bal', 'Revolving Balance'),
        ('total_acc', 'Total Credit Lines'),
        ('pub_rec_bankruptcies', 'Bankruptcies')
    ]:
        val = to_float(inputs.get(col))
        if val is not None and val < 0:
            errors.append(f"{name} cannot be negative.")

    return errors


def get_risk_tier(probability: float, low_threshold: float = DEFAULT_LOW_THRESHOLD, med_threshold: float = DEFAULT_MED_THRESHOLD) -> dict:
    """
    Classifies estimated default probability into a risk tier with styling and description.
    """
    prob_pct = probability * 100.0

    if probability < low_threshold:
        tier = "Low"
        badge_color = "#10B981"  # Emerald Green
        bg_color = "rgba(16, 185, 129, 0.1)"
        description = (
            f"The predicted probability of default is {prob_pct:.1f}%, which is below the "
            f"{low_threshold*100:.0f}% threshold. The borrower profile demonstrates strong "
            "credit indicators relative to the training distribution."
        )
    elif probability < med_threshold:
        tier = "Medium"
        badge_color = "#F59E0B"  # Amber Orange
        bg_color = "rgba(245, 158, 11, 0.1)"
        description = (
            f"The predicted probability of default is {prob_pct:.1f}%, falling in the moderate risk band "
            f"({low_threshold*100:.0f}% - {med_threshold*100:.0f}%). Closer underwriting or risk-based pricing "
            "is indicated."
        )
    else:
        tier = "High"
        badge_color = "#EF4444"  # Rose Red
        bg_color = "rgba(239, 68, 68, 0.1)"
        description = (
            f"The predicted probability of default is {prob_pct:.1f}%, which meets or exceeds the "
            f"{med_threshold*100:.0f}% high-risk threshold. Elevated debt-to-income, interest rates, or derogatory "
            "history indicate substantial credit vulnerability."
        )

    return {
        "tier": tier,
        "probability_pct": prob_pct,
        "badge_color": badge_color,
        "bg_color": bg_color,
        "description": description
    }


def predict_loan_default(
    input_dict: dict,
    model_name: str = DEFAULT_MODEL_NAME,
    models_dir: str = MODELS_DIR,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    med_threshold: float = DEFAULT_MED_THRESHOLD
) -> dict:
    """
    Orchestrates validation, preprocessing, and inference.
    Returns prediction results, probabilities, risk tier, and transformed features.
    """
    # 1. Validation
    validation_errors = validate_user_inputs(input_dict)
    if validation_errors:
        return {
            "success": False,
            "errors": validation_errors
        }

    # 2. Load model and preprocessor
    preprocessor, models, metrics, metadata = load_model_artifacts(models_dir)
    if model_name not in models:
        return {
            "success": False,
            "errors": [f"Selected model '{model_name}' is not in the list of available models: {list(models.keys())}"]
        }

    model = models[model_name]

    # 3. Format and transform input
    X_raw = format_user_input(input_dict)
    X_trans = preprocessor.transform(X_raw)

    # 4. Predict
    proba = float(model.predict_proba(X_trans)[0, 1])
    pred = int(proba >= 0.5)

    risk_info = get_risk_tier(proba, low_threshold=low_threshold, med_threshold=med_threshold)

    return {
        "success": True,
        "model_name": model_name,
        "probability": proba,
        "prediction": pred,
        "risk_tier": risk_info["tier"],
        "risk_info": risk_info,
        "X_raw": X_raw,
        "X_trans": X_trans,
        "feature_names": metadata["feature_names"],
        "friendly_feature_names": metadata["friendly_feature_names"]
    }
