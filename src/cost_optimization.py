"""
cost_optimization.py
--------------------
Future Scope Item 1: Cost-Sensitive Decision Threshold Adjustment.
Evaluates classification decision thresholds on a held-out validation set to minimize
illustrative financial error costs: Cost = (FN * C_FN) + (FP * C_FP).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

from src.data_processing import (
    clean_raw_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL
)
from src.prediction import load_model_artifacts

DATA_PATH = os.path.join("data", "loan_train.csv")


def get_validation_data_and_probabilities(
    data_path: str = DATA_PATH,
    model_name: str = "Random Forest",
    val_size: float = 0.2,
    split_random_state: int = 99
) -> Dict[str, Any]:
    """
    Extracts a dedicated validation split separate from model training and separate from the test set.
    Computes model predicted default probabilities on this validation set.

    Guarantees:
    - Test set (held out with random_state=42) is NOT touched.
    - Training data is partitioned so validation data acts as independent out-of-sample data.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Load artifacts
    preprocessor, models, _, _ = load_model_artifacts()
    if model_name not in models:
        model_name = "Random Forest"
    model = models[model_name]

    # 2. Load and clean raw data
    raw_df = pd.read_csv(data_path, low_memory=False)
    cleaned_df = clean_raw_data(raw_df)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = cleaned_df[feature_cols].copy()
    y = cleaned_df[TARGET_COL].astype(int).copy()

    # 3. Partition into Train (80%) and Test (20%) using original seed=42
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Partition X_train_full into Sub-Train and Validation using separate seed=split_random_state
    # This guarantees X_val is completely separate from model test set
    _, X_val, _, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=split_random_state, stratify=y_train_full
    )

    # 5. Transform validation features and compute probabilities
    X_val_trans = preprocessor.transform(X_val)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val_trans)[:, 1]
    else:
        # Fallback if model only outputs decision function
        y_proba = model.predict(X_val_trans)

    return {
        "success": True,
        "n_validation_samples": len(y_val),
        "y_true": y_val.values,
        "y_proba": y_proba,
        "model_name": model_name,
        "validation_defaults": int(y_val.sum()),
        "validation_non_defaults": int(len(y_val) - y_val.sum())
    }


def compute_threshold_cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    c_fn: float = 5000.0,
    c_fp: float = 1000.0,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01
) -> pd.DataFrame:
    """
    Evaluates candidate classification thresholds from threshold_min to threshold_max.
    Computes False Negatives, False Positives, and total illustrative error cost:
        Total Cost = (FN * C_FN) + (FP * C_FP)
    """
    thresholds = np.arange(threshold_min, threshold_max + 1e-5, threshold_step)
    curve_data = []

    y_true_bool = y_true.astype(bool)
    n_positive = np.sum(y_true_bool)
    n_negative = len(y_true) - n_positive

    for t in thresholds:
        y_pred = y_proba >= t

        tp = int(np.sum(y_true_bool & y_pred))
        fp = int(np.sum((~y_true_bool) & y_pred))
        fn = int(np.sum(y_true_bool & (~y_pred)))
        tn = int(np.sum((~y_true_bool) & (~y_pred)))

        total_cost = (fn * c_fn) + (fp * c_fp)
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        curve_data.append({
            "threshold": round(float(t), 2),
            "false_negatives": fn,
            "false_positives": fp,
            "true_positives": tp,
            "true_negatives": tn,
            "total_cost": float(total_cost),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall)
        })

    return pd.DataFrame(curve_data)


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    c_fn: float = 5000.0,
    c_fp: float = 1000.0
) -> Dict[str, Any]:
    """
    Identifies the optimal decision threshold tau* that minimizes total error cost.
    Compares against default threshold 0.50.
    """
    df_curve = compute_threshold_cost_curve(y_true, y_proba, c_fn, c_fp)

    # Find minimum cost entry
    best_row = df_curve.loc[df_curve["total_cost"].idxmin()]
    opt_threshold = float(best_row["threshold"])
    opt_cost = float(best_row["total_cost"])
    opt_fn = int(best_row["false_negatives"])
    opt_fp = int(best_row["false_positives"])

    # Find default 0.50 threshold entry
    idx_50 = (df_curve["threshold"] - 0.50).abs().idxmin()
    row_50 = df_curve.loc[idx_50]
    cost_50 = float(row_50["total_cost"])
    fn_50 = int(row_50["false_negatives"])
    fp_50 = int(row_50["false_positives"])

    # Cost savings
    savings = cost_50 - opt_cost
    savings_pct = (savings / cost_50 * 100.0) if cost_50 > 0 else 0.0

    return {
        "optimal_threshold": opt_threshold,
        "optimal_cost": opt_cost,
        "optimal_fn": opt_fn,
        "optimal_fp": opt_fp,
        "default_threshold": 0.50,
        "default_cost": cost_50,
        "default_fn": fn_50,
        "default_fp": fp_50,
        "cost_savings": savings,
        "cost_savings_pct": savings_pct,
        "c_fn": c_fn,
        "c_fp": c_fp,
        "cost_curve_df": df_curve
    }
