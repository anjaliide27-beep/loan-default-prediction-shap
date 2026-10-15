"""
train_model.py
--------------
Training, evaluation, metric calculation, and artifact persistence
for Explainable Loan Default Prediction models.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from src.data_processing import (
    clean_raw_data,
    get_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    get_friendly_feature_name
)

DATA_PATH = os.path.join("data", "loan_train.csv")
MODELS_DIR = "models"


def load_and_split_data(data_path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """
    Loads dataset, cleans application features, and performs stratified train/test split.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place loan_train.csv in data/ directory.")

    raw_df = pd.read_csv(data_path, low_memory=False)
    cleaned_df = clean_raw_data(raw_df)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = cleaned_df[feature_cols].copy()
    y = cleaned_df[TARGET_COL].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, raw_df


def train_and_evaluate_all():
    """
    Fits preprocessor and candidate models, evaluates metrics, computes SHAP, and saves artifacts.
    """
    print("=" * 60)
    print("Starting Explainable Loan Default Model Training Pipeline")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load and split data
    X_train, X_test, y_train, y_test, raw_df = load_and_split_data()
    print(f"Dataset loaded: {len(raw_df):,} total rows.")
    print(f"Training set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")
    print(f"Default rate in train: {y_train.mean():.2%}, in test: {y_test.mean():.2%}")

    # 2. Fit Preprocessor on X_train only (strictly preventing data leakage)
    preprocessor = get_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    friendly_feature_names = [get_friendly_feature_name(fn) for fn in feature_names]

    print(f"Transformed feature space: {len(feature_names)} features.")

    # 3. Define candidate algorithms with imbalance compensation
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / max(1, pos_count))

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        )
    }

    # 4. Train, evaluate, and collect metrics
    trained_models = {}
    metrics_summary = {}

    for model_name, clf in candidate_models.items():
        print(f"\n--- Training {model_name} ---")
        clf.fit(X_train_trans, y_train)
        trained_models[model_name] = clf

        # Evaluate on unseen test data
        y_pred = clf.predict(X_test_trans)
        y_proba = clf.predict_proba(X_test_trans)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        # Sample points along ROC curve to keep JSON lightweight
        sample_indices = np.linspace(0, len(fpr) - 1, num=min(100, len(fpr)), dtype=int)
        roc_data = {
            "fpr": [float(fpr[i]) for i in sample_indices],
            "tpr": [float(tpr[i]) for i in sample_indices]
        }

        metrics_summary[model_name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "roc_curve": roc_data
        }

        print(f"{model_name} Results: AUC={auc:.4f}, Recall={rec:.4f}, Prec={prec:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

    # 5. Fit SHAP Explainer (Primary Model: Random Forest & XGBoost)
    print("\n--- Initializing SHAP Explainers ---")
    rf_model = trained_models["Random Forest"]
    rf_explainer = shap.TreeExplainer(rf_model)

    xgb_model = trained_models["XGBoost"]
    xgb_explainer = shap.TreeExplainer(xgb_model)

    # Compute global SHAP values on a stratified test sample (e.g. 300 instances) for fast global visualizations
    np.random.seed(42)
    sample_size = min(300, len(X_test_trans))
    indices = np.random.choice(len(X_test_trans), sample_size, replace=False)
    X_test_sample_trans = X_test_trans[indices]
    X_test_sample_raw = X_test.iloc[indices].copy()

    # Pre-calculate global mean absolute SHAP values for fast loading in UI
    rf_shap_raw = rf_explainer.shap_values(X_test_sample_trans)
    if isinstance(rf_shap_raw, list):
        rf_shap_class1 = rf_shap_raw[1]
    elif len(rf_shap_raw.shape) == 3:
        rf_shap_class1 = rf_shap_raw[:, :, 1]
    else:
        rf_shap_class1 = rf_shap_raw

    global_importance_rf = np.abs(rf_shap_class1).mean(axis=0).tolist()

    # 6. Save all artifacts to disk
    print("\n--- Saving Artifacts to models/ ---")
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    joblib.dump(trained_models, os.path.join(MODELS_DIR, "models.joblib"))
    joblib.dump(rf_explainer, os.path.join(MODELS_DIR, "rf_explainer.joblib"))
    joblib.dump(xgb_explainer, os.path.join(MODELS_DIR, "xgb_explainer.joblib"))
    joblib.dump(X_test_sample_trans, os.path.join(MODELS_DIR, "test_sample_trans.joblib"))
    joblib.dump(X_test_sample_raw, os.path.join(MODELS_DIR, "test_sample_raw.joblib"))

    metadata = {
        "feature_names": feature_names,
        "friendly_feature_names": friendly_feature_names,
        "global_importance_rf": global_importance_rf,
        "primary_model": "Random Forest",
        "total_dataset_rows": len(raw_df),
        "total_dataset_cols": raw_df.shape[1],
        "selected_feature_count": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES),
        "target_distribution": {
            "fully_paid_0": int((raw_df[TARGET_COL] == 0).sum()),
            "default_1": int((raw_df[TARGET_COL] == 1).sum())
        }
    }

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Model training and artifact persistence complete!")
    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate_all()
