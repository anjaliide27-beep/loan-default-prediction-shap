"""
test_application.py
-------------------
Automated verification test suite for Explainable Loan Default Prediction.
Validates all 10 requirements from Section 8 of the project specifications.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data_processing import (
    clean_raw_data,
    get_preprocessor,
    format_user_input,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COL
)
from src.prediction import (
    predict_loan_default,
    predict_batch,
    validate_user_inputs,
    load_model_artifacts
)
from src.explainability import (
    load_explainer_artifacts,
    compute_instance_shap,
    create_local_waterfall_plot,
    create_global_importance_chart,
    generate_plain_language_summary
)
from src.ui_pages.prediction_page import PRESETS


def test_verification_suite():
    print("=" * 70)
    print("RUNNING VERIFICATION SUITE: EXPLAINABLE LOAN DEFAULT PREDICTION")
    print("=" * 70)

    # 1. Dataset loads correctly
    print("\n[CHECK 1] Dataset Loading...")
    data_path = os.path.join("data", "loan_train.csv")
    assert os.path.exists(data_path), "loan_train.csv not found in data/!"
    df = pd.read_csv(data_path, low_memory=False)
    assert len(df) == 27003, f"Expected 27,003 rows, got {len(df)}"
    assert TARGET_COL in df.columns, "loan_status missing from dataset"
    print(f" PASS: Dataset loaded successfully with {len(df):,} rows and {df.shape[1]} columns.")

    # 2. Preprocessing handles missing and categorical values
    print("\n[CHECK 2] Preprocessing Pipeline...")
    cleaned = clean_raw_data(df)
    preprocessor = get_preprocessor()
    X = cleaned[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    X_trans = preprocessor.fit_transform(X)
    assert not np.isnan(X_trans).any(), "Found NaNs in transformed feature matrix!"
    assert X_trans.shape[1] == 46, f"Expected 46 transformed features, got {X_trans.shape[1]}"
    print(f" PASS: Transformed shape {X_trans.shape}. Missing values imputed and categoricals encoded.")

    # 3. Training completed successfully & models saved
    print("\n[CHECK 3] Model Artifact Persistence...")
    preproc, models, metrics, metadata = load_model_artifacts()
    expected_models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    for m in expected_models:
        assert m in models, f"Model {m} missing from models dict!"
    print(f" PASS: All 4 models loaded successfully: {list(models.keys())}")

    # 4. Predictions run for valid input
    print("\n[CHECK 4] Prediction Execution for Valid Input...")
    sample_input = PRESETS["⭐ Prime Borrower (Low Risk Profile)"]
    pred_res = predict_loan_default(sample_input, model_name="Random Forest")
    assert pred_res["success"] is True, f"Prediction failed: {pred_res.get('errors')}"
    assert 0.0 <= pred_res["probability"] <= 1.0, f"Probability out of range: {pred_res['probability']}"
    assert pred_res["risk_tier"] in ["Low", "Medium", "High"], f"Invalid risk tier: {pred_res['risk_tier']}"
    print(f" PASS: Valid input produced probability = {pred_res['probability']*100:.2f}%, Tier = {pred_res['risk_tier']}")

    # 5. Invalid input produces useful error message
    print("\n[CHECK 5] Input Validation & Error Messaging...")
    invalid_input = dict(sample_input)
    invalid_input["loan_amnt"] = -5000  # Negative loan amount
    invalid_input["annual_inc"] = -100  # Negative income
    invalid_input["int_rate"] = 65.0   # Out of bound rate
    invalid_res = predict_loan_default(invalid_input)
    assert invalid_res["success"] is False, "Validation failed to flag invalid inputs!"
    assert len(invalid_res["errors"]) >= 3, "Did not capture all expected validation errors!"
    print(f" PASS: Successfully caught {len(invalid_res['errors'])} validation errors with clear messages.")

    # 6. Target and leakage columns excluded from prediction inputs
    print("\n[CHECK 6] Target & Leakage Column Exclusion...")
    input_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    assert TARGET_COL not in input_features, "TARGET_COL leaked into input features!"
    for col in LEAKAGE_COLUMNS:
        assert col not in input_features, f"Leakage column {col} found in input features!"
    for col in IDENTIFIER_COLUMNS:
        assert col not in input_features, f"Identifier {col} found in input features!"
    print(f" PASS: Target and all {len(LEAKAGE_COLUMNS)} leakage columns strictly excluded.")

    # 7. Model metrics computed from held-out test data
    print("\n[CHECK 7] Test Set Evaluation Metrics Integrity...")
    with open(os.path.join("models", "metrics.json"), "r") as f:
        saved_metrics = json.load(f)
    for model_name, m in saved_metrics.items():
        assert "roc_auc" in m and 0.5 < m["roc_auc"] <= 1.0, f"Invalid ROC-AUC for {model_name}"
        assert "recall" in m and 0.0 < m["recall"] <= 1.0, f"Invalid Recall for {model_name}"
        assert "confusion_matrix" in m, f"Missing confusion matrix for {model_name}"
        cm = np.array(m["confusion_matrix"])
        assert cm.sum() == 5401, f"Expected 5,401 test instances, got {cm.sum()}"
    print(f" PASS: Held-out test set size = 5,401. Verified metrics for all 4 models.")

    # 8. SHAP explanations correspond to selected model and prediction
    print("\n[CHECK 8] SHAP Attribution & Scaling Correctness...")
    explainer, metadata, _ = load_explainer_artifacts()
    shap_info = compute_instance_shap(pred_res["X_trans"], explainer=explainer)
    base_val = shap_info["base_value"]
    shap_sum = shap_info["class1_shap"].sum()
    reconstructed_proba = base_val + shap_sum
    # TreeExplainer probability output matches prediction
    diff = abs(reconstructed_proba - pred_res["probability"])
    assert diff < 1e-4, f"SHAP sum mismatch: {reconstructed_proba} vs {pred_res['probability']}"
    fig_waterfall = create_local_waterfall_plot(shap_info)
    assert len(fig_waterfall.data) > 0, "Waterfall plot has no traces"
    summary_txt = generate_plain_language_summary(shap_info, sample_input)
    assert len(summary_txt) > 50, "Summary text too short or empty"
    print(f" PASS: SHAP efficiency verified: Base ({base_val:.4f}) + SHAP sum ({shap_sum:+.4f}) = Model Output ({pred_res['probability']:.4f})")

    # 9. What-If analysis recalculates results
    print("\n[CHECK 9] What-If Counterfactual Recalculation...")
    revised_input = dict(sample_input)
    revised_input["annual_inc"] = sample_input["annual_inc"] * 0.4  # Severe income cut
    revised_input["dti"] = 35.0                                    # Elevated DTI
    revised_res = predict_loan_default(revised_input, model_name="Random Forest")
    assert revised_res["success"] is True
    # Default risk should increase when income drops and DTI rises
    assert revised_res["probability"] > pred_res["probability"], "What-If risk did not respond to financial stress!"
    print(f" PASS: What-If simulation adjusted probability from {pred_res['probability']*100:.1f}% to {revised_res['probability']*100:.1f}%.")

    # 10. Website pages import without runtime errors
    print("\n[CHECK 10] Streamlit Page Renderers Integrity...")
    from src.ui_pages.home import render_home_page
    from src.ui_pages.upload_page import render_upload_page
    from src.ui_pages.eda import render_eda_page
    from src.ui_pages.prediction_page import render_prediction_page
    from src.ui_pages.shap_page import render_shap_page
    from src.ui_pages.models_page import render_models_page
    from src.ui_pages.whatif_page import render_whatif_page
    from src.ui_pages.clustering_page import render_clustering_page
    print(" PASS: All 8 active page modules imported cleanly without syntax or dependency errors.")

    # 11. Batch dataset prediction and risk scoring
    print("\n[CHECK 11] Batch Dataset Prediction & Risk Scoring...")
    sample_batch = df.head(15).copy()
    batch_res = predict_batch(sample_batch, model_name="Random Forest")
    assert batch_res["success"] is True, f"Batch prediction failed: {batch_res.get('errors')}"
    assert len(batch_res["scored_df"]) == 15, "Batch scored dataframe length mismatch!"
    assert "Predicted_Risk_Tier" in batch_res["scored_df"].columns, "Predicted_Risk_Tier column missing!"
    assert "Default_Probability (%)" in batch_res["scored_df"].columns, "Default_Probability column missing!"
    assert batch_res["summary"]["total_records"] == 15, "Summary total_records mismatch!"
    print(f" PASS: Batch prediction successfully scored 15 records. Avg probability = {batch_res['summary']['avg_probability_pct']}%.")

    # 12. Feature 1: K-Means Borrower Segmentation (Unsupervised Learning)
    print("\n[CHECK 12] Feature 1: K-Means Clustering & Leakage Safeguards...")
    from src.clustering import run_kmeans_segmentation
    cluster_res = run_kmeans_segmentation(df.head(500), n_clusters=3, random_state=42)
    assert cluster_res["n_clusters"] == 3, "K-Means cluster count mismatch!"
    assert TARGET_COL not in cluster_res["features_used"], "TARGET_COL leaked into clustering features!"
    for lk in LEAKAGE_COLUMNS:
        assert lk not in cluster_res["features_used"], f"Leakage column {lk} leaked into clustering features!"
    for id_col in IDENTIFIER_COLUMNS:
        assert id_col not in cluster_res["features_used"], f"Identifier {id_col} leaked into clustering features!"
    assert len(cluster_res["distribution_df"]) == 3, "Distribution dataframe must have 3 clusters"
    assert cluster_res["pca_df"].shape[1] >= 4, "PCA dataframe missing projection columns"
    print(f" PASS: K-Means successfully segmented cohort into 3 clusters using {len(cluster_res['features_used'])} features. Target and leakage strictly excluded.")

    # 13. Feature 2: Cost-Sensitive Decision Threshold Optimization
    print("\n[CHECK 13] Feature 2: Cost-Sensitive Threshold Optimization...")
    from src.cost_optimization import get_validation_data_and_probabilities, optimize_decision_threshold
    val_info = get_validation_data_and_probabilities(data_path, model_name="Random Forest")
    assert val_info["success"] is True, "Validation data extraction failed!"
    assert val_info["n_validation_samples"] == 4321, f"Expected 4,321 validation samples, got {val_info['n_validation_samples']}"
    opt_cost_res = optimize_decision_threshold(val_info["y_true"], val_info["y_proba"], c_fn=5000.0, c_fp=1000.0)
    assert 0.05 <= opt_cost_res["optimal_threshold"] <= 0.95, "Optimal threshold out of bounds!"
    assert opt_cost_res["cost_savings"] > 0, "Optimal threshold failed to reduce total error cost vs default 0.50!"
    print(f" PASS: Optimal threshold tau* = {opt_cost_res['optimal_threshold']:.2f}. Total validation savings = ${opt_cost_res['cost_savings']:,.0f} ({opt_cost_res['cost_savings_pct']:.1f}% reduction).")

    # 14. Feature 3: Actionable Counterfactual Recourse
    print("\n[CHECK 14] Feature 3: Actionable Counterfactual Recourse...")
    from src.counterfactuals import generate_actionable_counterfactuals, IMMUTABLE_FEATURES
    # Test on an elevated risk applicant
    elevated_input = dict(PRESETS["⚖️ Average Borrower (Moderate Risk Profile)"])
    elevated_input["dti"] = 28.0
    elevated_input["loan_amnt"] = 25000
    cf_res = generate_actionable_counterfactuals(elevated_input, model_name="Random Forest", decision_threshold=0.25)
    assert cf_res["success"] is True, "Counterfactual generation failed!"
    assert cf_res["valid_count"] > 0, "Expected valid counterfactual recourse recommendations for elevated borrower!"
    for r in cf_res["recommendations"]:
        for imm in IMMUTABLE_FEATURES:
            if imm in elevated_input:
                assert r["input_data"][imm] == elevated_input[imm], f"Immutable feature {imm} was modified!"
        assert r["new_proba"] < r["base_proba"], "Recommended counterfactual did not reduce default probability!"
    print(f" PASS: Found {cf_res['valid_count']} recourse recommendations for elevated borrower. Immutable credit history strictly preserved.")

    print("\n" + "=" * 70)
    print("ALL 14 VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_verification_suite()

