"""
test_application.py
-------------------
Automated verification test suite for:
Unsupervised Learning for Borrower Segmentation Using K-Means Clustering.

Validates all pure unsupervised learning requirements, data leakage safeguards,
K-Means pipeline fitting, Davies–Bouldin Index evaluation, cluster profiling,
Cost-Sensitive Decision Support (Feature 1), and Counterfactual Borrower Recourse (Feature 5).
"""

import os
import numpy as np
import pandas as pd

from src.data_processing import (
    LEAKAGE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COL
)
from src.clustering import (
    CANDIDATE_CLUSTERING_FEATURES,
    prepare_clustering_data,
    run_kmeans_segmentation,
    generate_cluster_semantic_labels,
    predict_borrower_cluster
)
from src.cost_support import (
    calculate_segment_costs,
    compare_operational_strategies
)
from src.cluster_counterfactuals import (
    generate_cluster_counterfactuals,
    IMMUTABLE_FEATURES
)
from src.ui_pages.whatif_cluster_page import BORROWER_PRESETS


def test_verification_suite():
    print("=" * 75)
    print("RUNNING VERIFICATION SUITE: UNSUPERVISED BORROWER SEGMENTATION (K-MEANS)")
    print("=" * 75)

    # 1. Dataset loads correctly
    print("\n[CHECK 1] Dataset Loading...")
    data_path = os.path.join("data", "loan_train.csv")
    assert os.path.exists(data_path), "loan_train.csv not found in data/!"
    df = pd.read_csv(data_path, low_memory=False)
    assert len(df) == 27003, f"Expected 27,003 rows, got {len(df)}"
    print(f" PASS: Dataset loaded successfully with {len(df):,} rows and {df.shape[1]} columns.")

    # 2. Data Cleaning & Feature Preparation without requiring target
    print("\n[CHECK 2] Preprocessing Pipeline & Feature Engineering (Unsupervised)...")
    unlabeled_df = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df.copy()
    X_df, X_scaled, feat_names, scaler, imputer, target_series = prepare_clustering_data(unlabeled_df)
    assert not np.isnan(X_scaled).any(), "Found NaNs in scaled clustering feature matrix!"
    assert X_scaled.shape[0] == len(df), f"Expected {len(df)} rows, got {X_scaled.shape[0]}"
    assert len(feat_names) >= 10, f"Expected at least 10 clustering features, got {len(feat_names)}"
    print(f" PASS: Cleaned & scaled matrix shape: {X_scaled.shape}. Used {len(feat_names)} financial features.")

    # 3. Target and Leakage Columns Strictly Excluded
    print("\n[CHECK 3] Target & Leakage Column Exclusion...")
    for feat in feat_names:
        assert feat not in LEAKAGE_COLUMNS, f"Leakage column {feat} leaked into clustering features!"
        assert feat not in IDENTIFIER_COLUMNS, f"Identifier {feat} found in clustering features!"
        assert feat != TARGET_COL, f"Target column '{TARGET_COL}' found in clustering features!"
    print(f" PASS: All {len(LEAKAGE_COLUMNS)} leakage columns, identifiers, and target strictly excluded.")

    # 4. K-Means Clustering Pipeline Execution (k=3)
    print("\n[CHECK 4] K-Means Clustering Pipeline Fitting...")
    sample_df = df.head(1000).copy()
    cluster_res = run_kmeans_segmentation(sample_df, n_clusters=3, random_state=42)
    assert cluster_res["success"] is True, f"Clustering failed: {cluster_res.get('error')}"
    assert cluster_res["n_clusters"] == 3, f"Expected 3 clusters, got {cluster_res['n_clusters']}"
    assert "kmeans" in cluster_res["fitted_pipeline"], "Fitted KMeans model missing from pipeline dict"
    assert "scaler" in cluster_res["fitted_pipeline"], "Fitted StandardScaler missing from pipeline dict"
    assert len(cluster_res["distribution_df"]) == 3, "Distribution dataframe missing cluster rows"
    print(f" PASS: K-Means successfully partitioned {cluster_res['n_records']} records into {cluster_res['n_clusters']} clusters.")

    # 5. Davies–Bouldin Index (DBI) & Inertia (WCSS)
    print("\n[CHECK 5] Davies–Bouldin Index & Performance Metrics Computation...")
    from sklearn.metrics import davies_bouldin_score
    dbi = cluster_res.get("davies_bouldin_index")
    assert dbi is not None, "Davies–Bouldin Index was not computed in cluster results!"
    assert dbi > 0.0, f"Davies–Bouldin Index must be positive, got {dbi}"
    
    # Verify DBI is recalculated when cluster count changes
    cluster_res_k4 = run_kmeans_segmentation(sample_df, n_clusters=4, random_state=42)
    dbi_k4 = cluster_res_k4.get("davies_bouldin_index")
    assert dbi_k4 is not None, "DBI not computed for k=4"
    assert dbi_k4 != dbi, "DBI must update dynamically when cluster count changes"

    # Verify Inertia / WCSS
    assert "inertia" in cluster_res and cluster_res["inertia"] > 0, "Inertia invalid or missing"
    print(f" PASS: Davies–Bouldin Index verified (k=3: {dbi:.4f}, k=4: {dbi_k4:.4f}). Inertia: {cluster_res['inertia']:,.1f}.")

    # 6. Silhouette Score Calculation
    print("\n[CHECK 6] Silhouette Score Computation...")
    sil_score = cluster_res["silhouette_score"]
    assert sil_score is not None, "Silhouette score was not computed"
    assert -1.0 <= sil_score <= 1.0, f"Silhouette score out of bounds: {sil_score}"
    print(f" PASS: Silhouette score computed successfully: {sil_score:.4f}.")

    # 7. Cluster Profile Table & Semantic Personas
    print("\n[CHECK 7] Cluster Profile Table & Persona Interpretation...")
    profile_df = cluster_res["profile_df"]
    assert len(profile_df) == 3, "Profile table row count mismatch"
    assert "loan_amnt" in profile_df.columns, "loan_amnt missing from profile table"
    semantic_labels = generate_cluster_semantic_labels(profile_df)
    assert len(semantic_labels) == 3, "Semantic label count mismatch"
    for cid, label in semantic_labels.items():
        assert len(label) > 0, f"Empty label for cluster {cid}"
    print(f" PASS: Generated profiles and semantic personas: {semantic_labels}")

    # 8. Single Borrower Cluster Assignment
    print("\n[CHECK 8] Single Borrower Prediction via predict_borrower_cluster...")
    test_borrower = BORROWER_PRESETS["⭐ Lower-Debt High-Capacity Borrower"]
    pred_info = predict_borrower_cluster(test_borrower, cluster_res["fitted_pipeline"])
    assert "assigned_cluster" in pred_info, "Assigned cluster missing from prediction"
    assert 0 <= pred_info["assigned_cluster"] < 3, f"Assigned cluster out of range: {pred_info['assigned_cluster']}"
    assert "distance_to_center" in pred_info and pred_info["distance_to_center"] >= 0, "Invalid distance to center"
    print(f" PASS: Sample borrower assigned to Cluster {pred_info['assigned_cluster']} (Distance: {pred_info['distance_to_center']:.2f}).")

    # 9. Feature 1: Cost-Sensitive Decision Support
    print("\n[CHECK 9] Feature 1: Cost-Sensitive Decision Support...")
    custom_costs = {0: 120.0, 1: 350.0, 2: 700.0}
    cost_res = compare_operational_strategies(
        distribution_df=cluster_res["distribution_df"],
        c_baseline_uniform=300.0,
        cluster_action_costs=custom_costs
    )
    assert "comparison_df" in cost_res, "Cost comparison dataframe missing"
    assert len(cost_res["comparison_df"]) == 3, "Expected 3 strategies in comparison table"
    assert cost_res["baseline_total_cost"] == 300.0 * len(sample_df), "Baseline total cost calculation mismatch"
    assert "recommendation" in cost_res, "Strategy recommendation missing"
    print(f" PASS: Cost strategies compared successfully. Baseline: ${cost_res['baseline_total_cost']:,.0f}, Segment-Tailored: ${cost_res['tailored_total_cost']:,.0f}.")

    # 10. Feature 5: Counterfactual Borrower Recourse
    print("\n[CHECK 10] Feature 5: Counterfactual Borrower Recourse...")
    borrower_for_cf = BORROWER_PRESETS["⚖️ Moderate-Debt Balanced Borrower"]
    cf_results = generate_cluster_counterfactuals(
        borrower_input=borrower_for_cf,
        fitted_pipeline=cluster_res["fitted_pipeline"],
        target_cluster_id=None,
        max_recommendations=3
    )
    assert cf_results["success"] is True, f"Counterfactual generation failed: {cf_results.get('error')}"
    assert len(cf_results["recommendations"]) > 0, "No counterfactual recommendations found"
    best_rec = cf_results["recommendations"][0]
    assert best_rec["target_cluster"] != cf_results["current_cluster"], "Recourse did not change cluster"
    assert best_rec["euclidean_distance"] > 0, "Invalid Euclidean distance"
    print(f" PASS: Found {len(cf_results['recommendations'])} recourse paths. Closest transition: Cluster {cf_results['current_cluster']} -> Cluster {best_rec['target_cluster']} (Dist: {best_rec['euclidean_distance']:.2f}).")

    # 11. Immutable Feature Preservation in Counterfactual Recourse
    print("\n[CHECK 11] Immutable Feature Preservation in Recourse...")
    for rec in cf_results["recommendations"]:
        for imm_feat in IMMUTABLE_FEATURES:
            if imm_feat in borrower_for_cf:
                original_val = borrower_for_cf[imm_feat]
                rec_val = rec["proposed_values"].get(imm_feat, original_val)
                assert rec_val == original_val, f"Immutable feature '{imm_feat}' was altered in recourse! {original_val} -> {rec_val}"
    print(f" PASS: All immutable features ({', '.join(IMMUTABLE_FEATURES)}) strictly held constant.")

    # 12. Input Validation Safeguards
    print("\n[CHECK 12] Input Validation Safeguards for Small / Malformed Data...")
    tiny_df = pd.DataFrame({"colA": [1, 2]})  # Only 2 rows, not enough for 3 clusters
    tiny_res = run_kmeans_segmentation(tiny_df, n_clusters=3)
    assert tiny_res["success"] is False, "Expected error on small dataset"
    assert "error" in tiny_res, "Expected descriptive error message"
    print(f" PASS: Safely caught insufficient data: '{tiny_res['error']}'.")

    # 13. Streamlit Page Renderers Integrity
    print("\n[CHECK 13] Streamlit Page Renderers Integrity...")
    from src.ui_pages.home import render_home_page
    from src.ui_pages.upload_page import render_upload_page
    from src.ui_pages.clustering_page import render_clustering_page
    from src.ui_pages.cluster_insights import render_cluster_insights_page
    from src.ui_pages.cost_support_page import render_cost_support_page
    from src.ui_pages.whatif_cluster_page import render_whatif_cluster_page
    from src.ui_pages.about_page import render_about_page
    assert callable(render_home_page)
    assert callable(render_upload_page)
    assert callable(render_clustering_page)
    assert callable(render_cluster_insights_page)
    assert callable(render_cost_support_page)
    assert callable(render_whatif_cluster_page)
    assert callable(render_about_page)
    print(" PASS: All 7 pure unsupervised page renderers imported and verified callable.")

    # 14. Verification of Supervised Exclusion
    print("\n[CHECK 14] Supervised Learning Architecture Exclusion...")
    from app import main as app_main
    import inspect
    app_source = inspect.getsource(app_main)
    forbidden_terms = [
        "render_prediction_page",
        "render_shap_page",
        "render_models_page",
        "render_whatif_page",
        "render_eda_page",
        "predict_loan_default",
        "TreeExplainer",
        "Supervised Mode Active"
    ]
    for term in forbidden_terms:
        assert term not in app_source, f"Forbidden supervised term '{term}' found in app.py!"
    print(" PASS: All supervised pages, classification models, and SHAP terms completely excluded from app.py.")

    print("\n" + "=" * 75)
    print("ALL 14 UNSUPERVISED VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 75)


if __name__ == "__main__":
    test_verification_suite()
