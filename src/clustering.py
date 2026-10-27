"""
clustering.py
-------------
Core Unsupervised Learning module for Borrower Segmentation using K-Means Clustering.
Operates purely without supervision. Strictly excludes loan_status, IDs, and leakage columns.
Provides pipeline fitting, instance prediction, clustering performance metrics (Silhouette, Davies-Bouldin, Inertia), and cluster profiling.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer

from src.data_processing import (
    clean_raw_data,
    TARGET_COL,
    LEAKAGE_COLUMNS,
    IDENTIFIER_COLUMNS,
    REDUNDANT_POST_ORIGINATION_COLUMNS
)

DATA_PATH = os.path.join("data", "loan_train.csv")

# Available candidate numerical financial features for borrower segmentation
CANDIDATE_CLUSTERING_FEATURES = [
    'loan_amnt',
    'annual_inc',
    'int_rate_clean',
    'installment',
    'dti',
    'emp_length_years',
    'delinq_2yrs',
    'inq_last_6mths',
    'open_acc',
    'revol_bal',
    'revol_util_clean',
    'total_acc',
    'loan_to_income',
    'installment_to_income',
    'revol_to_income'
]


def prepare_clustering_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], StandardScaler, SimpleImputer, Optional[pd.Series]]:
    """
    Cleans raw data and extracts numerical financial features for clustering.
    Guarantees strict exclusion of target (loan_status), leakage, and identifier columns.

    Returns:
        X_df: Cleaned DataFrame with only selected financial features
        X_scaled: Standardized numpy array ready for KMeans
        feature_names: List of features actually used
        scaler: Fitted StandardScaler instance
        imputer: Fitted SimpleImputer instance
        target_series: Observed loan_status series (if present) strictly for post-hoc descriptive comparison
    """
    if df is None or len(df) == 0:
        raise ValueError("Provided dataset is empty.")

    # 1. Clean data using application-time transformation
    if 'int_rate_clean' not in df.columns or 'loan_to_income' not in df.columns:
        cleaned_df = clean_raw_data(df)
    else:
        cleaned_df = df.copy()

    # 2. Extract observed target if available (strictly isolated for post-hoc descriptive check)
    target_series = None
    if TARGET_COL in cleaned_df.columns:
        target_series = cleaned_df[TARGET_COL].copy()

    # 3. Filter strictly to candidate features that are NOT in target, leakage, identifiers, or post-origination
    forbidden = set([TARGET_COL] + LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS + REDUNDANT_POST_ORIGINATION_COLUMNS)
    selected_features = [
        col for col in CANDIDATE_CLUSTERING_FEATURES
        if col in cleaned_df.columns and col not in forbidden
    ]

    if not selected_features:
        raise ValueError("No valid financial clustering features found in dataset.")

    if len(selected_features) < 2:
        raise ValueError(
            f"Clustering requires at least 2 financial numerical features for meaningful multi-dimensional segmentation. "
            f"Found only {len(selected_features)}: {selected_features}."
        )

    if len(cleaned_df) < 3:
        raise ValueError(
            f"Clustering requires at least 3 borrower records. Provided dataset has only {len(cleaned_df)}."
        )

    X_df = cleaned_df[selected_features].copy()

    # 4. Handle infinite and missing values
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)
    X_df = pd.DataFrame(X_imputed, columns=selected_features, index=cleaned_df.index)

    # 5. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    return X_df, X_scaled, selected_features, scaler, imputer, target_series


def generate_cluster_semantic_labels(profile_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generates non-judgmental, descriptive labels for clusters based on actual financial metrics.
    Avoids unsupported terms like 'safe' or 'defaulter'.
    """
    semantic_labels = {}
    for c_name in profile_df.index:
        row = profile_df.loc[c_name]
        loan = row.get('loan_amnt', 0)
        inc = row.get('annual_inc', 0)
        dti = row.get('dti', 0)
        util = row.get('revol_util_clean', 0)

        traits = []
        if loan > 18000:
            traits.append("High-Principal")
        elif loan < 9000:
            traits.append("Low-Principal")
        else:
            traits.append("Moderate-Principal")

        if dti > 20.0 or util > 60.0:
            traits.append("High-Leverage")
        elif dti < 12.0 and util < 35.0:
            traits.append("Low-Leverage")
        else:
            traits.append("Balanced-Debt")

        if inc > 80000:
            traits.append("High-Income")
        elif inc < 45000:
            traits.append("Modest-Income")

        semantic_labels[c_name] = f"{c_name}: {' • '.join(traits)}"
    return semantic_labels


def run_kmeans_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
    max_silhouette_samples: int = 3000
) -> Dict[str, Any]:
    """
    Fits K-Means clustering, calculates performance evaluation metrics (Davies-Bouldin Index,
    Silhouette Score, Inertia), and produces cluster profiles.
    Returns fitted pipeline objects for consistent downstream scoring.
    """
    try:
        X_df, X_scaled, feature_names, scaler, imputer, target_series = prepare_clustering_data(df)
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    n_samples = len(X_df)

    # Constrain clusters between 2 and 6, and not exceeding n_samples - 1
    n_clusters = max(2, min(n_clusters, min(6, n_samples - 1)))

    # 1. Fit KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )
    labels = kmeans.fit_predict(X_scaled)
    cluster_names = [f"Cluster {i}" for i in range(n_clusters)]
    cluster_series = pd.Series([f"Cluster {i}" for i in labels], index=X_df.index, name="Cluster")

    # 2. Performance Evaluation Metrics
    # A. Silhouette Score
    sil_score = None
    if n_samples >= n_clusters * 2 and len(np.unique(labels)) >= 2:
        try:
            if n_samples > max_silhouette_samples:
                np.random.seed(random_state)
                sample_idx = np.random.choice(n_samples, size=max_silhouette_samples, replace=False)
                sil_score = float(silhouette_score(X_scaled[sample_idx], labels[sample_idx]))
            else:
                sil_score = float(silhouette_score(X_scaled, labels))
        except Exception:
            sil_score = None

    # B. Davies–Bouldin Index (DBI)
    # Valid only when there are at least two distinct clusters and k < n_samples
    db_index = None
    unique_clusters = len(np.unique(labels))
    if unique_clusters >= 2 and unique_clusters < n_samples:
        try:
            db_index = float(davies_bouldin_score(X_scaled, labels))
        except Exception:
            db_index = None

    # C. Within-Cluster Sum of Squares (Inertia / WCSS)
    inertia = float(kmeans.inertia_)

    # 3. Cluster Distribution & Statistics
    counts = pd.Series(labels).value_counts().sort_index()
    percentages = (counts / n_samples) * 100.0

    cluster_distribution = []
    for i in range(n_clusters):
        c_count = int(counts.get(i, 0))
        c_pct = float(percentages.get(i, 0.0))
        cluster_distribution.append({
            "Cluster": f"Cluster {i}",
            "Cluster_ID": i,
            "Borrower_Count": c_count,
            "Percentage": round(c_pct, 2)
        })
    distribution_df = pd.DataFrame(cluster_distribution)

    # 4. Cluster Profile Table (Mean financial characteristics)
    df_with_clusters = X_df.copy()
    df_with_clusters["Cluster"] = cluster_series
    profile_df = df_with_clusters.groupby("Cluster").mean().reindex(cluster_names).dropna(how='all')

    semantic_labels = generate_cluster_semantic_labels(profile_df)
    distribution_df["Descriptive_Segment"] = distribution_df["Cluster"].map(semantic_labels)

    # 5. Optional Post-Hoc Observed Target Rates (strictly descriptive, never used in fitting)
    observed_default_rates = {}
    if target_series is not None:
        target_eval_df = pd.DataFrame({
            "Cluster": cluster_series,
            "Target": target_series
        })
        rate_series = target_eval_df.groupby("Cluster")["Target"].mean() * 100.0
        for c_name in cluster_names:
            observed_default_rates[c_name] = round(float(rate_series.get(c_name, 0.0)), 2)
        distribution_df["Observed_Historical_Default_Rate (%)"] = distribution_df["Cluster"].map(observed_default_rates)

    # 6. Natural Language Statistical Interpretations
    interpretations = {}
    for c_name in cluster_names:
        if c_name in profile_df.index:
            row = profile_df.loc[c_name]
            avg_loan = row.get('loan_amnt', 0)
            avg_inc = row.get('annual_inc', 0)
            avg_rate = row.get('int_rate_clean', 0)
            avg_dti = row.get('dti', 0)
            avg_util = row.get('revol_util_clean', 0)
            obs_rate = observed_default_rates.get(c_name, None)

            obs_txt = f" (Post-hoc historical default rate: {obs_rate:.1f}%)" if obs_rate is not None else ""
            desc = (
                f"**{semantic_labels.get(c_name, c_name)}**: Average loan request of **${avg_loan:,.0f}**, "
                f"mean annual earnings of **${avg_inc:,.0f}**, average interest rate of **{avg_rate:.1f}%**, "
                f"and DTI ratio of **{avg_dti:.1f}%** (revolving utilization: {avg_util:.1f}%){obs_txt}."
            )
            interpretations[c_name] = desc

    fitted_pipeline = {
        "scaler": scaler,
        "imputer": imputer,
        "kmeans": kmeans,
        "features": feature_names,
    }

    return {
        "success": True,
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "n_records": n_samples,
        "features_used": feature_names,
        "labels": labels,
        "silhouette_score": sil_score,
        "davies_bouldin_index": db_index,
        "inertia": inertia,
        "distribution_df": distribution_df,
        "profile_df": profile_df,
        "interpretations": interpretations,
        "semantic_labels": semantic_labels,
        "observed_default_rates": observed_default_rates,
        "target_available": target_series is not None,
        "fitted_pipeline": fitted_pipeline,
        "scaler": scaler,
        "imputer": imputer,
        "kmeans": kmeans,
        "cluster_centers": kmeans.cluster_centers_
    }


def predict_borrower_cluster(
    arg1: Dict[str, Any],
    arg2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transforms a single borrower dictionary using the exact fitted scaler & imputer,
    and assigns it to a cluster via the fitted KMeans model.
    Accepts either (borrower_input, fitted_pipeline) or (fitted_pipeline, borrower_input).
    """
    if "kmeans" in arg1 or "scaler" in arg1:
        fitted_pipeline, borrower_input = arg1, arg2
    else:
        borrower_input, fitted_pipeline = arg1, arg2

    scaler: StandardScaler = fitted_pipeline["scaler"]
    imputer: SimpleImputer = fitted_pipeline["imputer"]
    kmeans: KMeans = fitted_pipeline["kmeans"]
    feature_names: List[str] = fitted_pipeline.get("features_used") or fitted_pipeline.get("features", [])

    # Build single-row DataFrame
    b_df = pd.DataFrame([borrower_input])

    # Recompute financial ratios if not present or changed
    if 'annual_inc' in b_df.columns:
        inc = float(b_df['annual_inc'].iloc[0]) + 1.0
        if 'loan_amnt' in b_df.columns and 'loan_to_income' not in b_df.columns:
            b_df['loan_to_income'] = float(b_df['loan_amnt'].iloc[0]) / inc
        if 'installment' in b_df.columns and 'installment_to_income' not in b_df.columns:
            b_df['installment_to_income'] = (float(b_df['installment'].iloc[0]) * 12.0) / inc
        if 'revol_bal' in b_df.columns and 'revol_to_income' not in b_df.columns:
            b_df['revol_to_income'] = float(b_df['revol_bal'].iloc[0]) / inc

    # Align columns to feature_names
    for col in feature_names:
        if col not in b_df.columns:
            b_df[col] = np.nan

    X_single = b_df[feature_names].copy()
    X_imputed = imputer.transform(X_single)
    X_scaled = scaler.transform(pd.DataFrame(X_imputed, columns=feature_names))

    cluster_id = int(kmeans.predict(X_scaled)[0])
    cluster_name = f"Cluster {cluster_id}"
    semantic_name = fitted_pipeline.get("semantic_labels", {}).get(cluster_name, cluster_name)

    # Compute Euclidean distance to cluster centroid in normalized space
    centroid = kmeans.cluster_centers_[cluster_id]
    dist_to_center = float(np.linalg.norm(X_scaled[0] - centroid))

    return {
        "cluster_id": cluster_id,
        "assigned_cluster": cluster_id,
        "cluster_name": cluster_name,
        "semantic_name": semantic_name,
        "distance_to_center": round(dist_to_center, 4),
        "X_scaled": X_scaled[0]
    }
