"""
clustering.py
-------------
Unsupervised Learning module using K-Means clustering for borrower financial segmentation.
Excludes identifiers, loan_status, and target-derived leakage features.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

from src.data_processing import clean_raw_data, TARGET_COL, LEAKAGE_COLUMNS, IDENTIFIER_COLUMNS

# Target candidate numerical features for unsupervised borrower segmentation
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


def prepare_clustering_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], Optional[pd.Series]]:
    """
    Cleans raw data and extracts numerical financial features for clustering.
    Guarantees strict exclusion of target (loan_status), leakage, and identifier columns.

    Returns:
        X_df: Cleaned DataFrame with only selected financial features
        X_scaled: Standardized numpy array ready for KMeans
        feature_names: List of features actually used
        target_series: Observed loan_status series (if present) for post-hoc analysis only
    """
    # 1. Clean if not already cleaned
    if 'int_rate_clean' not in df.columns or 'loan_to_income' not in df.columns:
        cleaned_df = clean_raw_data(df)
    else:
        cleaned_df = df.copy()

    # 2. Extract observed target if available (strictly isolated for post-hoc evaluation)
    target_series = None
    if TARGET_COL in cleaned_df.columns:
        target_series = cleaned_df[TARGET_COL].copy()

    # 3. Filter strictly to available candidate features that are NOT in target/leakage/identifiers
    forbidden = set([TARGET_COL] + LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS)
    selected_features = [
        col for col in CANDIDATE_CLUSTERING_FEATURES
        if col in cleaned_df.columns and col not in forbidden
    ]

    if not selected_features:
        raise ValueError("No valid clustering features found in dataset.")

    X_df = cleaned_df[selected_features].copy()

    # 4. Handle infinite and missing values
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)
    X_df = pd.DataFrame(X_imputed, columns=selected_features, index=cleaned_df.index)

    # 5. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    return X_df, X_scaled, selected_features, target_series


def run_kmeans_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
    max_silhouette_samples: int = 3000
) -> Dict[str, Any]:
    """
    Performs K-Means clustering, 2D PCA projection, silhouette score calculation,
    and profile generation.

    Args:
        df: Input DataFrame (raw or cleaned)
        n_clusters: Number of clusters (2 to 6)
        random_state: Random state for reproducibility
        n_init: Number of initializations
        max_silhouette_samples: Sample size for fast silhouette scoring

    Returns:
        Dictionary with clustering results, metadata, profiles, and PCA coords.
    """
    n_clusters = max(2, min(n_clusters, 6))

    X_df, X_scaled, feature_names, target_series = prepare_clustering_data(df)
    n_samples = len(X_df)

    # Ensure n_clusters does not exceed number of samples
    if n_samples < n_clusters:
        n_clusters = max(2, n_samples)

    # 1. Fit KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )
    labels = kmeans.fit_predict(X_scaled)
    cluster_names = [f"Cluster {i}" for i in range(n_clusters)]
    cluster_series = pd.Series([f"Cluster {i}" for i in labels], index=X_df.index, name="Cluster")

    # 2. 2D PCA Projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PCA_1": pca_coords[:, 0],
        "PCA_2": pca_coords[:, 1],
        "Cluster": cluster_series,
        "Cluster_ID": labels
    }, index=X_df.index)

    # Add key hover features
    for hover_col in ['loan_amnt', 'annual_inc', 'int_rate_clean', 'dti']:
        if hover_col in X_df.columns:
            pca_df[hover_col] = X_df[hover_col]

    # 3. Silhouette Score
    sil_score = None
    if n_samples >= n_clusters * 2:
        try:
            if n_samples > max_silhouette_samples:
                # Stratified or random sample for snappy responsive UI
                np.random.seed(random_state)
                sample_idx = np.random.choice(n_samples, size=max_silhouette_samples, replace=False)
                sil_score = float(silhouette_score(X_scaled[sample_idx], labels[sample_idx]))
            else:
                sil_score = float(silhouette_score(X_scaled, labels))
        except Exception:
            sil_score = None

    # 4. Cluster Distribution & Statistics
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

    # 5. Cluster Profile Table (Mean financial characteristics)
    df_with_clusters = X_df.copy()
    df_with_clusters["Cluster"] = cluster_series

    profile_df = df_with_clusters.groupby("Cluster").mean()
    # Reindex to order Cluster 0, Cluster 1, ...
    profile_df = profile_df.reindex(cluster_names).dropna(how='all')

    # 6. Observed target default rate per cluster (strictly for post-hoc validation)
    observed_default_rates = {}
    if target_series is not None:
        target_eval_df = pd.DataFrame({
            "Cluster": cluster_series,
            "Target": target_series
        })
        rate_series = target_eval_df.groupby("Cluster")["Target"].mean() * 100.0
        for c_name in cluster_names:
            observed_default_rates[c_name] = round(float(rate_series.get(c_name, 0.0)), 2)
        distribution_df["Observed_Default_Rate (%)"] = distribution_df["Cluster"].map(observed_default_rates)

    # 7. Generate Natural Language Interpretations based on statistical profiles
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

            obs_txt = f" (Observed Historical Default Rate: {obs_rate:.1f}%)" if obs_rate is not None else ""
            desc = (
                f"**{c_name}** comprises borrowers with average loan requests of **${avg_loan:,.0f}**, "
                f"mean annual income of **${avg_inc:,.0f}**, average interest rate of **{avg_rate:.1f}%**, "
                f"and average DTI of **{avg_dti:.1f}%** (utilization: {avg_util:.1f}%){obs_txt}."
            )
            interpretations[c_name] = desc

    return {
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "features_used": feature_names,
        "labels": labels,
        "silhouette_score": sil_score,
        "distribution_df": distribution_df,
        "profile_df": profile_df,
        "pca_df": pca_df,
        "pca_explained_variance": [float(v) for v in explained_var],
        "interpretations": interpretations,
        "observed_default_rates": observed_default_rates,
        "target_available": target_series is not None
    }
