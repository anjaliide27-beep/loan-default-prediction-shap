"""
upload_page.py
--------------
Simplified Dataset Upload Page for Unsupervised K-Means Borrower Segmentation.
Enables uploading CSV datasets, validates financial numerical attributes,
and runs K-Means segmentation without requiring loan outcomes (loan_status).
"""

import os
import io
import hashlib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from src.clustering import run_kmeans_segmentation, CANDIDATE_CLUSTERING_FEATURES
from src.ui_pages.styles import render_disclaimer, render_metric_card


def compute_df_hash(df: pd.DataFrame) -> str:
    """Computes a deterministic MD5 hash of a DataFrame to track dataset identity."""
    val_str = f"{df.shape}_{list(df.columns)}_{df.iloc[:min(10, len(df))].values.tobytes()}"
    return hashlib.md5(val_str.encode('utf-8', errors='ignore')).hexdigest()


def render_upload_page():
    st.markdown('<div class="section-header">📁 Upload Dataset for Borrower Segmentation</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a CSV dataset containing borrower financial records to discover natural applicant segments 
    using **Unsupervised K-Means Clustering**. 
    **Repayment outcome labels (`loan_status`) are NOT required.** 
    The clustering engine standardizes numerical attributes and identifies behavioral clusters without supervision.
    """)

    # Upload Section
    col_upload, col_sample = st.columns([1.8, 1.2])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Borrower Dataset (.CSV):",
            type=["csv"],
            help="Upload a CSV file containing borrower loan attributes (e.g. loan_amnt, annual_inc, dti, etc.)."
        )

    with col_sample:
        st.caption("Quick Test Demonstration:")
        load_sample = st.button("📋 Load Sample LendingClub Dataset (30 Records)", use_container_width=True)

    input_df = None
    input_file_name = "Uploaded Dataset"

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, low_memory=False)
            input_file_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error parsing uploaded CSV file: {e}")
            return
    elif load_sample:
        data_path = os.path.join("data", "loan_train.csv")
        if os.path.exists(data_path):
            sample_df = pd.read_csv(data_path, low_memory=False, nrows=30)
            if 'loan_status' in sample_df.columns:
                sample_df = sample_df.drop(columns=['loan_status'])
            input_df = sample_df
            input_file_name = "sample_unlabeled_borrowers_30.csv"
            st.session_state["cached_sample_df"] = input_df
            st.session_state["cached_sample_name"] = input_file_name
    elif "cached_sample_df" in st.session_state:
        input_df = st.session_state["cached_sample_df"]
        input_file_name = st.session_state.get("cached_sample_name", "Sample Dataset")
    elif "uploaded_df" in st.session_state:
        input_df = st.session_state["uploaded_df"]
        input_file_name = st.session_state.get("uploaded_file_name", "Uploaded Dataset")

    if input_df is None or len(input_df) == 0:
        st.info("👆 Please upload a `.csv` file above or click **'Load Sample LendingClub Dataset'** to begin.")
        return

    # Cache invalidation on dataset identity change
    current_hash = compute_df_hash(input_df)
    prev_hash = st.session_state.get("uploaded_dataset_hash")

    if prev_hash != current_hash:
        st.session_state.pop("clustering_results", None)
        st.session_state.pop("whatif_custom_borrower", None)
        st.session_state["uploaded_dataset_hash"] = current_hash
        st.session_state["uploaded_df"] = input_df
        st.session_state["uploaded_file_name"] = input_file_name

    # Validate features
    available_features = [c for c in CANDIDATE_CLUSTERING_FEATURES if c in input_df.columns or (c == 'loan_to_income' and 'loan_amnt' in input_df.columns and 'annual_inc' in input_df.columns)]
    
    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.85rem 1.15rem; margin-bottom: 1rem;">
        <strong>Active Dataset:</strong> <code>{input_file_name}</code> &nbsp;|&nbsp; 
        <strong>Dimensions:</strong> {len(input_df):,} rows × {input_df.shape[1]} columns &nbsp;|&nbsp; 
        <span style="background-color: #E0F2FE; color: #0369A1; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">
            {len(available_features)} Financial Features Detected
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👁️ View Dataset Preview (First 5 Rows)", expanded=False):
        st.dataframe(input_df.head(5), use_container_width=True)

    # Configuration Controls
    st.subheader("Clustering Configuration")
    col_c1, col_c2 = st.columns([1.5, 2.5])
    with col_c1:
        n_clusters = st.slider(
            "Select Number of Clusters (k):",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            help="Number of borrower segments to discover using K-Means (default: 3)."
        )
    with col_c2:
        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.65rem 0.9rem; font-size: 0.82rem; color: #475569; margin-top: 0.25rem;">
            <strong>Unsupervised Integrity:</strong> Identifiers, loan outcomes, and leakage features are strictly excluded. 
            All financial attributes are imputed and standardized with <code>StandardScaler</code>.
        </div>
        """, unsafe_allow_html=True)

    cluster_btn = st.button("🚀 Load Dataset and Run K-Means Clustering", type="primary", use_container_width=True)

    # Run clustering if button clicked or if already cached
    if cluster_btn or "clustering_results" not in st.session_state:
        with st.spinner(f"Clustering {len(input_df):,} borrower records into {n_clusters} segments with K-Means..."):
            try:
                results = run_kmeans_segmentation(input_df, n_clusters=n_clusters)
                st.session_state["clustering_results"] = results
                st.session_state["uploaded_df"] = input_df
                st.session_state["uploaded_file_name"] = input_file_name
            except Exception as e:
                st.error(f"❌ Clustering failed: {e}")
                return

    c_res = st.session_state["clustering_results"]

    st.markdown("---")
    st.subheader("Segmentation Summary")

    # KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card("Total Borrowers", f"{c_res['n_samples']:,}", f"Source: {input_file_name}", "#0EA5E9"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card("Discovered Clusters", f"{c_res['n_clusters']}", f"{len(c_res['features_used'])} Features Used", "#10B981"), unsafe_allow_html=True)
    with m3:
        db_s = c_res.get("davies_bouldin_index")
        db_txt = f"{db_s:.3f}" if db_s is not None else "N/A"
        st.markdown(render_metric_card("Davies–Bouldin Index", db_txt, "Lower is preferred (Separation)", "#0284C7"), unsafe_allow_html=True)
    with m4:
        sil_s = c_res.get("silhouette_score")
        sil_txt = f"{sil_s:.3f}" if sil_s is not None else "N/A"
        st.markdown(render_metric_card("Silhouette Score", sil_txt, "Higher is preferred [-1, +1]", "#8B5CF6"), unsafe_allow_html=True)

    # Visualizations & Performance Metrics
    col_v1, col_v2 = st.columns([1.1, 1.4])
    cluster_colors = ["#0F766E", "#0284C7", "#D97706", "#7C3AED", "#DC2626", "#059669"]

    with col_v1:
        st.markdown("##### Borrower Distribution by Segment")
        dist_df = c_res["distribution_df"]
        fig_bar = px.bar(
            dist_df,
            x="Cluster",
            y="Borrower_Count",
            text=dist_df.apply(lambda r: f"{r['Borrower_Count']:,} ({r['Percentage']}%)", axis=1),
            color="Cluster",
            color_discrete_sequence=cluster_colors[:c_res['n_clusters']]
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            yaxis=dict(title="Borrower Count", gridcolor="#F1F5F9"),
            height=340,
            margin=dict(t=20, b=30, l=40, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_v2:
        st.markdown("##### Clustering Evaluation & Quality Metrics")
        db_val = c_res.get("davies_bouldin_index")
        sil_val = c_res.get("silhouette_score")
        inertia_val = c_res.get("inertia", 0.0)

        db_disp = f"{db_val:.4f}" if db_val is not None else "Davies–Bouldin Index unavailable for this clustering configuration"
        sil_disp = f"{sil_val:.4f}" if sil_val is not None else "Silhouette Score unavailable for this clustering configuration"
        inertia_disp = f"{inertia_val:,.1f}"

        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="margin-bottom: 0.85rem; padding-bottom: 0.65rem; border-bottom: 1px solid #F1F5F9;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-weight: 700; color: #0F172A; font-size: 0.92rem;">📊 Davies–Bouldin Index (DBI)</span>
                    <span style="font-family: monospace; font-size: 1.1rem; font-weight: 800; color: #0284C7;">{db_disp}</span>
                </div>
                <div style="font-size: 0.76rem; color: #0D9488; font-weight: 600; margin-top: 2px;">
                    Preference: <strong>Lower values generally indicate better cluster separation and compactness</strong>.
                </div>
            </div>
            
            <div style="margin-bottom: 0.85rem; padding-bottom: 0.65rem; border-bottom: 1px solid #F1F5F9;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-weight: 700; color: #0F172A; font-size: 0.92rem;">🎯 Silhouette Coefficient</span>
                    <span style="font-family: monospace; font-size: 1.1rem; font-weight: 800; color: #10B981;">{sil_disp}</span>
                </div>
                <div style="font-size: 0.76rem; color: #0D9488; font-weight: 600; margin-top: 2px;">
                    Preference: <strong>Higher values are preferred</strong> (Range: [-1, +1]).
                </div>
            </div>

            <div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-weight: 700; color: #0F172A; font-size: 0.92rem;">📐 Inertia (WCSS)</span>
                    <span style="font-family: monospace; font-size: 1.1rem; font-weight: 800; color: #7C3AED;">{inertia_disp}</span>
                </div>
                <div style="font-size: 0.76rem; color: #0D9488; font-weight: 600; margin-top: 2px;">
                    Preference: <strong>Lower values are preferred</strong> for a given number of clusters <em>k</em>.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Download Clustered Dataset
    clustered_df = input_df.copy()
    clustered_df["Assigned_Cluster"] = [f"Cluster {i}" for i in c_res["labels"]]
    clustered_df["Segment_Description"] = clustered_df["Assigned_Cluster"].map(c_res["semantic_labels"])
    buf = io.StringIO()
    clustered_df.to_csv(buf, index=False)

    st.download_button(
        label="📥 Download Clustered Dataset with Segment Labels (CSV)",
        data=buf.getvalue(),
        file_name=f"segmented_{input_file_name}",
        mime="text/csv",
        use_container_width=True
    )

    st.info("💡 **Next Steps:** Navigate using the sidebar to view **Cluster Insights & Profiles**, test **Cost-Sensitive Decision Support**, or explore **What-If & Counterfactual Recourse**.")

    st.markdown(render_disclaimer(
        "Unsupervised Learning Notice: Borrower segmentation discovers statistical similarities in financial attributes. "
        "Cluster assignments do not constitute credit approval, loan denial, or default risk predictions."
    ), unsafe_allow_html=True)
