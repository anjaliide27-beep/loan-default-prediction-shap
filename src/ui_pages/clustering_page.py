"""
clustering_page.py
------------------
UI Page: Borrower Segmentation using Unsupervised K-Means Clustering.
Discovers natural financial groupings across borrowers without labels.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.clustering import run_kmeans_segmentation
from src.ui_pages.styles import render_disclaimer, render_metric_card


@st.cache_data(show_spinner=False)
def load_and_cluster_dataset(data_path: str, n_clusters: int):
    """Loads dataset and runs K-Means segmentation with caching."""
    df = pd.read_csv(data_path, low_memory=False)
    results = run_kmeans_segmentation(df, n_clusters=n_clusters)
    return results


def render_clustering_page():
    st.markdown('<div class="section-header">👥 Borrower Segmentation (K-Means)</div>', unsafe_allow_html=True)
    st.markdown("""
    Explore unsupervised financial segmentation of borrowers using **K-Means Clustering**. 
    This technique identifies natural groupings based entirely on borrower financial behaviors, debt levels, 
    and credit attributes—**without using repayment labels (`loan_status`) or loan outcome data**.
    """)

    # Sidebar / Top parameter controls
    st.markdown("### 1. Segmentation Controls")
    col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])

    with col_ctrl1:
        n_clusters = st.slider(
            "Select Number of Clusters (k):",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            help="Choose the number of distinct borrower segments to identify (default: 3)."
        )

    with col_ctrl2:
        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #475569; margin-top: 0.25rem;">
            <strong>Unsupervised Safeguards:</strong> <code>loan_status</code> and all post-issuance leakage features 
            have been strictly isolated. All numerical attributes are standardized with <code>StandardScaler</code>.
        </div>
        """, unsafe_allow_html=True)

    # Load dataset & run clustering (Active uploaded dataset or default dataset)
    uploaded_df = st.session_state.get("uploaded_df")
    if uploaded_df is not None:
        source_name = st.session_state.get("uploaded_file_name", "Uploaded Dataset")
        st.info(f"📁 Analyzing active uploaded dataset: **{source_name}** ({len(uploaded_df):,} records).")
        target_df = uploaded_df
    else:
        data_path = os.path.join("data", "loan_train.csv")
        if not os.path.exists(data_path):
            st.error(f"Dataset not found at `{data_path}`. Please verify the dataset location.")
            return
        st.caption("Analyzing baseline LendingClub dataset (27,003 records). Upload a custom CSV via **Upload Dataset** to segment new cohorts.")
        target_df = pd.read_csv(data_path, low_memory=False)

    with st.spinner(f"Clustering {len(target_df):,} borrower records into {n_clusters} segments with K-Means..."):
        try:
            results = run_kmeans_segmentation(target_df, n_clusters=n_clusters)
        except Exception as e:
            st.error(f"Error executing K-Means clustering: {e}")
            return

    # Top KPI Metrics
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card(
            "Total Borrowers Analyzed",
            f"{results['n_samples']:,}",
            "LendingClub Dataset"
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card(
            "Configured Clusters (k)",
            f"{results['n_clusters']}",
            f"{len(results['features_used'])} Financial Features"
        ), unsafe_allow_html=True)
    with m3:
        sil_score = results["silhouette_score"]
        sil_text = f"{sil_score:.3f}" if sil_score is not None else "N/A"
        st.markdown(render_metric_card(
            "Silhouette Score",
            sil_text,
            "Cluster cohesion & separation"
        ), unsafe_allow_html=True)
    with m4:
        pca_var = sum(results["pca_explained_variance"]) * 100.0
        st.markdown(render_metric_card(
            "2D PCA Explained Variance",
            f"{pca_var:.1f}%",
            f"PC1: {results['pca_explained_variance'][0]*100:.1f}%, PC2: {results['pca_explained_variance'][1]*100:.1f}%"
        ), unsafe_allow_html=True)

    # Visualizations Section
    st.markdown("### 2. Segment Distribution & 2D Projection")
    col_vis1, col_vis2 = st.columns([1.1, 1.4])

    # Cluster Size Distribution Bar Chart
    with col_vis1:
        st.markdown("##### Borrower Count per Cluster")
        dist_df = results["distribution_df"]
        
        # Color palette for clusters
        cluster_colors = ["#0F766E", "#0284C7", "#D97706", "#7C3AED", "#DC2626", "#059669"]
        
        fig_bar = px.bar(
            dist_df,
            x="Cluster",
            y="Borrower_Count",
            text=dist_df.apply(lambda r: f"{r['Borrower_Count']:,} ({r['Percentage']}%)", axis=1),
            color="Cluster",
            color_discrete_sequence=cluster_colors[:n_clusters]
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            yaxis=dict(title="Borrower Count", gridcolor="#F1F5F9"),
            xaxis=dict(title="Cluster"),
            height=380,
            margin=dict(t=20, b=30, l=40, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2D PCA Scatter Plot
    with col_vis2:
        st.markdown("##### 2D Principal Component Projection (PCA)")
        pca_df = results["pca_df"]
        
        # Sample for fast, crisp scatter rendering if large
        plot_df = pca_df.sample(min(3500, len(pca_df)), random_state=42)
        
        fig_scatter = px.scatter(
            plot_df,
            x="PCA_1",
            y="PCA_2",
            color="Cluster",
            color_discrete_sequence=cluster_colors[:n_clusters],
            hover_data={
                "PCA_1": False,
                "PCA_2": False,
                "Cluster": True,
                "loan_amnt": ":$,.0f",
                "annual_inc": ":$,.0f",
                "int_rate_clean": ":.1f%",
                "dti": ":.1f%"
            },
            opacity=0.7
        )
        fig_scatter.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(title=f"Principal Component 1 ({results['pca_explained_variance'][0]*100:.1f}% var)", gridcolor="#F1F5F9"),
            yaxis=dict(title=f"Principal Component 2 ({results['pca_explained_variance'][1]*100:.1f}% var)", gridcolor="#F1F5F9"),
            height=380,
            margin=dict(t=20, b=30, l=40, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Cluster Financial Profiles Table
    st.markdown("### 3. Cluster Financial Profile Comparison")
    st.markdown("Average financial indicators by borrower segment:")

    profile_df = results["profile_df"].copy()

    # Create formatted display dataframe with readable column names
    display_cols_map = {
        'loan_amnt': 'Avg Loan Amount ($)',
        'annual_inc': 'Avg Annual Income ($)',
        'int_rate_clean': 'Avg Interest Rate (%)',
        'installment': 'Avg Monthly Installment ($)',
        'dti': 'Avg Debt-to-Income (DTI %)',
        'revol_util_clean': 'Avg Revolving Util (%)',
        'loan_to_income': 'Loan-to-Income Ratio',
        'open_acc': 'Avg Open Accounts',
        'delinq_2yrs': 'Avg Delinquencies (2Y)'
    }
    present_cols = [c for c in display_cols_map.keys() if c in profile_df.columns]
    formatted_profile = profile_df[present_cols].rename(columns=display_cols_map)

    # Format values cleanly
    for c in formatted_profile.columns:
        if '($)' in c:
            formatted_profile[c] = formatted_profile[c].apply(lambda x: f"${x:,.0f}")
        elif '(%)' in c or 'Ratio' in c:
            formatted_profile[c] = formatted_profile[c].apply(lambda x: f"{x:.2f}%" if '(%)' in c else f"{x:.2f}")
        else:
            formatted_profile[c] = formatted_profile[c].apply(lambda x: f"{x:.1f}")

    st.dataframe(formatted_profile, use_container_width=True)

    # Natural Language Interpretations
    st.markdown("### 4. Segment Statistical Interpretations")
    for c_name, interpretation in results["interpretations"].items():
        st.markdown(f"- {interpretation}")

    # Post-Hoc Historical Default Rate Evaluation (Optional Analysis)
    if results["target_available"]:
        st.markdown("### 5. Observed Post-Hoc Default Rate Comparison")
        st.markdown("""
        The table below displays the **historical observed default rate** within each cluster. 
        """)

        eval_summary = results["distribution_df"][["Cluster", "Borrower_Count", "Percentage", "Observed_Default_Rate (%)"]]
        st.dataframe(eval_summary.set_index("Cluster"), use_container_width=True)

        st.markdown(render_disclaimer(
            "Important Unsupervised Learning Notice: The historical loan outcome (`loan_status`) was strictly "
            "excluded from the K-Means clustering algorithm. Observed default rates are reported post-hoc solely to evaluate "
            "whether unsupervised financial segmentation correlates naturally with credit risk outcomes. "
            "Cluster labels (Cluster 0, Cluster 1, ...) are neutral descriptions and should not be used as automatic lending decisions."
        ), unsafe_allow_html=True)
