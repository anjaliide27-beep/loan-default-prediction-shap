"""
cluster_insights.py
-------------------
Page 4: Detailed Cluster Insights, Financial Distributions, and Profile Comparisons.
Provides deep-dive exploratory analytics into borrower segments discovered via K-Means.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.clustering import run_kmeans_segmentation
from src.ui_pages.styles import render_disclaimer, render_metric_card

DATA_PATH = os.path.join("data", "loan_train.csv")


def render_cluster_insights_page():
    st.markdown('<div class="section-header">📊 Cluster Insights & Segment Profiles</div>', unsafe_allow_html=True)
    st.markdown("""
    Explore detailed multi-dimensional financial distributions across borrower segments. 
    Analyze how loan requests, debt-to-income ratios, annual earnings, and revolving utilization 
    vary between clusters to inform credit policies and operational portfolio management.
    """)

    # Ensure clustering results are active
    if "clustering_results" not in st.session_state:
        target_df = st.session_state.get("uploaded_df")
        if target_df is None:
            if not os.path.exists(DATA_PATH):
                st.error("Baseline dataset not found. Please upload a dataset.")
                return
            target_df = pd.read_csv(DATA_PATH, low_memory=False)

        with st.spinner("Generating borrower segmentation profile..."):
            st.session_state["clustering_results"] = run_kmeans_segmentation(target_df, n_clusters=3)

    results = st.session_state["clustering_results"]
    n_clusters = results["n_clusters"]
    cluster_names = [f"Cluster {i}" for i in range(n_clusters)]
    semantic_labels = results.get("semantic_labels", {})

    st.markdown("### 1. Discovered Segment Personas")
    cols = st.columns(min(n_clusters, 4))
    cluster_colors = ["#0F766E", "#0284C7", "#D97706", "#7C3AED", "#DC2626", "#059669"]

    for i, c_name in enumerate(cluster_names):
        col_idx = i % len(cols)
        row_stats = results["profile_df"].loc[c_name] if c_name in results["profile_df"].index else {}
        b_count = int(results["distribution_df"].loc[results["distribution_df"]["Cluster"] == c_name, "Borrower_Count"].values[0])
        b_pct = float(results["distribution_df"].loc[results["distribution_df"]["Cluster"] == c_name, "Percentage"].values[0])

        with cols[col_idx]:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {cluster_colors[i]};">
                <div style="font-size: 0.82rem; font-weight: 700; color: #64748B;">{c_name}</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin: 0.25rem 0;">
                    {semantic_labels.get(c_name, c_name)}
                </div>
                <div style="font-size: 0.85rem; color: #475569;">
                    <strong>{b_count:,} borrowers</strong> ({b_pct}%)
                </div>
                <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #E2E8F0;">
                <div style="font-size: 0.78rem; color: #64748B; line-height: 1.4;">
                    • Avg Loan: <strong>${row_stats.get('loan_amnt', 0):,.0f}</strong><br>
                    • Avg Income: <strong>${row_stats.get('annual_inc', 0):,.0f}</strong><br>
                    • Avg DTI: <strong>{row_stats.get('dti', 0):.1f}%</strong><br>
                    • Avg Util: <strong>{row_stats.get('revol_util_clean', 0):.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature Distribution Deep Dive
    st.markdown("### 2. Financial Metric Distribution Comparison")
    st.markdown("Select a core financial metric to compare the underlying distribution across clusters:")

    metric_options = {
        "loan_amnt": "Loan Amount ($)",
        "annual_inc": "Annual Income ($)",
        "dti": "Debt-to-Income Ratio (DTI %)",
        "int_rate_clean": "Interest Rate (%)",
        "revol_util_clean": "Revolving Line Utilization (%)",
        "installment": "Monthly Installment ($)",
        "loan_to_income": "Loan-to-Income Ratio"
    }

    avail_metric_options = {k: v for k, v in metric_options.items() if k in results["pca_df"].columns or k in results["profile_df"].columns}
    selected_metric_key = st.selectbox(
        "Choose Financial Metric to Inspect:",
        options=list(avail_metric_options.keys()),
        format_func=lambda k: avail_metric_options[k]
    )

    # Box Plot Comparison
    plot_df = results["pca_df"].copy()
    if selected_metric_key in plot_df.columns:
        fig_box = px.box(
            plot_df,
            x="Cluster",
            y=selected_metric_key,
            color="Cluster",
            color_discrete_sequence=cluster_colors[:n_clusters],
            points="outliers",
            title=f"<b>Distribution of {avail_metric_options[selected_metric_key]} Across Clusters</b>"
        )
        fig_box.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            yaxis=dict(gridcolor="#F1F5F9"),
            height=380,
            showlegend=False,
            margin=dict(t=40, b=30, l=50, r=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Comparative Radar / Relative Feature Strength
    st.markdown("### 3. Relative Segment Profiles (Normalized Means)")
    st.markdown("Comparison of standardized average financial attributes across clusters (0 = cohort average, positive = above average):")

    profile_df = results["profile_df"].copy()
    # Normalize features relative to overall dataset mean for standardized radar/bar comparison
    norm_profile = (profile_df - profile_df.mean()) / (profile_df.std() + 1e-6)

    fig_heat = px.imshow(
        norm_profile.T,
        labels=dict(x="Borrower Cluster", y="Financial Feature", color="Z-Score"),
        x=norm_profile.index,
        y=norm_profile.columns,
        color_continuous_scale="Tealgrn",
        aspect="auto"
    )
    fig_heat.update_layout(
        height=380,
        margin=dict(t=30, b=30, l=80, r=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Strategic Portfolio Implications
    st.markdown("### 4. Strategic Portfolio Servicing Implications")
    st.markdown("""
    Based on the distinct financial leverage, capacity, and utilization traits of each segment, lenders can structure differentiated portfolio workflows:
    - **Low-Leverage / High-Income Segments:** Candidates for automated underwriting, preferred pricing, and pre-approved cross-selling.
    - **Balanced / Moderate-Debt Segments:** Standard servicing, periodic credit monitoring, and automated balance alerts.
    - **High-Leverage / High-Utilization Segments:** Priority monitoring, early intervention financial wellness resources, and structured debt advisory programs.
    """)

    st.markdown(render_disclaimer(
        "Advisory Notice: Cluster insights represent mathematical data segmentations. "
        "Lending policies and customer actions should comply with fair lending requirements and institutional credit mandates."
    ), unsafe_allow_html=True)
