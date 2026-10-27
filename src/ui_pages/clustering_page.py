"""
clustering_page.py
------------------
Interactive K-Means Borrower Segmentation Page.
Enables dynamic exploration of cluster counts (k=2 to 6), Davies–Bouldin and silhouette evaluation metrics,
and downloadable segmented cohorts.
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.clustering import run_kmeans_segmentation
from src.ui_pages.styles import render_disclaimer, render_metric_card

DATA_PATH = os.path.join("data", "loan_train.csv")


def render_clustering_page():
    st.markdown('<div class="section-header">👥 Borrower Segmentation (K-Means)</div>', unsafe_allow_html=True)
    st.markdown("""
    Segment borrowers into cohesive financial groups using **Unsupervised K-Means Clustering**. 
    The algorithm computes Euclidean distances across standardized financial features—such as loan amount, 
    debt-to-income ratio, annual income, and revolving utilization—**without relying on repayment outcome labels**.
    """)

    # Active Dataset Source Detection
    uploaded_df = st.session_state.get("uploaded_df")
    if uploaded_df is not None:
        source_name = st.session_state.get("uploaded_file_name", "Uploaded Dataset")
        st.info(f"📁 Analyzing active uploaded dataset: **{source_name}** ({len(uploaded_df):,} records).")
        target_df = uploaded_df
    else:
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset not found at `{DATA_PATH}`.")
            return
        st.caption("📁 Analyzing baseline LendingClub dataset (27,003 records). Upload a custom CSV via **Upload Dataset** to segment new cohorts.")
        target_df = pd.read_csv(DATA_PATH, low_memory=False)

    # Segmentation Controls
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
            <strong>Unsupervised Safeguards:</strong> <code>loan_status</code> and identifiers are strictly excluded. 
            All numerical features are standardized using <code>StandardScaler</code> to ensure equal distance weighting.
        </div>
        """, unsafe_allow_html=True)

    # Run K-Means Clustering
    with st.spinner(f"Clustering {len(target_df):,} records into {n_clusters} segments with K-Means..."):
        try:
            results = run_kmeans_segmentation(target_df, n_clusters=n_clusters)
            # Store in session state so downstream pages (Insights, Cost Support, What-If) use the exact same fitted pipeline
            st.session_state["clustering_results"] = results
        except Exception as e:
            st.error(f"Error executing K-Means clustering: {e}")
            return

    # Top KPI Metrics
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card(
            "Total Borrowers",
            f"{results['n_samples']:,}",
            "Analyzed Cohort"
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card(
            "Discovered Segments (k)",
            f"{results['n_clusters']}",
            f"{len(results['features_used'])} Standardized Features"
        ), unsafe_allow_html=True)
    with m3:
        db_score = results.get("davies_bouldin_index")
        db_text = f"{db_score:.3f}" if db_score is not None else "N/A"
        st.markdown(render_metric_card(
            "Davies–Bouldin Index",
            db_text,
            "Lower is preferred • Separation & Compactness",
            "#0284C7"
        ), unsafe_allow_html=True)
    with m4:
        sil_score = results.get("silhouette_score")
        sil_text = f"{sil_score:.3f}" if sil_score is not None else "N/A"
        st.markdown(render_metric_card(
            "Silhouette Score",
            sil_text,
            "Higher is preferred • [-1 to +1]",
            "#10B981"
        ), unsafe_allow_html=True)

    # Visualizations & Performance Metrics Section
    st.markdown("### 2. Segment Distribution & Clustering Performance Evaluation")
    col_vis1, col_vis2 = st.columns([1.1, 1.4])

    cluster_colors = ["#0F766E", "#0284C7", "#D97706", "#7C3AED", "#DC2626", "#059669"]

    with col_vis1:
        st.markdown("##### Borrower Distribution by Segment")
        dist_df = results["distribution_df"]
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
            height=370,
            margin=dict(t=20, b=30, l=40, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_vis2:
        st.markdown("##### Clustering Evaluation & Quality Metrics")
        
        db_val = results.get("davies_bouldin_index")
        sil_val = results.get("silhouette_score")
        inertia_val = results.get("inertia", 0.0)

        db_display = f"{db_val:.4f}" if db_val is not None else "Davies–Bouldin Index unavailable for this clustering configuration"
        sil_display = f"{sil_val:.4f}" if sil_val is not None else "Silhouette Score unavailable for this clustering configuration"
        inertia_display = f"{inertia_val:,.1f}"

        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.15rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid #F1F5F9;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">📊 Davies–Bouldin Index (DBI)</span>
                    <span style="font-family: monospace; font-size: 1.15rem; font-weight: 800; color: #0284C7;">{db_display}</span>
                </div>
                <div style="font-size: 0.78rem; color: #0D9488; font-weight: 600; margin-top: 3px;">
                    Preference: <strong>Lower values generally indicate better cluster separation and compactness</strong> (Minimum: 0).
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px; line-height: 1.4;">
                    Evaluates the similarity ratio between each cluster and its most similar neighbor, comparing intra-cluster spread against between-cluster centroid distances.
                </div>
            </div>
            
            <div style="margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid #F1F5F9;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">🎯 Silhouette Coefficient</span>
                    <span style="font-family: monospace; font-size: 1.15rem; font-weight: 800; color: #10B981;">{sil_display}</span>
                </div>
                <div style="font-size: 0.78rem; color: #0D9488; font-weight: 600; margin-top: 3px;">
                    Preference: <strong>Higher values are preferred</strong> (Theoretical range: [-1, +1]).
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px; line-height: 1.4;">
                    Quantifies how similar an individual borrower is to their assigned segment compared to points in neighboring segments.
                </div>
            </div>

            <div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">📐 Inertia (Within-Cluster Sum of Squares)</span>
                    <span style="font-family: monospace; font-size: 1.15rem; font-weight: 800; color: #7C3AED;">{inertia_display}</span>
                </div>
                <div style="font-size: 0.78rem; color: #0D9488; font-weight: 600; margin-top: 3px;">
                    Preference: <strong>Lower values are preferred</strong> for a given number of clusters <em>k</em>.
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px; line-height: 1.4;">
                    The cumulative sum of squared Euclidean distances from all borrower feature vectors to their respective cluster centroids.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Cluster Financial Profiles Table
    st.markdown("### 3. Cluster Financial Profile Comparison")
    st.markdown("Average financial characteristics across borrower segments:")

    profile_df = results["profile_df"].copy()
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

    for c in formatted_profile.columns:
        if '($)' in c:
            formatted_profile[c] = formatted_profile[c].apply(lambda x: f"${x:,.0f}")
        elif '(%)' in c or 'Ratio' in c:
            formatted_profile[c] = formatted_profile[c].apply(lambda x: f"{x:.2f}%" if '(%)' in c else f"{x:.2f}")
        else:
            formatted_profile[c] = formatted_profile[c].apply(lambda x: f"{x:.1f}")

    st.dataframe(formatted_profile, use_container_width=True)

    # Statistical Interpretations
    st.markdown("### 4. Segment Statistical Interpretations")
    for c_name, interpretation in results["interpretations"].items():
        st.markdown(f"- {interpretation}")

    # Post-Hoc Historical Outcome Check (Optional Descriptive Evaluation)
    if results["target_available"]:
        st.markdown("### 5. Post-Hoc Historical Outcome Comparison")
        st.markdown("""
        The table below displays the **historical observed default rate** within each cluster for descriptive verification. 
        """)
        eval_summary = results["distribution_df"][["Cluster", "Descriptive_Segment", "Borrower_Count", "Percentage", "Observed_Historical_Default_Rate (%)"]]
        st.dataframe(eval_summary.set_index("Cluster"), use_container_width=True)

        st.markdown(render_disclaimer(
            "Unsupervised Safeguards Notice: Historical loan outcomes (`loan_status`) were strictly excluded from the "
            "K-Means clustering algorithm. Observed default rates are reported post-hoc purely for descriptive validation "
            "to assess whether unsupervised financial clusters align naturally with real-world credit risk. "
            "Cluster labels are non-judgmental and do not constitute automatic credit approval or denial."
        ), unsafe_allow_html=True)

    # Download Clustered CSV
    st.markdown("### 6. Export Segmented Cohort")
    clustered_df = target_df.copy()
    clustered_df["Assigned_Cluster"] = [f"Cluster {i}" for i in results["labels"]]
    clustered_df["Segment_Description"] = clustered_df["Assigned_Cluster"].map(results["semantic_labels"])
    buf = io.StringIO()
    clustered_df.to_csv(buf, index=False)

    st.download_button(
        label="📥 Download Segmented Dataset with Cluster Assignments (CSV)",
        data=buf.getvalue(),
        file_name=f"kmeans_clustered_borrowers_k{n_clusters}.csv",
        mime="text/csv",
        use_container_width=True
    )
