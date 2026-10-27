"""
whatif_cluster_page.py
----------------------
Page 6: Interactive What-If Scenario Analysis & Counterfactual Borrower Recommendations.
Adapts actionable recourse to K-Means clustering.
Applies the exact same fitted preprocessing scaler and KMeans model.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.clustering import run_kmeans_segmentation, predict_borrower_cluster
from src.cluster_counterfactuals import generate_cluster_counterfactuals, recalculate_installment
from src.ui_pages.styles import render_disclaimer, render_metric_card

DATA_PATH = os.path.join("data", "loan_train.csv")

# Baseline profile presets for demonstration
BORROWER_PRESETS = {
    "⚖️ Moderate-Debt Balanced Borrower": {
        'loan_amnt': 15000.0,
        'annual_inc': 52000.0,
        'int_rate_clean': 13.5,
        'installment': 509.0,
        'dti': 17.5,
        'emp_length_years': 4.0,
        'delinq_2yrs': 0.0,
        'inq_last_6mths': 1.0,
        'open_acc': 9.0,
        'revol_bal': 11500.0,
        'revol_util_clean': 58.0,
        'total_acc': 16.0,
        'term': '36 months'
    },
    "⚠️ High-Leverage Credit-Active Borrower": {
        'loan_amnt': 26000.0,
        'annual_inc': 42000.0,
        'int_rate_clean': 19.5,
        'installment': 682.0,
        'dti': 28.5,
        'emp_length_years': 2.0,
        'delinq_2yrs': 1.0,
        'inq_last_6mths': 3.0,
        'open_acc': 14.0,
        'revol_bal': 22000.0,
        'revol_util_clean': 82.0,
        'total_acc': 24.0,
        'term': '60 months'
    },
    "⭐ Lower-Debt High-Capacity Borrower": {
        'loan_amnt': 10000.0,
        'annual_inc': 95000.0,
        'int_rate_clean': 7.5,
        'installment': 311.0,
        'dti': 8.5,
        'emp_length_years': 10.0,
        'delinq_2yrs': 0.0,
        'inq_last_6mths': 0.0,
        'open_acc': 12.0,
        'revol_bal': 5200.0,
        'revol_util_clean': 22.0,
        'total_acc': 24.0,
        'term': '36 months'
    }
}


def render_whatif_cluster_page():
    st.markdown('<div class="section-header">🔄 What-If Analysis & Counterfactual Recourse</div>', unsafe_allow_html=True)
    st.markdown("""
    Explore how plausible adjustments to actionable financial terms—such as requesting a smaller loan 
    or paying down revolving debt to lower DTI—transition a borrower into a different K-Means segment. 
    **All scenario evaluations and counterfactual recommendations use the exact same fitted preprocessing scaler 
    and KMeans model instance.**
    """)

    # Ensure clustering results are active
    if "clustering_results" not in st.session_state:
        target_df = st.session_state.get("uploaded_df")
        if target_df is None:
            if not os.path.exists(DATA_PATH):
                st.error("Baseline dataset not found. Please upload a dataset.")
                return
            target_df = pd.read_csv(DATA_PATH, low_memory=False)

        with st.spinner("Initializing borrower segmentation pipeline..."):
            st.session_state["clustering_results"] = run_kmeans_segmentation(target_df, n_clusters=3)

    fitted_pipeline = st.session_state["clustering_results"]

    # 1. Select / Configure Baseline Borrower Profile
    st.markdown("### 1. Baseline Borrower Profile Selection")
    col_pre, col_pick = st.columns([1.5, 1.5])

    with col_pre:
        preset_choice = st.selectbox(
            "Load Preset Borrower Profile:",
            options=list(BORROWER_PRESETS.keys())
        )
        base_input = dict(BORROWER_PRESETS[preset_choice])

    with col_pick:
        uploaded_df = st.session_state.get("uploaded_df")
        if uploaded_df is not None and len(uploaded_df) > 0:
            st.caption("Or pick from uploaded dataset:")
            row_idx = st.selectbox(
                "Select Row from Uploaded Dataset:",
                options=list(range(min(20, len(uploaded_df)))),
                format_func=lambda i: f"Row #{i+1} (${uploaded_df.iloc[i].get('loan_amnt', 0):,.0f} loan, {uploaded_df.iloc[i].get('dti', 0)}% DTI)"
            )
            if st.button("📥 Load Selected Upload Row", use_container_width=True):
                base_input = uploaded_df.iloc[row_idx].to_dict()
                st.session_state["active_whatif_input"] = base_input

    if "active_whatif_input" in st.session_state:
        base_input = dict(st.session_state["active_whatif_input"])

    # Evaluate Baseline Borrower Cluster
    base_pred = predict_borrower_cluster(fitted_pipeline, base_input)
    current_c_name = base_pred["cluster_name"]
    current_semantic = base_pred["semantic_name"]

    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid #0F766E; border-radius: 8px; padding: 0.9rem 1.15rem; margin: 0.75rem 0 1.25rem 0;">
        <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">
            Current Cluster Assignment: <span style="color: #0F766E;">{current_c_name}</span> ({current_semantic})
        </div>
        <div style="font-size: 0.86rem; color: #475569; margin-top: 0.25rem;">
            Loan Amount: <strong>${base_input.get('loan_amnt', 0):,.0f}</strong> &nbsp;|&nbsp; 
            Annual Income: <strong>${base_input.get('annual_inc', 0):,.0f}</strong> &nbsp;|&nbsp; 
            DTI: <strong>{base_input.get('dti', 0):.1f}%</strong> &nbsp;|&nbsp; 
            Distance to Centroid: <code>{base_pred['distance_to_center']:.3f}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Manual What-If Scenario Exploration
    st.markdown("### 2. Manual What-If Scenario Exploration")
    st.caption("Adjust actionable financial attributes below to test cluster sensitivity in real time:")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        rev_loan = st.slider(
            "Revised Loan Amount ($):",
            min_value=1000,
            max_value=40000,
            value=int(base_input.get('loan_amnt', 15000)),
            step=1000,
            help=f"Baseline: ${base_input.get('loan_amnt', 0):,.0f}"
        )
    with col_s2:
        rev_dti = st.slider(
            "Revised Debt-to-Income (DTI %):",
            min_value=1.0,
            max_value=50.0,
            value=float(base_input.get('dti', 18.0)),
            step=0.5,
            help=f"Baseline: {base_input.get('dti', 0):.1f}%"
        )
    with col_s3:
        rev_inc = st.slider(
            "Revised Annual Income ($):",
            min_value=15000,
            max_value=250000,
            value=int(base_input.get('annual_inc', 55000)),
            step=5000,
            help=f"Baseline: ${base_input.get('annual_inc', 0):,.0f}"
        )

    # Build revised borrower dict with consistent derived ratios
    revised_input = dict(base_input)
    revised_input["loan_amnt"] = rev_loan
    revised_input["dti"] = rev_dti
    revised_input["annual_inc"] = rev_inc

    int_rate = float(str(base_input.get("int_rate_clean", base_input.get("int_rate", 12.5))).replace("%", ""))
    term_m = 36 if "36" in str(base_input.get("term", "36")) else 60
    revised_input["installment"] = recalculate_installment(rev_loan, int_rate, term_m)

    # Predict revised borrower
    rev_pred = predict_borrower_cluster(fitted_pipeline, revised_input)
    new_c_name = rev_pred["cluster_name"]
    new_semantic = rev_pred["semantic_name"]
    transitioned = (new_c_name != current_c_name)

    # Scenario Comparison Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card("Baseline Segment", current_c_name, current_semantic.split(": ")[-1], "#64748B"), unsafe_allow_html=True)
    with m2:
        tag_color = "#10B981" if transitioned else "#0EA5E9"
        st.markdown(render_metric_card("Revised Segment", new_c_name, new_semantic.split(": ")[-1], tag_color), unsafe_allow_html=True)
    with m3:
        status_txt = "Transitioned 🎉" if transitioned else "Unchanged"
        status_sub = "Moved to new segment" if transitioned else "Remains in current cluster"
        st.markdown(render_metric_card("Cluster Status", status_txt, status_sub, tag_color), unsafe_allow_html=True)
    with m4:
        shift_dist = float(np.linalg.norm(rev_pred["X_scaled"] - base_pred["X_scaled"]))
        st.markdown(render_metric_card("Normalized Shift (Δ)", f"{shift_dist:.3f}", "Euclidean distance", "#8B5CF6"), unsafe_allow_html=True)

    st.markdown("---")

    # 3. Automated Counterfactual Recommendations
    st.markdown("### 3. 🎯 Algorithmic Counterfactual Recourse Recommendations")
    st.markdown("""
    Instead of manual trial-and-error slider adjustment, the algorithmic recourse engine evaluates 
    plausible perturbations to actionable attributes to find the **smallest modifications** that transition 
    this borrower into a different segment.
    """)

    cf_results = generate_cluster_counterfactuals(
        borrower_input=base_input,
        fitted_pipeline=fitted_pipeline,
        max_recommendations=4
    )

    if not cf_results["success"]:
        st.error(f"Error generating counterfactuals: {cf_results.get('errors')}")
    elif cf_results["valid_count"] == 0:
        st.markdown(f"""
        <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 5px solid #D97706; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0;">
            <div style="font-weight: 700; color: #92400E; margin-bottom: 0.25rem;">
                ⚠️ No Feasible Counterfactual Within Plausible Bounds
            </div>
            <div style="font-size: 0.88rem; color: #78350F; line-height: 1.5;">
                {cf_results['message']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**Found {cf_results['valid_count']} feasible recourse options** to transition this borrower:")

        for i, rec in enumerate(cf_results["recommendations"], 1):
            col_r1, col_r2, col_r3 = st.columns([1.5, 1.3, 1.2])
            with col_r1:
                st.markdown(f"""
                <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">
                    Option {i}: {rec['feature_label']}
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-top: 0.15rem;">
                    {rec['change_desc']}
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #64748B;">Input Modification:</div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A;">
                    <code>{rec['original_val']}</code> ➔ <code style="color: #0F766E;">{rec['suggested_val']}</code>
                </div>
                """, unsafe_allow_html=True)
            with col_r3:
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #64748B;">Target Segment:</div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #0284C7;">
                    {rec['new_cluster_name']} (Δ = {rec['normalized_distance']})
                </div>
                <div style="font-size: 0.75rem; color: #64748B;">
                    {rec['new_semantic'].split(': ')[-1] if ': ' in rec['new_semantic'] else rec['new_semantic']}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<hr style='margin: 0.4rem 0 0.65rem 0; border: 0; border-top: 1px dashed #E2E8F0;'>", unsafe_allow_html=True)

    st.markdown(render_disclaimer(
        "Recourse & Cluster Transition Disclaimer: Counterfactual recommendations explore mathematical distance boundaries "
        "within the fitted K-Means feature space. Transitioning to a different cluster does not guarantee loan approval, "
        "interest rate reduction, or credit limit increase. Real-world lending decisions depend upon comprehensive underwriting validation."
    ), unsafe_allow_html=True)
