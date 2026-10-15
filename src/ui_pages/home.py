"""
home.py
-------
Page 1: Professional Home Dashboard for Explainable Loan Default Prediction.
"""

import json
import os
import streamlit as st
import plotly.graph_objects as go
from src.ui_pages.styles import render_metric_card, render_disclaimer

MODELS_DIR = "models"


def render_home_page():
    # Load metadata and metrics
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    has_artifacts = os.path.exists(metadata_path) and os.path.exists(metrics_path)

    if has_artifacts:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        total_rows = metadata.get("total_dataset_rows", 27003)
        total_cols = metadata.get("total_dataset_cols", 47)
        num_features = metadata.get("selected_feature_count", 19)
        dist = metadata.get("target_distribution", {"fully_paid_0": 23026, "default_1": 3977})
        paid_cnt = dist.get("fully_paid_0", 23026)
        def_cnt = dist.get("default_1", 3977)
        def_rate = def_cnt / total_rows * 100
        best_auc = max([m.get("roc_auc", 0) for m in metrics.values()])
    else:
        total_rows = 27003
        total_cols = 47
        num_features = 19
        paid_cnt = 23026
        def_cnt = 3977
        def_rate = 14.73
        best_auc = 0.7103

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; gap: 8px; margin-bottom: 12px;">
            <span class="badge badge-teal">AIML Capstone Project</span>
            <span class="badge" style="background: rgba(255,255,255,0.15); color: #fff;">XAI & Credit Scoring</span>
        </div>
        <div class="hero-title">Explainable Loan Default Prediction</div>
        <div class="hero-sub">
            Empowering transparent, fair, and interpretable credit risk assessment using supervised machine learning 
            and SHapley Additive exPlanations (SHAP) on historical LendingClub records.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Dataset Size", f"{total_rows:,}", f"{total_cols} raw columns", "#0EA5E9"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Application Features", f"{num_features}", "0 post-origination leakage", "#10B981"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Historical Defaults", f"{def_cnt:,}", f"{def_rate:.1f}% class imbalance", "#EF4444"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Peak Model ROC-AUC", f"{best_auc:.3f}", "Logistic Reg & Random Forest", "#8B5CF6"), unsafe_allow_html=True)

    # Project Overview & Class Distribution
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.markdown('<div class="section-header">Problem Statement & Objectives</div>', unsafe_allow_html=True)
        st.markdown("""
        Traditional credit underwriting models rely heavily on black-box scoring or inflexible rule sets. When an applicant is denied or charged higher interest, conventional machine learning models cannot provide justifiable, transparent adverse action reasons required by financial regulations (such as the Equal Credit Opportunity Act).

        **Core Project Objectives:**
        - **Rigorous Data Preprocessing:** Eliminate post-loan leakage columns (`recoveries`, `total_pymnt`, `last_pymnt_amnt`) to ensure model uses only application-time information.
        - **Supervised Machine Learning:** Train and compare Logistic Regression, Decision Trees, Random Forests, and XGBoost with explicit class-imbalance compensation.
        - **Local & Global Explainability:** Implement SHAP to quantify exact feature attributions for individual applicant predictions and broad risk trends.
        - **Interactive Risk Management:** Provide scenario simulation (*What-If Analysis*) and configurable risk thresholds for informed, responsible decision support.
        """)

    with col_right:
        st.markdown('<div class="section-header">Target Distribution</div>', unsafe_allow_html=True)
        # Plotly Donut Chart
        labels = ['Fully Paid (Non-Default)', 'Charged Off (Default)']
        values = [paid_cnt, def_cnt]
        colors = ['#10B981', '#EF4444']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.55,
            marker_colors=colors,
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )])
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            height=260,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Class imbalance ratio: ~5.8:1. Managed through balanced class weighting.")

    # Overview of Algorithms
    st.markdown('<div class="section-header">Machine Learning Architecture</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 4px;">Logistic Regression</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">Linear Baseline</div>
            <div style="font-size: 0.82rem; color: #334155;">
                Interpretable linear benchmark with balanced class weighting. Serves as reference for non-linear models.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 4px;">Decision Tree</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">Rule-Based Splits</div>
            <div style="font-size: 0.82rem; color: #334155;">
                Pruned tree with max depth 6 to prevent overfitting while capturing hierarchical credit feature interactions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #0EA5E9;">
            <div style="font-weight: 700; color: #0EA5E9; margin-bottom: 4px;">Random Forest ★</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">Ensemble Averaging</div>
            <div style="font-size: 0.82rem; color: #334155;">
                100 bagged trees mitigating individual variance. High stability for continuous SHAP attribution.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 4px;">XGBoost</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">Gradient Boosting</div>
            <div style="font-size: 0.82rem; color: #334155;">
                Gradient-boosted decision trees optimizing binary logloss with positive scale weights.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Guided Quick Links
    st.markdown('<div class="section-header">Dashboard Navigation</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        **1. Exploratory Data Analysis**  
        Inspect feature correlations, distribution skews, missingness patterns, and target separation.
        """)
    with g2:
        st.markdown("""
        **2. Interactive Prediction & SHAP**  
        Input borrower credit profiles, evaluate risk tiers, and generate local Shapley waterfall breakdowns.
        """)
    with g3:
        st.markdown("""
        **3. What-If & Model Comparison**  
        Simulate counterfactual scenarios and compare held-out test evaluation metrics across all 4 models.
        """)

    st.markdown(render_disclaimer(
        "This system is an academic machine learning prototype trained on historical LendingClub US records. "
        "It is not intended as an automated credit underwriting tool for real-world borrowers and is not certified "
        "for regulatory lending decisions or Indian microfinance borrowers."
    ), unsafe_allow_html=True)
