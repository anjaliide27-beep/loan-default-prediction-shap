"""
home.py
-------
Page 1: Executive Dashboard for Unsupervised Borrower Segmentation Using K-Means Clustering.
Presents project objectives, financial feature architecture, and interactive portfolio overview.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui_pages.styles import render_metric_card, render_disclaimer
from src.clustering import run_kmeans_segmentation, CANDIDATE_CLUSTERING_FEATURES

DATA_PATH = os.path.join("data", "loan_train.csv")


def render_home_page():
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; gap: 8px; margin-bottom: 12px;">
            <span class="badge badge-teal">Unsupervised Machine Learning</span>
            <span class="badge" style="background: rgba(255,255,255,0.15); color: #fff;">K-Means & Portfolio Analytics</span>
        </div>
        <div class="hero-title">Borrower Segmentation Using K-Means Clustering</div>
        <div class="hero-sub">
            Discover natural financial patterns, debt behaviors, and credit profiles across loan applicants 
            without relying on historical repayment labels. Features cost-sensitive operational decision support 
            and counterfactual recourse recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Cards Row
    total_records = 27003
    num_features = 15

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Analyzed Cohort", f"{total_records:,}", "LendingClub records", "#0EA5E9"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Financial Features", f"{num_features}", "Standardized with StandardScaler", "#10B981"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Cluster Capacity", "2 to 6", "Default k = 3 segments", "#F59E0B"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Supervision Required", "0%", "Strictly label-free K-Means", "#8B5CF6"), unsafe_allow_html=True)

    # Project Overview & Architecture
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.markdown('<div class="section-header">Project Rationale & Methodology</div>', unsafe_allow_html=True)
        st.markdown(r"""
        In modern credit portfolio management, lenders require objective ways to categorize borrowers based on 
        multi-dimensional financial health—without solely relying on backward-looking default labels that may reflect 
        historical underwriting bias.

        **Core Architecture & Features:**
        - **Label-Free Feature Engineering:** Eliminates post-loan leakage variables and isolates target columns (`loan_status`). Standardizes 15 core financial attributes using `StandardScaler`.
        - **K-Means Clustering:** Identifies cohesive borrower segments with configurable cluster counts ($k \in [2, 6]$) and fixed random seeds for strict reproducibility.
        - **2D Dimensionality Reduction (PCA):** Projects multi-dimensional standardized feature space into 2 principal components for clear visual separation.
        - **Cost-Sensitive Decision Support:** Translates cluster distributions into actionable operational servicing strategies and compares estimated portfolio expenditures.
        - **Actionable Counterfactual Recourse:** Identifies minimal plausible adjustments to actionable attributes (loan amount, DTI) that transition a borrower into a different cluster.
        """)

    with col_right:
        st.markdown('<div class="section-header">Feature Space Composition</div>', unsafe_allow_html=True)
        feature_categories = {
            "Debt & Leverage (DTI, Util, Revolving)": 5,
            "Income & Capacity (Annual Inc, Ratios)": 4,
            "Loan Terms (Amount, Rate, Installment)": 4,
            "Credit History (Open/Total Accounts, Delinq)": 2
        }
        fig_donut = go.Figure(data=[go.Pie(
            labels=list(feature_categories.keys()),
            values=list(feature_categories.values()),
            hole=0.55,
            marker_colors=['#0F766E', '#0284C7', '#D97706', '#7C3AED'],
            textinfo='label+percent',
            hoverinfo='label+value'
        )])
        fig_donut.update_layout(
            height=320,
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Quick Navigation Cards
    st.markdown("---")
    st.markdown('<div class="section-header">Interactive Exploration Modules</div>', unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">📁</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Upload Dataset</div>
            <div style="font-size: 0.8rem; color: #64748B;">Upload custom CSV datasets to cluster new borrower cohorts without outcome labels.</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">👥</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">K-Means Clustering</div>
            <div style="font-size: 0.8rem; color: #64748B;">Explore 2–6 clusters, silhouette separation scores, and 2D PCA projections.</div>
        </div>
        """, unsafe_allow_html=True)
    with q3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">💰</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Cost Decision Support</div>
            <div style="font-size: 0.8rem; color: #64748B;">Compare operational servicing strategies and optimize portfolio expenditures.</div>
        </div>
        """, unsafe_allow_html=True)
    with q4:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">🔄</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">What-If Recourse</div>
            <div style="font-size: 0.8rem; color: #64748B;">Test plausible financial perturbations to transition borrowers between segments.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(render_disclaimer(
        "Scientific & Regulatory Notice: Unsupervised K-Means clustering groups applicants based on mathematical similarity "
        "across financial dimensions. Clusters do not predict individual loan defaults or guarantee credit approvals. "
        "Operational cost estimates are user-defined simulations for portfolio resource allocation."
    ), unsafe_allow_html=True)
