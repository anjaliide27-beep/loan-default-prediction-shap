"""
about_page.py
-------------
Page 7: Comprehensive Project Documentation, Methodology, and Architectural Guide.
Details the K-Means clustering pipeline, feature boundaries, cost support methodology,
counterfactual recourse logic, and regulatory safeguards.
"""

import streamlit as st
from src.ui_pages.styles import render_disclaimer, render_metric_card


def render_about_page():
    st.markdown('<div class="section-header">ℹ️ About the Project: Unsupervised Borrower Segmentation</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Executive Overview
    This project demonstrates the application of **Unsupervised Machine Learning (K-Means Clustering)** 
    to segment loan applicants based on multi-dimensional financial health and leverage attributes. 
    By operating in a purely unsupervised setting, the system avoids relying on historical default labels 
    (`loan_status`) that may encode past institutional bias or macroeconomic distortions.
    """)

    # Three Pillar Cards
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">👥</div>
            <div style="font-weight: 700; color: #0F172A; font-size: 1rem; margin-bottom: 0.25rem;">K-Means Segmentation</div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.45;">
                Discovers natural clusters across standardized financial indicators with configurable cluster counts (k=2 to 6).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">💰</div>
            <div style="font-weight: 700; color: #0F172A; font-size: 1rem; margin-bottom: 0.25rem;">Cost Decision Support</div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.45;">
                Simulates operational portfolio management budgets by comparing segment-tailored action costs against uniform servicing.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🔄</div>
            <div style="font-weight: 700; color: #0F172A; font-size: 1rem; margin-bottom: 0.25rem;">Counterfactual Recourse</div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.45;">
                Explores minimal, feasible financial perturbations (loan size, DTI) to transition borrowers into different clusters.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Methodology Section
    st.markdown("### Methodology & Algorithmic Pipeline")
    st.markdown("""
    1. **Data Cleaning & Ratio Engineering:**
       - Formats terms into numerical months (`term_months`).
       - Strips percentage symbols from interest rates (`int_rate_clean`) and revolving utilization (`revol_util_clean`).
       - Synthesizes core financial leverage ratios:
         - $\\text{Loan-to-Income} = \\frac{\\text{loan\\_amnt}}{\\text{annual\\_inc} + 1}$
         - $\\text{Installment-to-Income} = \\frac{\\text{installment} \\times 12}{\\text{annual\\_inc} + 1}$
         - $\\text{Revolving-to-Income} = \\frac{\\text{revol\\_bal}}{\\text{annual\\_inc} + 1}$
    2. **Strict Leakage & Identifier Isolation:**
       - All post-origination loan performance attributes (`total_pymnt`, `recoveries`, `last_pymnt_amnt`, etc.) and borrower identifiers (`id`, `url`, `zip_code`) are **strictly excluded**.
       - Outcome labels (`loan_status`) are strictly excluded from clustering inputs, pipeline fitting, and cluster assignments.
    3. **Standardization & Distance Metric:**
       - Standardizes all features using `StandardScaler` (zero mean, unit variance) so that high-magnitude attributes (e.g. annual income) do not overwhelm scale-sensitive attributes (e.g. DTI or interest rate).
       - Imputes missing values with median imputation.
    4. **K-Means Optimization:**
       - Partitions records by minimizing within-cluster sum of squares (inertia):
         $$\\arg\\min_{\\mathbf{S}} \\sum_{i=1}^{k} \\sum_{\\mathbf{x} \\in S_i} \\|\\mathbf{x} - \\boldsymbol{\\mu}_i\\|^2$$
       - Uses `n_init=10` and `random_state=42` to guarantee deterministic, reproducible assignments.
    5. **Silhouette Cohesion Scoring:**
       - Evaluates cluster quality using the silhouette coefficient:
         $$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$
         where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance.
    6. **Davies–Bouldin Index (DBI):**
       - Evaluates the average similarity between each cluster and its most similar counterpart:
         $$DB = \\frac{1}{k} \\sum_{i=1}^{k} \\max_{j \\neq i} \\left( \\frac{s_i + s_j}{d(\\boldsymbol{\\mu}_i, \\boldsymbol{\\mu}_j)} \\right)$$
         where $s_i$ is the average distance from points in cluster $i$ to centroid $\\boldsymbol{\\mu}_i$, and $d(\\boldsymbol{\\mu}_i, \\boldsymbol{\\mu}_j)$ is the centroid separation.
       - **Lower values** generally indicate better cluster separation and compactness (bounded at minimum 0).
    """)

    st.markdown("---")

    # Feature Catalog
    st.markdown("### Financial Feature Catalog")
    st.markdown("""
    | Category | Features Included | Rationale |
    | :--- | :--- | :--- |
    | **Loan Specifications** | `loan_amnt`, `installment`, `int_rate_clean` | Defines immediate borrower credit obligations |
    | **Income & Capacity** | `annual_inc`, `loan_to_income`, `installment_to_income` | Quantifies debt servicing capacity relative to earnings |
    | **Leverage & Debt** | `dti`, `revol_bal`, `revol_util_clean`, `revol_to_income` | Measures existing credit burden and line utilization |
    | **Credit History** | `open_acc`, `total_acc`, `delinq_2yrs`, `inq_last_6mths` | Captures historical credit activity (strictly immutable) |
    | **Strictly Excluded** | `loan_status`, `id`, `recoveries`, `total_pymnt` | Avoids target leakage and preserves pure unsupervised integrity |
    """)

    st.markdown("---")

    # Cost-Sensitive Decision Support Explanation
    st.markdown("### How Cost-Sensitive Decision Support Works")
    st.markdown("""
    Unlike supervised learning (which balances false-positive vs. false-negative error costs), **unsupervised cost-sensitive decision support** 
    evaluates the **operational expenditure of portfolio servicing policies**:
    - **Uniform Policy:** Assumes every borrower receives identical account reviews ($C_{\\text{uniform}} = \\$120$/borrower).
    - **Segment-Tailored Policy:** Dynamically allocates lower operational costs to low-debt, high-capacity segments (e.g. automated low-touch servicing at $\\$35$/borrower) and reserves intensive review and counseling for high-leverage segments ($\\$320$/borrower).
    - **Total Expenditure Calculation:**
      $$\\text{Total Portfolio Cost} = \\sum_{k=0}^{K-1} N_k \\times C_k$$
      where $N_k$ is the number of borrowers in segment $k$ and $C_k$ is the action cost.
    """)

    st.markdown(render_disclaimer(
        "Scientific & Regulatory Compliance Notice: This application is an unsupervised exploratory data analysis system. "
        "It does not perform binary loan default prediction, does not generate credit approval/denial scores, "
        "and does not provide formal legal or financial advice. All scenario evaluations are mathematical simulations."
    ), unsafe_allow_html=True)
