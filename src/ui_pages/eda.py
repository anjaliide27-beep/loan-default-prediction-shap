"""
eda.py
------
Page 2: Comprehensive Exploratory Data Analysis (EDA) on LendingClub dataset.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import (
    clean_raw_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    IDENTIFIER_COLUMNS,
    FEATURE_LABELS
)
from src.ui_pages.styles import render_disclaimer

DATA_PATH = os.path.join("data", "loan_train.csv")


@st.cache_data
def load_eda_data():
    if not os.path.exists(DATA_PATH):
        return None
    raw_df = pd.read_csv(DATA_PATH, low_memory=False)
    cleaned_df = clean_raw_data(raw_df)
    return raw_df, cleaned_df


def render_eda_page():
    st.markdown('<div class="section-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown("""
    Explore empirical data distributions, missingness patterns, and bivariate relationships within the LendingClub dataset.
    *Notice: All statistics displayed here reflect historical exploratory data patterns prior to model training.*
    """)

    data_tuple = load_eda_data()
    if data_tuple is None:
        st.error(f"Dataset file `{DATA_PATH}` not found! Please check data placement.")
        return

    raw_df, cleaned_df = data_tuple

    # Tabbed navigation within EDA
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Preview & Schema",
        "❓ Missing Value Analysis",
        "📊 Target & Feature Distributions",
        "🔥 Correlation Heatmap",
        "💡 Key Empirical Insights"
    ])

    with tab1:
        st.subheader("Dataset Summary & Feature Categorization")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(raw_df):,}")
        c2.metric("Total Raw Columns", f"{raw_df.shape[1]}")
        c3.metric("Selected Application Features", f"{len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}")
        c4.metric("Excluded Leakage Columns", f"{len(LEAKAGE_COLUMNS)}")

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("**Selected Predictive Features (Application-Time):**")
            feat_df = pd.DataFrame({
                "Feature Name": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
                "Type": ["Numerical"] * len(NUMERICAL_FEATURES) + ["Categorical"] * len(CATEGORICAL_FEATURES),
                "Description": [FEATURE_LABELS.get(f, f) for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
            })
            st.dataframe(feat_df, use_container_width=True, height=280)

        with col_r:
            st.markdown("**Excluded Leakage & Identifier Columns:**")
            st.markdown("""
            To prevent **target leakage**, post-origination repayment metrics are strictly excluded:
            - **Identifiers:** `id`, `member_id`, `url`, `desc`, `emp_title`, `title`, `zip_code`
            - **Post-Loan Repayments:** `out_prncp`, `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `last_pymnt_d`, `last_pymnt_amnt`
            - **Collection / Recovery:** `recoveries`, `collection_recovery_fee`, `last_credit_pull_d`
            """)

        st.subheader("Raw Dataset Sample")
        st.dataframe(raw_df.head(20), use_container_width=True)

    with tab2:
        st.subheader("Missing Value Audit & Imputation Strategy")
        missing_counts = raw_df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing_counts.index,
            "Missing Count": missing_counts.values,
            "Missing Pct (%)": (missing_counts.values / len(raw_df) * 100).round(2)
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)

        fig_missing = px.bar(
            missing_df,
            x="Missing Pct (%)",
            y="Column",
            orientation="h",
            color="Missing Pct (%)",
            color_continuous_scale="Reds",
            title="Columns with Missing Values in Raw Dataset"
        )
        fig_missing.update_layout(yaxis=dict(autorange="reversed"), height=380, plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_missing, use_container_width=True)

        st.markdown("""
        **Imputation Design for Predictive Pipeline:**
        - `emp_length`: 730 missing records (2.7%) mapped to median tenure (4.0 years).
        - `revol_util`: 35 missing records (0.13%) median-imputed.
        - `pub_rec_bankruptcies`: 503 missing records (1.86%) median-imputed (0 bankruptcies).
        - Extremely sparse post-loan tracking fields like `mths_since_last_record` (92.8% missing) were discarded.
        """)

    with tab3:
        st.subheader("Feature Distributions by Loan Status")
        col_sel, col_stat = st.columns([1, 2])
        
        with col_sel:
            selected_num = st.selectbox(
                "Select Numerical Feature to Inspect:",
                options=[
                    'loan_amnt', 'int_rate_clean', 'annual_inc', 'dti',
                    'revol_bal', 'revol_util_clean', 'installment', 'open_acc'
                ],
                format_func=lambda x: FEATURE_LABELS.get(x, x)
            )
            plot_type = st.radio("Chart Type:", ["Box Plot", "Histogram / KDE"], horizontal=True)

        with col_stat:
            plot_df = cleaned_df[[selected_num, 'loan_status']].dropna().copy()
            plot_df['Loan Status'] = plot_df['loan_status'].map({0: 'Fully Paid (0)', 1: 'Default (1)'})

            if plot_type == "Box Plot":
                fig_dist = px.box(
                    plot_df,
                    x='Loan Status',
                    y=selected_num,
                    color='Loan Status',
                    color_discrete_map={'Fully Paid (0)': '#10B981', 'Default (1)': '#EF4444'},
                    title=f"Distribution of {FEATURE_LABELS.get(selected_num, selected_num)} by Loan Outcome"
                )
            else:
                fig_dist = px.histogram(
                    plot_df,
                    x=selected_num,
                    color='Loan Status',
                    barmode='overlay',
                    opacity=0.6,
                    color_discrete_map={'Fully Paid (0)': '#10B981', 'Default (1)': '#EF4444'},
                    title=f"Histogram of {FEATURE_LABELS.get(selected_num, selected_num)} by Loan Outcome"
                )

            fig_dist.update_layout(plot_bgcolor="#FFFFFF", height=380)
            st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Categorical Feature Proportions")
        cat_feature = st.selectbox(
            "Select Categorical Feature:",
            options=['grade', 'home_ownership_clean', 'verification_status', 'purpose'],
            format_func=lambda x: FEATURE_LABELS.get(x, x)
        )
        
        cat_grp = cleaned_df.groupby([cat_feature, 'loan_status']).size().unstack(fill_value=0)
        cat_grp.columns = ['Fully Paid', 'Default']
        cat_grp['Default Rate (%)'] = (cat_grp['Default'] / (cat_grp['Fully Paid'] + cat_grp['Default']) * 100).round(2)
        cat_grp = cat_grp.sort_values(by='Default Rate (%)', ascending=False).reset_index()

        fig_cat = px.bar(
            cat_grp,
            x=cat_feature,
            y='Default Rate (%)',
            text='Default Rate (%)',
            color='Default Rate (%)',
            color_continuous_scale='Tealgrn',
            title=f"Default Rate (%) across {FEATURE_LABELS.get(cat_feature, cat_feature)}"
        )
        fig_cat.update_layout(plot_bgcolor="#FFFFFF", height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

    with tab4:
        st.subheader("Correlation Matrix (Numerical Predictive Features)")
        corr_features = [
            'loan_amnt', 'term_months', 'int_rate_clean', 'installment',
            'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
            'open_acc', 'pub_rec', 'revol_bal', 'revol_util_clean', 'loan_status'
        ]
        corr_matrix = cleaned_df[corr_features].corr().round(2)
        corr_matrix.columns = [FEATURE_LABELS.get(c, c) for c in corr_features]
        corr_matrix.index = [FEATURE_LABELS.get(c, c) for c in corr_features]

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Correlation Heatmap of Application Features and Loan Status"
        )
        fig_corr.update_layout(height=550)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab5:
        st.subheader("Important Empirical Findings from Real Data")
        st.markdown("""
        1. **Interest Rate & Loan Grade Separation:**
           Borrowers assigned lower loan grades (E, F, G) and interest rates exceeding **15%** experience default rates above **25%**, compared to under **6%** for Grade A loans.
        2. **Debt-to-Income (DTI) Vulnerability:**
           Borrowers with DTI above **20%** show statistically higher probability of charge-off due to reduced liquidity cushions when minor financial shocks occur.
        3. **Inquiries in Past 6 Months:**
           Borrowers with multiple recent credit inquiries (3+) show significantly higher default propensity, reflecting credit hunger or distressed borrowing.
        4. **Class Imbalance Realities:**
           With only **14.73%** positive default cases, a naive model predicting 100% non-default would score 85.27% accuracy yet catch zero risky loans. Evaluation must prioritize **Recall and ROC-AUC**.
        """)

    st.markdown(render_disclaimer(
        "Exploratory statistics reflect sample correlations in historical lending data. "
        "Correlation does not imply causation, and past repayment trends do not guarantee future borrower performance."
    ), unsafe_allow_html=True)
