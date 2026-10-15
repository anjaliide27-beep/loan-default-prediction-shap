"""
shap_page.py
------------
Page 4: Explainable AI with SHAP (Local & Global Feature Attributions).
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.prediction import predict_loan_default
from src.explainability import (
    load_explainer_artifacts,
    compute_instance_shap,
    create_local_waterfall_plot,
    create_local_bar_chart,
    create_global_importance_chart,
    generate_plain_language_summary
)
from src.ui_pages.styles import render_disclaimer
from src.ui_pages.prediction_page import PRESETS


def render_shap_page():
    st.markdown('<div class="section-header">Explainable AI with SHAP (SHapley Additive exPlanations)</div>', unsafe_allow_html=True)
    st.markdown("""
    Explainable AI (XAI) bridges the gap between complex statistical models and human comprehension. 
    Using **Shapley values from cooperative game theory**, this page reveals exactly which applicant features drove 
    the model's prediction higher or lower.
    """)

    # Check if a prediction exists in session state; if not, use default sample
    if "current_prediction" not in st.session_state:
        st.info("ℹ️ No active applicant prediction found. Automatically loading **Average Borrower Profile** for demonstration.")
        sample_profile = PRESETS["⚖️ Average Borrower (Moderate Risk Profile)"]
        pred_res = predict_loan_default(sample_profile, model_name="Random Forest")
        st.session_state["current_prediction"] = pred_res
        st.session_state["current_applicant_input"] = sample_profile
        st.session_state["selected_model"] = "Random Forest"

    pred_res = st.session_state["current_prediction"]
    app_input = st.session_state.get("current_applicant_input", {})
    selected_model = st.session_state.get("selected_model", "Random Forest")

    # Load SHAP explainer
    try:
        explainer, metadata, test_sample = load_explainer_artifacts()
    except Exception as e:
        st.error(f"Error loading SHAP artifacts: {e}")
        return

    # Compute SHAP for current prediction
    shap_info = compute_instance_shap(pred_res["X_trans"], explainer=explainer, model_name=selected_model)

    # Top summary banner
    proba_pct = pred_res["probability"] * 100.0
    base_val_pct = shap_info["base_value"] * 100.0
    tier = pred_res["risk_tier"]
    badge_col = pred_res["risk_info"]["badge_color"]

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Base Population Risk", f"{base_val_pct:.1f}%", help="Prior default rate before considering applicant features")
    with s2:
        st.metric("Applicant Predicted Risk", f"{proba_pct:.1f}%", f"{proba_pct - base_val_pct:+.1f}% vs baseline", delta_color="inverse")
    with s3:
        st.metric("Assigned Risk Tier", tier)
    with s4:
        st.metric("Explained Model", selected_model)

    tab_local, tab_global, tab_theory = st.tabs([
        "🔍 Local Applicant Explanation",
        "🌐 Global Feature Importance",
        "📐 SHAP Theory & Compliance"
    ])

    with tab_local:
        st.subheader("Interactive Waterfall Attribution")
        st.caption(
            "The waterfall chart illustrates the step-by-step journey from the base rate to the final probability. "
            "Red bars push risk **higher** (increasing default likelihood); Green bars push risk **lower** (decreasing default likelihood)."
        )

        fig_waterfall = create_local_waterfall_plot(shap_info, max_features=8)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        col_l, col_r = st.columns([1.2, 0.8])
        with col_l:
            st.subheader("Top Divergent Factor Contributions")
            fig_bar = create_local_bar_chart(shap_info, top_n=10)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_r:
            st.subheader("Plain-Language Summary")
            summary_text = generate_plain_language_summary(shap_info, app_input)
            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem; font-size: 0.9rem; line-height: 1.6;">
                {summary_text}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Key Raw Input Values")
            raw_display = {
                "Loan Amount": f"${app_input.get('loan_amnt', 0):,}",
                "Interest Rate": f"{app_input.get('int_rate', 0)}%",
                "Annual Income": f"${app_input.get('annual_inc', 0):,}",
                "DTI Ratio": f"{app_input.get('dti', 0)}%",
                "Grade": app_input.get('grade', 'N/A'),
                "Home Ownership": app_input.get('home_ownership', 'N/A')
            }
            st.json(raw_display)

        st.subheader("Comprehensive Feature Contribution Table")
        df_table = shap_info["shap_df"].copy()
        df_table["Impact Direction"] = np.where(df_table["shap_value"] > 0, "Increases Default Risk (Red)", "Decreases Default Risk (Green)")
        df_table["SHAP Contribution"] = df_table["shap_value"].apply(lambda v: f"{v:+.4f}")
        st.dataframe(
            df_table[["feature", "Impact Direction", "SHAP Contribution"]].rename(columns={"feature": "Feature Name"}),
            use_container_width=True,
            height=300
        )

    with tab_global:
        st.subheader("Global Feature Importance (Held-Out Test Sample)")
        st.markdown("""
        While individual waterfall plots explain one borrower at a time, **Global Feature Importance** computes 
        the mean absolute SHAP value across a representative sample of test loans.
        This reveals what the model prioritizes across all decisions.
        """)
        fig_global = create_global_importance_chart(metadata, top_n=15)
        st.plotly_chart(fig_global, use_container_width=True)

        st.markdown("""
        **Top Global Determinants of Default in LendingClub:**
        1. **Interest Rate & Loan Grade:** The strongest leading indicators of risk assigned by initial screening.
        2. **Debt-to-Income (DTI):** High leverage relative to income consistently drives higher charge-off probabilities.
        3. **Annual Income & Loan Amount:** The scale of the obligation relative to total cash flow is paramount.
        4. **Credit Inquiries in Past 6 Months:** Signals urgent liquidity needs or distress borrowing.
        """)

    with tab_theory:
        st.subheader("Understanding SHAP (SHapley Additive exPlanations)")
        st.markdown(r"""
        SHAP unites cooperative game theory with machine learning interpretability. 
        In cooperative game theory, the **Shapley value** is the unique method of fairly distributing a collective payout 
        among players based on their marginal contributions across all possible subsets (coalitions).

        $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$

        **Fundamental Axiomatic Guarantees:**
        - **Local Accuracy (Efficiency):** The sum of feature contributions plus the base expected value equals the model output:
          $$\sum_{i=1}^M \phi_i(x) + E[f(x)] = f(x)$$
        - **Missingness:** Features with no impact receive an attribution of exactly 0.
        - **Consistency (Monotonicity):** If a model changes such that a feature's marginal contribution increases, its Shapley value cannot decrease.

        **Regulatory Relevance in Lending:**
        Under the **Equal Credit Opportunity Act (ECOA)** and the **Fair Credit Reporting Act (FCRA)**, creditors must provide 
        specific Adverse Action notices when adverse decisions are made. SHAP translates complex tree ensembles 
        into auditable, compliant reason codes.
        """)

    st.markdown(render_disclaimer(
        "SHAP values measure statistical attribution within the trained model, not causal physical mechanisms. "
        "A feature receiving high positive SHAP indicates that the model heavily weighed it toward a default prediction, "
        "not that altering this feature alone guarantees repayment in the physical world."
    ), unsafe_allow_html=True)
