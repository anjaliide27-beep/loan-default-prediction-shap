"""
whatif_page.py
--------------
Page 6: Interactive Counterfactual / What-If Scenario Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.prediction import predict_loan_default
from src.ui_pages.styles import render_disclaimer
from src.ui_pages.prediction_page import PRESETS


def render_whatif_page():
    st.markdown('<div class="section-header">What-If Scenario Simulation</div>', unsafe_allow_html=True)
    st.markdown("""
    Test hypothetical counterfactual scenarios to observe how adjustments to key financial attributes 
    (such as increasing income, requesting a smaller loan amount, or lowering interest rates) 
    influence the model's estimated default probability.
    """)

    # Use current applicant from session state or load default
    if "current_applicant_input" not in st.session_state:
        st.session_state["current_applicant_input"] = PRESETS["⚖️ Average Borrower (Moderate Risk Profile)"]
        st.session_state["selected_model"] = "Random Forest"

    base_input = dict(st.session_state["current_applicant_input"])
    selected_model = st.session_state.get("selected_model", "Random Forest")

    # Compute baseline prediction
    base_res = predict_loan_default(base_input, model_name=selected_model)
    base_proba = base_res["probability"]

    st.subheader("1. Counterfactual Scenario Controls")
    st.caption("Modify one or more variables below to evaluate the revised risk profile.")

    col1, col2, col3 = st.columns(3)
    with col1:
        new_inc = st.slider(
            "Revised Annual Income ($)",
            min_value=10000,
            max_value=250000,
            value=int(base_input.get("annual_inc", 50000)),
            step=5000,
            help="Original: ${:,.0f}".format(base_input.get("annual_inc", 50000))
        )
        new_loan = st.slider(
            "Revised Loan Amount ($)",
            min_value=1000,
            max_value=40000,
            value=int(base_input.get("loan_amnt", 15000)),
            step=1000,
            help="Original: ${:,.0f}".format(base_input.get("loan_amnt", 15000))
        )

    with col2:
        new_int_rate = st.slider(
            "Revised Interest Rate (%)",
            min_value=5.0,
            max_value=30.0,
            value=float(str(base_input.get("int_rate", 13.5)).replace("%", "")),
            step=0.5,
            help="Original: {:.1f}%".format(float(str(base_input.get("int_rate", 13.5)).replace("%", "")))
        )
        new_dti = st.slider(
            "Revised Debt-to-Income (DTI %)",
            min_value=0.0,
            max_value=50.0,
            value=float(base_input.get("dti", 18.0)),
            step=1.0,
            help="Original: {:.1f}%".format(float(base_input.get("dti", 18.0)))
        )

    with col3:
        new_inq = st.slider(
            "Revised Inquiries (Last 6 Mos)",
            min_value=0,
            max_value=10,
            value=int(base_input.get("inq_last_6mths", 1)),
            step=1,
            help="Original: {:d}".format(int(base_input.get("inq_last_6mths", 1)))
        )
        new_term = st.selectbox(
            "Revised Loan Term",
            options=["36 months", "60 months"],
            index=0 if "36" in str(base_input.get("term", "36")) else 1
        )

    # Construct revised input dict
    revised_input = dict(base_input)
    revised_input["annual_inc"] = new_inc
    revised_input["loan_amnt"] = new_loan
    revised_input["int_rate"] = new_int_rate
    revised_input["dti"] = new_dti
    revised_input["inq_last_6mths"] = new_inq
    revised_input["term"] = new_term

    # Re-estimate monthly installment for revised scenario
    term_months = 36 if "36" in new_term else 60
    r = (new_int_rate / 100.0) / 12.0
    if r > 0 and term_months > 0:
        revised_input["installment"] = round(new_loan * (r * (1 + r)**term_months) / ((1 + r)**term_months - 1), 2)

    # Predict revised scenario
    revised_res = predict_loan_default(revised_input, model_name=selected_model)
    revised_proba = revised_res["probability"]
    delta_proba = revised_proba - base_proba

    st.subheader("2. Scenario Comparison")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Baseline Default Risk", f"{base_proba*100:.1f}%", f"{base_res['risk_tier']} Tier")
    with m2:
        st.metric(
            "Revised Default Risk",
            f"{revised_proba*100:.1f}%",
            f"{delta_proba*100:+.1f}% Risk Change",
            delta_color="inverse"
        )
    with m3:
        st.metric("Revised Risk Tier", revised_res['risk_tier'])
    with m4:
        st.metric("Revised Installment", f"${revised_input['installment']:.2f}/mo")

    # Visual gauge / side-by-side comparison
    col_chart, col_details = st.columns([1.2, 0.8])
    with col_chart:
        fig_comp = go.Figure(data=[
            go.Bar(
                name='Baseline Profile',
                x=['Baseline', 'Revised Scenario'],
                y=[base_proba * 100, revised_proba * 100],
                marker_color=[base_res['risk_info']['badge_color'], revised_res['risk_info']['badge_color']],
                text=[f"{base_proba*100:.1f}%", f"{revised_proba*100:.1f}%"],
                textposition='auto'
            )
        ])
        fig_comp.update_layout(
            title="Baseline vs. Revised Estimated Default Probability",
            yaxis=dict(title="Default Probability (%)", range=[0, 100], gridcolor="#F1F5F9"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=360
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_details:
        st.subheader("Input Perturbation Delta")
        diff_data = [
            {"Attribute": "Annual Income", "Baseline": f"${base_input.get('annual_inc', 0):,}", "Revised": f"${new_inc:,}", "Direction": "Increased" if new_inc > base_input.get('annual_inc', 0) else "Decreased" if new_inc < base_input.get('annual_inc', 0) else "Unchanged"},
            {"Attribute": "Loan Amount", "Baseline": f"${base_input.get('loan_amnt', 0):,}", "Revised": f"${new_loan:,}", "Direction": "Increased" if new_loan > base_input.get('loan_amnt', 0) else "Decreased" if new_loan < base_input.get('loan_amnt', 0) else "Unchanged"},
            {"Attribute": "Interest Rate", "Baseline": f"{base_input.get('int_rate', 0)}%", "Revised": f"{new_int_rate}%", "Direction": "Increased" if new_int_rate > float(str(base_input.get('int_rate', 0)).replace('%','')) else "Decreased" if new_int_rate < float(str(base_input.get('int_rate', 0)).replace('%','')) else "Unchanged"},
            {"Attribute": "DTI Ratio", "Baseline": f"{base_input.get('dti', 0)}%", "Revised": f"{new_dti}%", "Direction": "Increased" if new_dti > base_input.get('dti', 0) else "Decreased" if new_dti < base_input.get('dti', 0) else "Unchanged"},
            {"Attribute": "Inquiries (6M)", "Baseline": f"{base_input.get('inq_last_6mths', 0)}", "Revised": f"{new_inq}", "Direction": "Increased" if new_inq > base_input.get('inq_last_6mths', 0) else "Decreased" if new_inq < base_input.get('inq_last_6mths', 0) else "Unchanged"},
            {"Attribute": "Loan Term", "Baseline": f"{base_input.get('term', '')}", "Revised": f"{new_term}", "Direction": "Changed" if str(base_input.get('term', '')) != str(new_term) else "Unchanged"}
        ]
        st.dataframe(pd.DataFrame(diff_data).set_index("Attribute"), use_container_width=True)

    st.markdown(render_disclaimer(
        "What-if simulations evaluate mathematical sensitivity across model features. "
        "These scenarios do not constitute causal estimates or guarantees that modifying individual parameters "
        "in isolation will produce equivalent credit outcomes in real-world lending markets."
    ), unsafe_allow_html=True)
