"""
prediction_page.py
------------------
Page 3: Interactive Loan Default Prediction Form & Real-Time Risk Assessment.
"""

import streamlit as st
import numpy as np
import pandas as pd
from src.prediction import predict_loan_default, DEFAULT_LOW_THRESHOLD, DEFAULT_MED_THRESHOLD
from src.ui_pages.styles import render_disclaimer

# Realistic presets for 1-click demonstration
PRESETS = {
    "Select a preset profile...": None,
    "⭐ Prime Borrower (Low Risk Profile)": {
        'loan_amnt': 10000,
        'term': '36 months',
        'int_rate': 7.5,
        'installment': 311.06,
        'grade': 'A',
        'emp_length': '10+ years',
        'home_ownership': 'MORTGAGE',
        'annual_inc': 95000.0,
        'verification_status': 'Verified',
        'purpose': 'debt_consolidation',
        'dti': 8.5,
        'delinq_2yrs': 0,
        'inq_last_6mths': 0,
        'open_acc': 12,
        'pub_rec': 0,
        'revol_bal': 5200,
        'revol_util': 22.0,
        'total_acc': 24,
        'pub_rec_bankruptcies': 0
    },
    "⚖️ Average Borrower (Moderate Risk Profile)": {
        'loan_amnt': 15000,
        'term': '36 months',
        'int_rate': 13.5,
        'installment': 509.0,
        'grade': 'C',
        'emp_length': '4 years',
        'home_ownership': 'RENT',
        'annual_inc': 52000.0,
        'verification_status': 'Source Verified',
        'purpose': 'debt_consolidation',
        'dti': 17.5,
        'delinq_2yrs': 0,
        'inq_last_6mths': 1,
        'open_acc': 9,
        'pub_rec': 0,
        'revol_bal': 11500,
        'revol_util': 58.0,
        'total_acc': 16,
        'pub_rec_bankruptcies': 0
    },
    "⚠️ Subprime Borrower (High Risk Profile)": {
        'loan_amnt': 22000,
        'term': '60 months',
        'int_rate': 19.8,
        'installment': 580.0,
        'grade': 'E',
        'emp_length': '1 year',
        'home_ownership': 'RENT',
        'annual_inc': 34000.0,
        'verification_status': 'Not Verified',
        'purpose': 'small_business',
        'dti': 28.5,
        'delinq_2yrs': 1,
        'inq_last_6mths': 3,
        'open_acc': 6,
        'pub_rec': 1,
        'revol_bal': 19000,
        'revol_util': 84.0,
        'total_acc': 14,
        'pub_rec_bankruptcies': 1
    }
}


def calculate_amortization_installment(principal: float, annual_rate_pct: float, months: int) -> float:
    """Standard loan amortization formula."""
    if months <= 0 or principal <= 0:
        return 0.0
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        return round(principal / months, 2)
    installment = principal * (r * (1 + r)**months) / ((1 + r)**months - 1)
    return round(installment, 2)


def render_prediction_page():
    st.markdown('<div class="section-header">Loan Default Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    Enter applicant details to generate an instantaneous, model-driven default probability estimate.
    All inputs feed through the pre-fitted Scikit-Learn ColumnTransformer pipeline.
    """)

    # Preset selection
    col_preset, col_model = st.columns([1.5, 1.5])
    with col_preset:
        preset_choice = st.selectbox(
            "Quick Demo: Load Preset Borrower Profile",
            options=list(PRESETS.keys())
        )
    with col_model:
        selected_model = st.selectbox(
            "Select Machine Learning Classifier:",
            options=["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"],
            index=0,
            help="Random Forest is the primary model optimized for balanced risk scoring and SHAP interpretability."
        )

    # Initialize form defaults from preset if chosen
    defaults = PRESETS.get(preset_choice) or st.session_state.get("current_applicant_input") or PRESETS["⭐ Prime Borrower (Low Risk Profile)"]

    # Interactive Form
    with st.form("loan_application_form"):
        st.subheader("1. Loan Specifications")
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=500,
                max_value=100000,
                value=int(defaults.get('loan_amnt', 12000)),
                step=500
            )
            term = st.selectbox(
                "Loan Term",
                options=["36 months", "60 months"],
                index=0 if defaults.get('term') == '36 months' else 1
            )
        with col_l2:
            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=1.0,
                max_value=40.0,
                value=float(defaults.get('int_rate', 11.5)),
                step=0.25
            )
            grade = st.selectbox(
                "Assigned Loan Grade",
                options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                index=['A', 'B', 'C', 'D', 'E', 'F', 'G'].index(defaults.get('grade', 'B'))
            )
        with col_l3:
            term_m = 36 if '36' in term else 60
            amort_installment = calculate_amortization_installment(loan_amnt, int_rate, term_m)
            installment = st.number_input(
                "Monthly Installment ($)",
                min_value=10.0,
                max_value=5000.0,
                value=float(defaults.get('installment', amort_installment)),
                step=10.0,
                help=f"Calculated standard amortization: ${amort_installment}/mo"
            )

        st.subheader("2. Borrower Financial & Employment Profile")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=1000,
                max_value=1000000,
                value=int(defaults.get('annual_inc', 60000)),
                step=2000
            )
            emp_length = st.selectbox(
                "Employment Length",
                options=[
                    '< 1 year', '1 year', '2 years', '3 years', '4 years',
                    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
                ],
                index=[
                    '< 1 year', '1 year', '2 years', '3 years', '4 years',
                    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
                ].index(defaults.get('emp_length', '5 years'))
            )
        with col_f2:
            home_ownership = st.selectbox(
                "Home Ownership Status",
                options=['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
                index=['RENT', 'MORTGAGE', 'OWN', 'OTHER'].index(defaults.get('home_ownership', 'RENT'))
            )
            verification_status = st.selectbox(
                "Income Verification Status",
                options=['Not Verified', 'Verified', 'Source Verified'],
                index=['Not Verified', 'Verified', 'Source Verified'].index(defaults.get('verification_status', 'Verified'))
            )
        with col_f3:
            purpose = st.selectbox(
                "Loan Purpose",
                options=[
                    'debt_consolidation', 'credit_card', 'home_improvement',
                    'major_purchase', 'small_business', 'car', 'wedding',
                    'medical', 'moving', 'vacation', 'house', 'educational',
                    'renewable_energy', 'other'
                ],
                index=[
                    'debt_consolidation', 'credit_card', 'home_improvement',
                    'major_purchase', 'small_business', 'car', 'wedding',
                    'medical', 'moving', 'vacation', 'house', 'educational',
                    'renewable_energy', 'other'
                ].index(defaults.get('purpose', 'debt_consolidation'))
            )
            dti = st.number_input(
                "Debt-to-Income (DTI %)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get('dti', 16.0)),
                step=0.5
            )

        st.subheader("3. Credit Bureau Indicators")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            delinq_2yrs = st.number_input("Past 2-Yr Delinquencies", min_value=0, max_value=20, value=int(defaults.get('delinq_2yrs', 0)))
            inq_last_6mths = st.number_input("Inquiries Last 6 Months", min_value=0, max_value=20, value=int(defaults.get('inq_last_6mths', 0)))
        with col_c2:
            open_acc = st.number_input("Open Credit Accounts", min_value=1, max_value=80, value=int(defaults.get('open_acc', 10)))
            total_acc = st.number_input("Total Credit Lines", min_value=1, max_value=150, value=int(defaults.get('total_acc', 20)))
        with col_c3:
            revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=1000000, value=int(defaults.get('revol_bal', 8500)), step=500)
            revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=float(defaults.get('revol_util', 45.0)), step=1.0)
        with col_c4:
            pub_rec = st.number_input("Derogatory Public Records", min_value=0, max_value=20, value=int(defaults.get('pub_rec', 0)))
            pub_rec_bankruptcies = st.number_input("Public Bankruptcies", min_value=0, max_value=10, value=int(defaults.get('pub_rec_bankruptcies', 0)))

        # Configurable Risk Thresholds Expander
        with st.expander("⚙️ Advanced: Adjust Project Risk Thresholds"):
            st.caption("Customize probability cutoffs for Low, Medium, and High risk classifications.")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                low_th = st.slider("Low Risk Threshold (< %)", 5, 30, int(DEFAULT_LOW_THRESHOLD * 100)) / 100.0
            with col_t2:
                med_th = st.slider("Medium Risk Threshold (< %)", 20, 60, int(DEFAULT_MED_THRESHOLD * 100)) / 100.0

        submit_btn = st.form_submit_button("🚀 Predict Default Probability", use_container_width=True)

    if submit_btn:
        input_data = {
            'loan_amnt': loan_amnt,
            'term': term,
            'int_rate': int_rate,
            'installment': installment,
            'grade': grade,
            'emp_length': emp_length,
            'home_ownership': home_ownership,
            'annual_inc': annual_inc,
            'verification_status': verification_status,
            'purpose': purpose,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'inq_last_6mths': inq_last_6mths,
            'open_acc': open_acc,
            'pub_rec': pub_rec,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'total_acc': total_acc,
            'pub_rec_bankruptcies': pub_rec_bankruptcies
        }

        # Run inference
        result = predict_loan_default(
            input_data,
            model_name=selected_model,
            low_threshold=low_th,
            med_threshold=med_th
        )

        if not result["success"]:
            st.error("❌ Prediction failed due to validation errors:")
            for err in result["errors"]:
                st.write(f"- {err}")
        else:
            # Store in session state for SHAP and What-If pages
            st.session_state["current_prediction"] = result
            st.session_state["current_applicant_input"] = input_data
            st.session_state["selected_model"] = selected_model

            proba_pct = result["probability"] * 100.0
            risk = result["risk_info"]

            st.markdown(f"""
            <div class="risk-box" style="background-color: {risk['bg_color']}; border-color: {risk['badge_color']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                            Model: {selected_model}
                        </span>
                        <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;">
                            Default Risk: {proba_pct:.1f}%
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span class="badge" style="background-color: {risk['badge_color']}; color: #FFFFFF; font-size: 1rem; padding: 0.4rem 1.2rem;">
                            {risk['tier']} Risk
                        </span>
                    </div>
                </div>
                <div style="font-size: 0.95rem; color: #334155; line-height: 1.5;">
                    {risk['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info("💡 **Next Step:** Navigate to **Explainable AI (SHAP)** in the sidebar to view the mathematical breakdown of features driving this specific prediction.")

    st.markdown(render_disclaimer(
        "Risk categories are strictly project-defined benchmarks based on statistical probabilities. "
        "They do not represent regulatory lending decisions, credit bureau scores, or contractual loan approvals."
    ), unsafe_allow_html=True)
