"""
prediction_page.py
------------------
Page 3: Interactive Loan Default Prediction Form, Real-Time Risk Assessment,
and Batch Dataset CSV Upload Scoring.
"""

import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.prediction import (
    predict_loan_default,
    predict_batch,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MED_THRESHOLD
)
from src.cost_optimization import (
    get_validation_data_and_probabilities,
    optimize_decision_threshold
)
from src.ui_pages.styles import render_disclaimer, render_metric_card


@st.cache_data(show_spinner=False)
def load_val_predictions_cached(model_name: str):
    """Loads validation predictions for cost-sensitive threshold optimization with caching."""
    data_path = os.path.join("data", "loan_train.csv")
    if not os.path.exists(data_path):
        return None
    try:
        return get_validation_data_and_probabilities(data_path, model_name=model_name)
    except Exception:
        return None


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
    st.markdown('<div class="section-header">Loan Default Prediction & Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("""
    Evaluate credit risk using your trained machine learning model. You can evaluate an **individual borrower** 
    or **upload a batch CSV dataset** to predict default probabilities and risk tiers at scale.
    """)

    tab_single, tab_batch = st.tabs([
        "📝 Single Borrower Assessment",
        "📁 Batch Dataset Upload & Prediction"
    ])

    # ---------------- TAB 1: SINGLE BORROWER ----------------
    with tab_single:
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
                key="single_model_select",
                help="Random Forest is the primary model optimized for continuous SHAP interpretability."
            )

        defaults = PRESETS.get(preset_choice) or st.session_state.get("current_applicant_input") or PRESETS["⭐ Prime Borrower (Low Risk Profile)"]

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

        # ---------------- COST-SENSITIVE THRESHOLD OPTIMIZATION ----------------
        st.markdown("---")
        st.markdown("### 💰 Cost-Sensitive Decision Threshold Optimization")
        st.caption("Future Scope Item 1: Optimizing the operational classification cutoff to minimize asymmetric financial error costs.")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            c_fn = st.number_input(
                "False-Negative Cost ($C_FN) - Missed Default:",
                min_value=500.0,
                max_value=50000.0,
                value=5000.0,
                step=500.0,
                help="Financial charge-off loss when a defaulting borrower is erroneously approved."
            )
        with col_c2:
            c_fp = st.number_input(
                "False-Positive Cost ($C_FP) - Unnecessary Rejection:",
                min_value=100.0,
                max_value=20000.0,
                value=1000.0,
                step=100.0,
                help="Opportunity cost and lost origination revenue when a creditworthy borrower is rejected."
            )

        val_data = load_val_predictions_cached(selected_model)
        if val_data is None:
            st.warning("⚠️ Validation dataset unavailable. Falling back to manual decision threshold adjustment.")
            manual_th = st.slider("Manual Decision Threshold:", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
            st.info(f"Manual threshold set to {manual_th*100:.1f}%. Applicants with probability >= {manual_th*100:.1f}% are flagged as Default Risk.")
        else:
            opt_res = optimize_decision_threshold(
                val_data["y_true"],
                val_data["y_proba"],
                c_fn=float(c_fn),
                c_fp=float(c_fp)
            )

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            with m_c1:
                st.markdown(render_metric_card(
                    "Optimal Cutoff (τ*)",
                    f"{opt_res['optimal_threshold']*100:.1f}%",
                    "Default Baseline: 50.0%"
                ), unsafe_allow_html=True)
            with m_c2:
                st.markdown(render_metric_card(
                    "Validation Total Cost",
                    f"${opt_res['optimal_cost']:,.0f}",
                    f"Baseline Cost: ${opt_res['default_cost']:,.0f}"
                ), unsafe_allow_html=True)
            with m_c3:
                st.markdown(render_metric_card(
                    "Projected Cost Savings",
                    f"${opt_res['cost_savings']:,.0f}",
                    f"-{opt_res['cost_savings_pct']:.1f}% Total Loss Reduction"
                ), unsafe_allow_html=True)
            with m_c4:
                st.markdown(render_metric_card(
                    "Validation Error Shift",
                    f"FN: {opt_res['optimal_fn']:,}",
                    f"FP: {opt_res['optimal_fp']:,} (Baseline FN: {opt_res['default_fn']:,})"
                ), unsafe_allow_html=True)

            # Interactive Plotly Cost Curve
            df_curve = opt_res["cost_curve_df"]
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=df_curve["threshold"] * 100.0,
                y=df_curve["total_cost"],
                mode='lines',
                name='Total Cost ($)',
                line=dict(color='#0F766E', width=3)
            ))
            fig_cost.add_trace(go.Scatter(
                x=[opt_res["optimal_threshold"] * 100.0],
                y=[opt_res["optimal_cost"]],
                mode='markers+text',
                name=f'Optimal τ* ({opt_res["optimal_threshold"]*100:.1f}%)',
                marker=dict(color='#DC2626', size=12, symbol='star'),
                text=[f"Min Cost: ${opt_res['optimal_cost']:,.0f}"],
                textposition='top center'
            ))
            fig_cost.add_trace(go.Scatter(
                x=[50.0],
                y=[opt_res["default_cost"]],
                mode='markers+text',
                name='Default 50% Threshold',
                marker=dict(color='#64748B', size=10, symbol='circle'),
                text=[f"Cost @ 50%: ${opt_res['default_cost']:,.0f}"],
                textposition='bottom center'
            ))
            fig_cost.update_layout(
                title=f"Validation Total Error Cost Curve vs. Decision Threshold ({selected_model})",
                xaxis=dict(title="Decision Threshold (%)", gridcolor="#F1F5F9"),
                yaxis=dict(title="Total Illustrative Error Cost ($)", gridcolor="#F1F5F9"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=350,
                margin=dict(t=35, b=30, l=60, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_cost, use_container_width=True)

            # Impact on current applicant if available
            curr_pred = st.session_state.get("current_prediction")
            if curr_pred and curr_pred.get("success"):
                p_val = curr_pred["probability"]
                opt_th = opt_res["optimal_threshold"]
                default_class = "Default Risk (Flagged)" if p_val >= 0.50 else "Repay (Approved)"
                opt_class = "Default Risk (Flagged)" if p_val >= opt_th else "Repay (Approved)"

                status_color = "#DC2626" if opt_class.startswith("Default") else "#0F766E"

                st.markdown(f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid {status_color}; border-radius: 8px; padding: 1rem 1.25rem; margin-top: 0.5rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 0.35rem;">
                        Operational Decision Impact for Current Applicant
                    </div>
                    <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                        • <strong>Model Predicted Default Probability:</strong> <code>{p_val*100:.1f}%</code> (Unchanged - model is not retrained)<br>
                        • <strong>Decision at Standard 0.50 Threshold:</strong> <strong>{default_class}</strong> (Probability {'≥' if p_val >= 0.50 else '<'} 50.0%)<br>
                        • <strong>Decision at Cost-Optimal Threshold ({opt_th*100:.1f}%):</strong> <strong style="color: {status_color};">{opt_class}</strong> (Probability {'≥' if p_val >= opt_th else '<'} {opt_th*100:.1f}%)<br>
                        <em>Note: Because missed defaults ($C_FN=${c_fn:,.0f}) are more expensive than unnecessary rejections ($C_FP=${c_fp:,.0f}), the optimal policy lowers the cutoff to protect portfolio capital.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("💡 **Tip:** Submit a loan application above or load a preset profile to observe how the cost-optimal cutoff classifies that specific borrower.")

            st.markdown(render_disclaimer(
                "Cost-Sensitive Safeguards: Decision thresholds are evaluated strictly on an independent validation partition "
                "(20% stratified subset of training data, random_state=99) completely isolated from model training and held-out test data. "
                "Cost parameters ($C_FN and $C_FP) are illustrative simulation values to model asymmetric credit risk and do not constitute actual lender underwriting mandates."
            ), unsafe_allow_html=True)

    # ---------------- TAB 2: BATCH DATASET UPLOAD ----------------
    with tab_batch:
        st.subheader("Batch Dataset Upload & Default Risk Prediction")
        st.markdown("""
        Upload a `.csv` file containing loan applications to calculate default probability and risk tier for every record.
        The uploaded data is transformed using the exact fitted preprocessing pipeline.
        """)

        col_b_model, col_b_btn = st.columns([1.5, 1.5])
        with col_b_model:
            batch_model = st.selectbox(
                "Select Model for Batch Scoring:",
                options=["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"],
                index=0,
                key="batch_model_select"
            )
        with col_b_btn:
            st.caption("Quick Test without file:")
            load_sample_btn = st.button("📋 Load 20 Sample Records from Dataset", use_container_width=True)

        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Upload any CSV with loan applicant columns")

        batch_df = None
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file, low_memory=False)
                st.success(f"Uploaded file successfully loaded: **{len(batch_df):,} records**.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

        elif load_sample_btn or "cached_sample_batch" in st.session_state:
            data_file = os.path.join("data", "loan_train.csv")
            if os.path.exists(data_file):
                raw_sample = pd.read_csv(data_file, low_memory=False, nrows=20)
                # Drop loan_status to simulate new applicants
                if 'loan_status' in raw_sample.columns:
                    raw_sample = raw_sample.drop(columns=['loan_status'])
                st.session_state["cached_sample_batch"] = raw_sample
                batch_df = raw_sample
                st.info(f"Loaded **20 sample loan applicant records** from dataset for evaluation.")

        if batch_df is not None:
            st.write("Preview of Uploaded Data (First 5 Rows):")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("🚀 Predict Batch Risk for All Records", type="primary", use_container_width=True):
                with st.spinner("Processing records through pipeline and generating risk estimates..."):
                    batch_res = predict_batch(batch_df, model_name=batch_model)

                if not batch_res["success"]:
                    st.error("Batch processing failed:")
                    for err in batch_res["errors"]:
                        st.write(f"- {err}")
                else:
                    st.session_state["batch_results"] = batch_res

        # Display batch results if available
        if "batch_results" in st.session_state:
            res = st.session_state["batch_results"]
            s = res["summary"]
            scored_df = res["scored_df"]

            st.markdown('<div class="section-header">Batch Prediction Summary</div>', unsafe_allow_html=True)
            
            # Metrics cards
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.markdown(render_metric_card("Total Applicants", f"{s['total_records']:,}", f"Model: {res['model_name']}", "#0EA5E9"), unsafe_allow_html=True)
            with k2:
                st.markdown(render_metric_card("Low Risk", f"{s['low_risk_count']:,}", f"{s['low_risk_pct']}% of total", "#10B981"), unsafe_allow_html=True)
            with k3:
                st.markdown(render_metric_card("Medium Risk", f"{s['med_risk_count']:,}", f"{s['med_risk_pct']}% of total", "#F59E0B"), unsafe_allow_html=True)
            with k4:
                st.markdown(render_metric_card("High Risk", f"{s['high_risk_count']:,}", f"{s['high_risk_pct']}% of total", "#EF4444"), unsafe_allow_html=True)
            with k5:
                st.markdown(render_metric_card("Avg Default Risk", f"{s['avg_probability_pct']}%", "Portfolio baseline", "#8B5CF6"), unsafe_allow_html=True)

            # Charts Row
            c_donut, c_hist = st.columns([1, 1.2])
            with c_donut:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Low Risk', 'Medium Risk', 'High Risk'],
                    values=[s['low_risk_count'], s['med_risk_count'], s['high_risk_count']],
                    hole=.5,
                    marker_colors=['#10B981', '#F59E0B', '#EF4444']
                )])
                fig_pie.update_layout(title="Batch Risk Distribution", height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with c_hist:
                fig_prob = px.histogram(
                    scored_df,
                    x="Default_Probability (%)",
                    color="Predicted_Risk_Tier",
                    color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                    nbins=20,
                    title="Default Probability Distribution Across Batch"
                )
                fig_prob.update_layout(height=300, plot_bgcolor="#FFFFFF")
                st.plotly_chart(fig_prob, use_container_width=True)

            # Interactive Scored Table
            st.subheader("Scored Loan Applicants Table")
            st.dataframe(scored_df, use_container_width=True, height=350)

            # Download CSV
            csv_buffer = io.StringIO()
            scored_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Download Scored Predictions CSV",
                data=csv_data,
                file_name="scored_loan_applicants.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Drill-down to SHAP
            st.subheader("🔍 Inspect Single Applicant from Batch with SHAP")
            col_sel_idx, col_sel_btn = st.columns([2, 1])
            with col_sel_idx:
                selected_idx = st.selectbox(
                    "Select Borrower Row Index to Explain:",
                    options=list(range(len(scored_df))),
                    format_func=lambda i: f"Row {i} - Default Prob: {scored_df.iloc[i]['Default_Probability (%)']}% ({scored_df.iloc[i]['Predicted_Risk_Tier']} Risk)"
                )
            with col_sel_btn:
                st.write("")
                st.write("")
                if st.button("🔬 View in SHAP Explainability", use_container_width=True):
                    row_data = scored_df.iloc[selected_idx].to_dict()
                    # Reconstruct prediction result for this row
                    p_val = float(row_data['Default_Probability (%)']) / 100.0
                    st.session_state["current_prediction"] = {
                        "success": True,
                        "model_name": res["model_name"],
                        "probability": p_val,
                        "prediction": int(p_val >= 0.5),
                        "risk_tier": row_data["Predicted_Risk_Tier"],
                        "risk_info": {
                            "tier": row_data["Predicted_Risk_Tier"],
                            "probability_pct": p_val * 100,
                            "badge_color": "#10B981" if row_data["Predicted_Risk_Tier"] == "Low" else "#F59E0B" if row_data["Predicted_Risk_Tier"] == "Medium" else "#EF4444",
                            "bg_color": "rgba(16, 185, 129, 0.1)",
                            "description": f"Batch prediction: {p_val*100:.1f}% estimated probability of default."
                        },
                        "X_trans": res["X_trans"][selected_idx:selected_idx+1]
                    }
                    st.session_state["current_applicant_input"] = row_data
                    st.session_state["selected_model"] = res["model_name"]
                    st.success(f"Row {selected_idx} loaded! Navigate to **🔍 Explainable AI (SHAP)** in the sidebar to view its full attribution.")

    st.markdown(render_disclaimer(
        "Risk categories are strictly project-defined benchmarks based on statistical probabilities. "
        "They do not represent regulatory lending decisions, credit bureau scores, or contractual loan approvals."
    ), unsafe_allow_html=True)
