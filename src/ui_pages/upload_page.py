"""
upload_page.py
--------------
Dedicated Page: Upload Dataset for Batch Default Risk Prediction & Portfolio Analytics.
"""

import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.prediction import (
    predict_batch,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MED_THRESHOLD
)
from src.ui_pages.styles import render_disclaimer, render_metric_card


def render_upload_page():
    st.markdown('<div class="section-header">📁 Upload Dataset for Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a batch dataset (`.csv`) containing loan applicant records to evaluate credit risk at scale.
    The uploaded dataset is automatically cleaned, scaled, encoded using the pre-fitted Scikit-Learn pipeline,
    and scored with your trained machine learning model.
    """)

    # Controls Row: Model Selection & Advanced Thresholds
    col_ctrl1, col_ctrl2 = st.columns([1.5, 1.5])
    with col_ctrl1:
        batch_model = st.selectbox(
            "Select Machine Learning Model for Prediction:",
            options=["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"],
            index=0,
            key="upload_page_model_select",
            help="Select which trained classifier will score the uploaded dataset."
        )

    with col_ctrl2:
        with st.expander("⚙️ Configure Risk Thresholds"):
            c_th1, c_th2 = st.columns(2)
            with c_th1:
                low_th = st.slider("Low Risk Cutoff (< %)", 5, 30, int(DEFAULT_LOW_THRESHOLD * 100)) / 100.0
            with c_th2:
                med_th = st.slider("Medium Risk Cutoff (< %)", 20, 60, int(DEFAULT_MED_THRESHOLD * 100)) / 100.0

    st.markdown("---")

    # Upload & Sample Data Buttons
    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Loan Dataset (.CSV)",
            type=["csv"],
            help="Upload a CSV file containing borrower loan attributes."
        )
    with col_sample:
        st.caption("Don't have a CSV on hand? Test instantly:")
        load_sample = st.button("📋 Load Sample Dataset (25 Records)", use_container_width=True)

    input_df = None

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, low_memory=False)
            st.success(f"Uploaded CSV successfully loaded: **{len(input_df):,} records**.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV file: {e}")

    elif load_sample or "cached_upload_sample" in st.session_state:
        data_path = os.path.join("data", "loan_train.csv")
        if os.path.exists(data_path):
            sample_df = pd.read_csv(data_path, low_memory=False, nrows=25)
            if 'loan_status' in sample_df.columns:
                sample_df = sample_df.drop(columns=['loan_status'])
            st.session_state["cached_upload_sample"] = sample_df
            input_df = sample_df
            st.info(f"Loaded **25 sample applicant records** from LendingClub dataset for risk assessment.")

    # Process and Predict
    if input_df is not None:
        st.subheader("1. Dataset Preview (First 5 Rows)")
        st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("🚀 Predict Default Risk for Uploaded Dataset", type="primary", use_container_width=True):
            with st.spinner("Processing records through pipeline and calculating default probabilities..."):
                results = predict_batch(
                    input_df,
                    model_name=batch_model,
                    low_threshold=low_th,
                    med_threshold=med_th
                )

            if not results["success"]:
                st.error("Batch prediction failed:")
                for err in results["errors"]:
                    st.write(f"- {err}")
            else:
                st.session_state["upload_batch_results"] = results

    # Display Scored Batch Results
    if "upload_batch_results" in st.session_state:
        res = st.session_state["upload_batch_results"]
        summary = res["summary"]
        scored_df = res["scored_df"]

        st.markdown('<div class="section-header">2. Prediction & Risk Assessment Results</div>', unsafe_allow_html=True)

        # Portfolio KPI Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(render_metric_card("Total Applicants", f"{summary['total_records']:,}", f"Model: {res['model_name']}", "#0EA5E9"), unsafe_allow_html=True)
        with m2:
            st.markdown(render_metric_card("Low Risk", f"{summary['low_risk_count']:,}", f"{summary['low_risk_pct']}% of portfolio", "#10B981"), unsafe_allow_html=True)
        with m3:
            st.markdown(render_metric_card("Medium Risk", f"{summary['med_risk_count']:,}", f"{summary['med_risk_pct']}% of portfolio", "#F59E0B"), unsafe_allow_html=True)
        with m4:
            st.markdown(render_metric_card("High Risk", f"{summary['high_risk_count']:,}", f"{summary['high_risk_pct']}% of portfolio", "#EF4444"), unsafe_allow_html=True)
        with m5:
            st.markdown(render_metric_card("Avg Default Risk", f"{summary['avg_probability_pct']}%", "Average probability", "#8B5CF6"), unsafe_allow_html=True)

        # Visual Analytics Row
        col_c1, col_c2 = st.columns([1, 1.2])
        with col_c1:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                values=[summary['low_risk_count'], summary['med_risk_count'], summary['high_risk_count']],
                hole=0.55,
                marker_colors=['#10B981', '#F59E0B', '#EF4444']
            )])
            fig_donut.update_layout(
                title="<b>Portfolio Risk Tier Breakdown</b>",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_c2:
            fig_hist = px.histogram(
                scored_df,
                x="Default_Probability (%)",
                color="Predicted_Risk_Tier",
                color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                nbins=20,
                title="<b>Default Probability Distribution</b>"
            )
            fig_hist.update_layout(height=320, plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Interactive Data Table
        st.subheader("3. Comprehensive Scored Applicants Table")
        st.caption("Each row contains the model-estimated default probability, assigned risk tier, and original applicant fields.")
        st.dataframe(scored_df, use_container_width=True, height=380)

        # Download Scored CSV
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Scored Dataset with Predictions (CSV)",
            data=csv_data,
            file_name="scored_loan_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Drill-down to SHAP
        st.subheader("4. Drill-Down: Explain Individual Applicant with SHAP")
        col_idx, col_go = st.columns([2, 1])
        with col_idx:
            sel_row = st.selectbox(
                "Choose a borrower row from this dataset to inspect:",
                options=list(range(len(scored_df))),
                format_func=lambda i: f"Applicant #{i+1} — Risk: {scored_df.iloc[i]['Predicted_Risk_Tier']} ({scored_df.iloc[i]['Default_Probability (%)']}%)"
            )
        with col_go:
            st.write("")
            st.write("")
            if st.button("🔍 Explain This Applicant with SHAP", use_container_width=True):
                applicant_dict = scored_df.iloc[sel_row].to_dict()
                p_prob = float(applicant_dict['Default_Probability (%)']) / 100.0
                st.session_state["current_prediction"] = {
                    "success": True,
                    "model_name": res["model_name"],
                    "probability": p_prob,
                    "prediction": int(p_prob >= 0.5),
                    "risk_tier": applicant_dict["Predicted_Risk_Tier"],
                    "risk_info": {
                        "tier": applicant_dict["Predicted_Risk_Tier"],
                        "probability_pct": p_prob * 100,
                        "badge_color": "#10B981" if applicant_dict["Predicted_Risk_Tier"] == "Low" else "#F59E0B" if applicant_dict["Predicted_Risk_Tier"] == "Medium" else "#EF4444",
                        "bg_color": "rgba(16, 185, 129, 0.1)",
                        "description": f"Batch applicant #{sel_row+1}: {p_prob*100:.1f}% estimated probability of default."
                    },
                    "X_trans": res["X_trans"][sel_row:sel_row+1]
                }
                st.session_state["current_applicant_input"] = applicant_dict
                st.session_state["selected_model"] = res["model_name"]
                st.success(f"Applicant #{sel_row+1} loaded! Navigate to **🔍 Explainable AI (SHAP)** in the sidebar to view the feature attribution waterfall chart.")

    st.markdown(render_disclaimer(
        "Batch predictions provide statistical risk estimates for portfolio management and educational demonstration. "
        "These estimates do not constitute official credit bureau decisions or contractual loan approvals."
    ), unsafe_allow_html=True)
