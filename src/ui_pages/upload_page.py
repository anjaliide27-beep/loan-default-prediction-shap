"""
upload_page.py
--------------
Dataset Upload & Machine Learning Model Selection Page.
Enables users to choose between Supervised Learning (default prediction with labeled data)
and Unsupervised Learning (borrower segmentation with K-Means without labels).
Only the selected workflow's interface is displayed.
"""

import os
import io
import hashlib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import TARGET_COL
from src.prediction import (
    predict_batch,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MED_THRESHOLD
)
from src.clustering import run_kmeans_segmentation, CANDIDATE_CLUSTERING_FEATURES
from src.ui_pages.styles import render_disclaimer, render_metric_card


def compute_df_hash(df: pd.DataFrame) -> str:
    """Computes a fast deterministic MD5 hash of a DataFrame to track dataset identity."""
    val_str = f"{df.shape}_{list(df.columns)}_{df.iloc[:min(10, len(df))].values.tobytes()}"
    return hashlib.md5(val_str.encode('utf-8', errors='ignore')).hexdigest()


def render_upload_page():
    st.markdown('<div class="section-header">📁 Upload Dataset & Machine Learning Model Selection</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a custom CSV loan dataset and select your machine learning approach. 
    You can evaluate credit default risk using **Supervised Learning** (requires labeled historical data) 
    or explore borrower financial patterns using **Unsupervised Learning** (K-Means clustering without labels).
    """)

    # ---------------- 1. CHOOSE MACHINE LEARNING APPROACH ----------------
    st.subheader("1. Choose Your Machine Learning Approach")
    st.markdown("Select **exactly one** analytical workflow for the uploaded dataset:")

    current_mode = st.session_state.get("workflow_mode", "supervised")
    approach_index = 0 if current_mode != "unsupervised" else 1

    selected_approach = st.radio(
        "Select Machine Learning Approach:",
        options=[
            "🎯 Option A: Supervised Learning (Loan Default Prediction)",
            "👥 Option B: Unsupervised Learning (K-Means Borrower Segmentation)"
        ],
        index=approach_index,
        help="Select Supervised Learning to predict default probabilities with labeled data, or Unsupervised Learning to cluster borrowers without labels."
    )

    is_supervised = "Supervised" in selected_approach
    chosen_mode = "supervised" if is_supervised else "unsupervised"

    # Dynamic explanation card for chosen approach
    if is_supervised:
        st.markdown("""
        <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 5px solid #16A34A; border-radius: 8px; padding: 0.9rem 1.15rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: #166534; font-size: 0.95rem; margin-bottom: 0.25rem;">
                🎯 Supervised Learning: Loan Default Risk Classification
            </div>
            <div style="font-size: 0.86rem; color: #14532D; line-height: 1.45;">
                • <strong>Purpose:</strong> Predict default probability and assign risk tiers (Low, Medium, High) for each loan application.<br>
                • <strong>Requirement:</strong> <strong>Requires labeled historical data containing the <code>loan_status</code> column</strong> to validate credit risk.<br>
                • <strong>Features:</strong> Batch scoring with pre-trained models (Random Forest, XGBoost), portfolio risk distribution, and SHAP explainability.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #F0F9FF; border: 1px solid #BAE6FD; border-left: 5px solid #0284C7; border-radius: 8px; padding: 0.9rem 1.15rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: #0369A1; font-size: 0.95rem; margin-bottom: 0.25rem;">
                👥 Unsupervised Learning: K-Means Borrower Segmentation
            </div>
            <div style="font-size: 0.86rem; color: #0C4A6E; line-height: 1.45;">
                • <strong>Purpose:</strong> Discover natural groupings of similar borrowers based purely on financial attributes.<br>
                • <strong>Requirement:</strong> <strong>Does NOT require repayment labels (<code>loan_status</code>)</strong> to form clusters.<br>
                • <strong>Features:</strong> Standardized K-Means clustering, 2D PCA projection scatter plot, cluster financial profile tables, and statistical interpretations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- 2. DATASET UPLOAD / SAMPLE SELECTION ----------------
    st.subheader("2. Upload Your Loan Dataset (CSV)")

    col_upload, col_sample = st.columns([1.8, 1.2])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload CSV Dataset:",
            type=["csv"],
            help="Upload a CSV file containing borrower loan attributes."
        )

    with col_sample:
        st.caption("Quick Test Options:")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            load_labeled_sample = st.button("📋 Labeled Data (with loan_status)", use_container_width=True, help="Loads 30 records with loan_status for Supervised testing")
        with c_btn2:
            load_unlabeled_sample = st.button("📋 Unlabeled Data (no loan_status)", use_container_width=True, help="Loads 30 records without loan_status for Unsupervised testing")

    input_df = None
    input_file_name = "Uploaded Dataset"

    # Handle file upload or sample load
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, low_memory=False)
            input_file_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading uploaded CSV file: {e}")
            return
    elif load_labeled_sample:
        data_path = os.path.join("data", "loan_train.csv")
        if os.path.exists(data_path):
            sample_df = pd.read_csv(data_path, low_memory=False, nrows=30)
            input_df = sample_df
            input_file_name = "sample_labeled_loans_30.csv"
            st.session_state["cached_sample_df"] = input_df
            st.session_state["cached_sample_name"] = input_file_name
    elif load_unlabeled_sample:
        data_path = os.path.join("data", "loan_train.csv")
        if os.path.exists(data_path):
            sample_df = pd.read_csv(data_path, low_memory=False, nrows=30)
            if TARGET_COL in sample_df.columns:
                sample_df = sample_df.drop(columns=[TARGET_COL])
            input_df = sample_df
            input_file_name = "sample_unlabeled_loans_30.csv"
            st.session_state["cached_sample_df"] = input_df
            st.session_state["cached_sample_name"] = input_file_name
    elif "cached_sample_df" in st.session_state:
        input_df = st.session_state["cached_sample_df"]
        input_file_name = st.session_state.get("cached_sample_name", "Sample Dataset")
    elif "uploaded_df" in st.session_state:
        input_df = st.session_state["uploaded_df"]
        input_file_name = st.session_state.get("uploaded_file_name", "Uploaded Dataset")

    # If no data loaded yet
    if input_df is None or len(input_df) == 0:
        st.info("👆 Please upload a `.csv` file above or click one of the sample buttons to get started.")
        return

    # Check for dataset identity change
    current_hash = compute_df_hash(input_df)
    prev_hash = st.session_state.get("uploaded_dataset_hash")

    if prev_hash != current_hash:
        # A new dataset was loaded: clear stale results
        st.session_state.pop("upload_batch_results", None)
        st.session_state.pop("upload_clustering_results", None)
        st.session_state["uploaded_dataset_hash"] = current_hash
        st.session_state["uploaded_df"] = input_df
        st.session_state["uploaded_file_name"] = input_file_name

    # Display dataset metadata and preview
    has_target = TARGET_COL in input_df.columns
    target_badge = (
        '<span style="background-color: #DCFCE7; color: #166534; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">🏷️ Labeled (loan_status detected)</span>'
        if has_target else
        '<span style="background-color: #FEF3C7; color: #92400E; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">⚪ Unlabeled (no loan_status)</span>'
    )

    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;">
        <strong>Active Dataset:</strong> <code>{input_file_name}</code> &nbsp;|&nbsp; 
        <strong>Dimensions:</strong> {len(input_df):,} rows × {input_df.shape[1]} columns &nbsp;|&nbsp; 
        {target_badge}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👁️ View Dataset Preview (First 5 Rows)", expanded=False):
        st.dataframe(input_df.head(5), use_container_width=True)

    st.markdown("---")

    # ---------------- 3. WORKFLOW CONFIGURATION & CONFIRMATION ----------------
    st.subheader(f"3. Configure & Load {chosen_mode.capitalize()} Learning Workflow")

    if is_supervised:
        col_m1, col_m2 = st.columns([1.5, 1.5])
        with col_m1:
            batch_model = st.selectbox(
                "Select Supervised Classifier for Batch Scoring:",
                options=["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"],
                index=0,
                help="Pre-trained supervised model that evaluates the default probability."
            )
        with col_m2:
            with st.expander("⚙️ Advanced: Configure Risk Cutoffs"):
                col_th1, col_th2 = st.columns(2)
                with col_th1:
                    low_th = st.slider("Low Risk (< %)", 5, 30, int(DEFAULT_LOW_THRESHOLD * 100)) / 100.0
                with col_th2:
                    med_th = st.slider("Medium Risk (< %)", 20, 60, int(DEFAULT_MED_THRESHOLD * 100)) / 100.0
    else:
        col_k1, col_k2 = st.columns([1.5, 2.5])
        with col_k1:
            n_clusters = st.slider(
                "Select Number of Clusters (k):",
                min_value=2,
                max_value=6,
                value=3,
                step=1,
                help="Number of borrower segments to identify using K-Means (default: 3)."
            )
        with col_k2:
            st.markdown("""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.65rem 0.85rem; font-size: 0.82rem; color: #475569; margin-top: 0.25rem;">
                <strong>Unsupervised Safeguards:</strong> <code>loan_status</code> and identifiers are strictly excluded from clustering inputs.
            </div>
            """, unsafe_allow_html=True)

    # Action Confirmation Button
    load_btn = st.button("🚀 Load Dataset and Continue", type="primary", use_container_width=True)

    # Check if workflow mode changed
    if st.session_state.get("workflow_mode") != chosen_mode:
        if load_btn:
            # Clear previous results when switching modes
            st.session_state["workflow_mode"] = chosen_mode
            st.session_state.pop("upload_batch_results", None)
            st.session_state.pop("upload_clustering_results", None)
            st.session_state["workflow_confirmed"] = True

    if load_btn:
        st.session_state["workflow_mode"] = chosen_mode
        st.session_state["workflow_confirmed"] = True

    # ---------------- 4. WORKFLOW EXECUTION & DISPLAY ----------------
    active_mode = st.session_state.get("workflow_mode", chosen_mode)
    is_confirmed = st.session_state.get("workflow_confirmed", False)

    if not is_confirmed and "upload_batch_results" not in st.session_state and "upload_clustering_results" not in st.session_state:
        st.info("💡 Click **'Load Dataset and Continue'** above to validate your dataset and launch the selected workflow.")
        return

    st.markdown("---")

    # ==================== WORKFLOW A: SUPERVISED LEARNING ====================
    if active_mode == "supervised":
        st.markdown('<div class="section-header">🎯 Supervised Default Prediction Results</div>', unsafe_allow_html=True)

        # VALIDATION 1: Check target column presence
        if TARGET_COL not in input_df.columns:
            st.error("❌ Target Column Missing for Supervised Learning")
            st.markdown(f"""
            <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-left: 5px solid #DC2626; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem;">
                <div style="font-size: 1rem; font-weight: 700; color: #991B1B; margin-bottom: 0.5rem;">
                    Supervised loan default evaluation requires labeled historical data containing the <code>loan_status</code> column.
                </div>
                <div style="font-size: 0.88rem; color: #7F1D1D; line-height: 1.5;">
                    The uploaded dataset (<code>{input_file_name}</code>) does not contain the required <code>loan_status</code> target attribute.<br><br>
                    <strong>How to proceed:</strong>
                    <ol style="margin-top: 0.5rem; margin-bottom: 0;">
                        <li><strong>Provide labeled data:</strong> Upload a historical dataset with repayment outcomes (e.g. click <em>'Labeled Data'</em> above).</li>
                        <li><strong>Switch to Unsupervised Learning:</strong> Select <strong>Option B: Unsupervised Learning</strong> above to segment borrowers with K-Means without requiring loan outcomes.</li>
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Execute Batch Prediction if not already cached
        if "upload_batch_results" not in st.session_state or load_btn:
            with st.spinner("Executing pre-fitted pipeline and calculating default probabilities..."):
                selected_model_name = batch_model if 'batch_model' in locals() else "Random Forest"
                l_th = low_th if 'low_th' in locals() else DEFAULT_LOW_THRESHOLD
                m_th = med_th if 'med_th' in locals() else DEFAULT_MED_THRESHOLD

                batch_res = predict_batch(
                    input_df,
                    model_name=selected_model_name,
                    low_threshold=l_th,
                    med_threshold=m_th
                )

            if not batch_res["success"]:
                st.error("❌ Batch prediction failed due to incompatible input features:")
                for err in batch_res["errors"]:
                    st.write(f"- {err}")
                return
            st.session_state["upload_batch_results"] = batch_res

        res = st.session_state["upload_batch_results"]
        summary = res["summary"]
        scored_df = res["scored_df"]

        # KPI Metrics Row
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
            st.markdown(render_metric_card("Avg Default Risk", f"{summary['avg_probability_pct']}%", "Mean probability", "#8B5CF6"), unsafe_allow_html=True)

        # Visualizations
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

        # Scored Applicants Table
        st.subheader("Scored Loan Applicants Table")
        st.caption("Each applicant row includes estimated default probability, risk tier, and original loan attributes.")
        st.dataframe(scored_df, use_container_width=True, height=350)

        # Download Button
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Scored Dataset (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"scored_{input_file_name}",
            mime="text/csv",
            use_container_width=True
        )

        # Drill-down to SHAP
        st.subheader("Drill-Down: Inspect Individual Applicant with SHAP")
        col_idx, col_go = st.columns([2, 1])
        with col_idx:
            sel_row = st.selectbox(
                "Choose an applicant row to inspect:",
                options=list(range(len(scored_df))),
                format_func=lambda i: f"Applicant #{i+1} — Tier: {scored_df.iloc[i]['Predicted_Risk_Tier']} ({scored_df.iloc[i]['Default_Probability (%)']}%)"
            )
        with col_go:
            st.write("")
            st.write("")
            if st.button("🔍 Explain with SHAP Waterfall", use_container_width=True):
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
                st.success(f"Applicant #{sel_row+1} loaded! Navigate to **🔍 Explainable AI (SHAP)** in the sidebar to view the attribution breakdown.")

        st.info("👉 **Supervised Mode Active:** You can now explore **🎯 Loan Default Prediction**, **🔍 Explainable AI (SHAP)**, **⚖️ Model Comparison**, and **🔄 What-If Analysis** in the sidebar navigation.")

    # ==================== WORKFLOW B: UNSUPERVISED LEARNING ====================
    elif active_mode == "unsupervised":
        st.markdown('<div class="section-header">👥 Unsupervised Borrower Segmentation (K-Means)</div>', unsafe_allow_html=True)

        k_val = n_clusters if 'n_clusters' in locals() else 3

        # Execute Clustering
        if "upload_clustering_results" not in st.session_state or load_btn:
            with st.spinner(f"Clustering {len(input_df):,} records into {k_val} borrower segments..."):
                try:
                    cluster_res = run_kmeans_segmentation(input_df, n_clusters=k_val)
                    st.session_state["upload_clustering_results"] = cluster_res
                except Exception as e:
                    st.error(f"❌ Clustering failed: {e}")
                    return

        c_res = st.session_state["upload_clustering_results"]

        # KPI Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(render_metric_card("Total Borrowers", f"{c_res['n_samples']:,}", f"Source: {input_file_name}", "#0EA5E9"), unsafe_allow_html=True)
        with m2:
            st.markdown(render_metric_card("Clusters (k)", f"{c_res['n_clusters']}", f"{len(c_res['features_used'])} Features Used", "#10B981"), unsafe_allow_html=True)
        with m3:
            sil_s = c_res["silhouette_score"]
            sil_txt = f"{sil_s:.3f}" if sil_s is not None else "N/A"
            st.markdown(render_metric_card("Silhouette Score", sil_txt, "Cluster Cohesion", "#8B5CF6"), unsafe_allow_html=True)
        with m4:
            pca_pct = sum(c_res["pca_explained_variance"]) * 100.0
            st.markdown(render_metric_card("2D PCA Variance", f"{pca_pct:.1f}%", f"PC1+PC2 representation", "#F59E0B"), unsafe_allow_html=True)

        # Visualizations
        col_v1, col_v2 = st.columns([1.1, 1.4])
        cluster_colors = ["#0F766E", "#0284C7", "#D97706", "#7C3AED", "#DC2626", "#059669"]

        with col_v1:
            st.markdown("##### Borrower Distribution by Segment")
            dist_df = c_res["distribution_df"]
            fig_cbar = px.bar(
                dist_df,
                x="Cluster",
                y="Borrower_Count",
                text=dist_df.apply(lambda r: f"{r['Borrower_Count']:,} ({r['Percentage']}%)", axis=1),
                color="Cluster",
                color_discrete_sequence=cluster_colors[:c_res['n_clusters']]
            )
            fig_cbar.update_traces(textposition='outside')
            fig_cbar.update_layout(
                showlegend=False,
                plot_bgcolor="#FFFFFF",
                yaxis=dict(title="Borrower Count", gridcolor="#F1F5F9"),
                height=350,
                margin=dict(t=20, b=30, l=40, r=20)
            )
            st.plotly_chart(fig_cbar, use_container_width=True)

        with col_v2:
            st.markdown("##### 2D Principal Component Projection (PCA)")
            pca_df = c_res["pca_df"]
            plot_pca = pca_df.sample(min(3000, len(pca_df)), random_state=42)
            fig_scatter = px.scatter(
                plot_pca,
                x="PCA_1",
                y="PCA_2",
                color="Cluster",
                color_discrete_sequence=cluster_colors[:c_res['n_clusters']],
                opacity=0.75
            )
            fig_scatter.update_layout(
                plot_bgcolor="#FFFFFF",
                xaxis=dict(title="Principal Component 1", gridcolor="#F1F5F9"),
                yaxis=dict(title="Principal Component 2", gridcolor="#F1F5F9"),
                height=350,
                margin=dict(t=20, b=30, l=40, r=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Cluster Financial Profiles Table
        st.subheader("Cluster Financial Profile Comparison")
        st.markdown("Mean financial metrics across discovered borrower segments:")
        profile_df = c_res["profile_df"].copy()
        st.dataframe(profile_df, use_container_width=True)

        # Interpretations
        st.subheader("Segment Statistical Interpretations")
        for c_lbl, desc in c_res["interpretations"].items():
            st.markdown(f"- {desc}")

        # Post-hoc default rate if available
        if c_res["target_available"]:
            st.subheader("Post-Hoc Historical Default Rate Comparison")
            eval_df = c_res["distribution_df"][["Cluster", "Borrower_Count", "Percentage", "Observed_Default_Rate (%)"]]
            st.dataframe(eval_df.set_index("Cluster"), use_container_width=True)

        # Download Clustered Dataset
        clustered_out = input_df.copy()
        clustered_out["Assigned_Cluster"] = [f"Cluster {i}" for i in c_res["labels"]]
        buf_c = io.StringIO()
        clustered_out.to_csv(buf_c, index=False)
        st.download_button(
            label="📥 Download Clustered Dataset with Segment Labels (CSV)",
            data=buf_c.getvalue(),
            file_name=f"clustered_{input_file_name}",
            mime="text/csv",
            use_container_width=True
        )

        st.info("👉 **Unsupervised Mode Active:** You can also explore the dedicated **👥 Borrower Segmentation (K-Means)** page in the sidebar navigation.")

    st.markdown(render_disclaimer(
        "Model Workflow Integrity Notice: Supervised loan default predictions are generated using pre-trained classifiers "
        "and require labeled historical data for supervised validation. Unsupervised clustering identifies natural behavioral "
        "segments without outcome labels. Results are provided for educational and analytical purposes."
    ), unsafe_allow_html=True)
