"""
app.py
------
Main Streamlit application entry point for:
Explainable Loan Default Prediction Using Machine Learning and SHAP.
"""

import os
import streamlit as st

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="Explainable Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Import modular page renderers
from src.ui_pages.styles import get_custom_css
from src.ui_pages.home import render_home_page
from src.ui_pages.upload_page import render_upload_page
from src.ui_pages.eda import render_eda_page
from src.ui_pages.prediction_page import render_prediction_page
from src.ui_pages.shap_page import render_shap_page
from src.ui_pages.models_page import render_models_page
from src.ui_pages.whatif_page import render_whatif_page

# 3. Inject Custom Styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


def main():
    # Sidebar Branding
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size: 2.2rem;">🏦</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-top: 4px;">Loan Default XAI</div>
        <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
            Explainable AI & ML Dashboard
        </div>
    </div>
    <hr style="margin: 0.75rem 0 1.25rem 0; border: 0; border-top: 1px solid #E2E8F0;">
    """, unsafe_allow_html=True)

    # Navigation Menu
    pages = {
        "🏠 Home Dashboard": render_home_page,
        "📁 Upload Dataset & Predict Risk": render_upload_page,
        "📊 Exploratory Data Analysis": render_eda_page,
        "🎯 Loan Default Prediction": render_prediction_page,
        "🔍 Explainable AI (SHAP)": render_shap_page,
        "⚖️ Model Comparison": render_models_page,
        "🔄 What-If Analysis": render_whatif_page
    }

    selected_page_name = st.sidebar.radio(
        "Navigation Menu",
        options=list(pages.keys()),
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # App Info & Footnote
    st.sidebar.markdown("""
    <div style="padding-top: 1.5rem; font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">
        <strong>AIML Mini-Project</strong><br>
        Framework: Streamlit & Scikit-Learn<br>
        Explainability: SHAP TreeExplainer<br>
        Dataset: LendingClub (27,003 records)
    </div>
    """, unsafe_allow_html=True)

    # Render Active Page
    render_func = pages[selected_page_name]
    render_func()

    # Universal Footer
    st.markdown("""
    <hr style="margin-top: 3rem; margin-bottom: 1rem; border: 0; border-top: 1px solid #E2E8F0;">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #94A3B8;">
        <div>Explainable Loan Default Prediction Using Machine Learning and SHAP</div>
        <div>AIML Mini-Project • Python & Streamlit</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
