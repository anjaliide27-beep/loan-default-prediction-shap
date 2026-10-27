"""
app.py
------
Main Streamlit application entry point for:
Unsupervised Learning for Borrower Segmentation Using K-Means Clustering.
"""

import streamlit as st

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="Borrower Segmentation Using K-Means",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Import modular page renderers
from src.ui_pages.styles import get_custom_css
from src.ui_pages.home import render_home_page
from src.ui_pages.upload_page import render_upload_page
from src.ui_pages.clustering_page import render_clustering_page
from src.ui_pages.cluster_insights import render_cluster_insights_page
from src.ui_pages.cost_support_page import render_cost_support_page
from src.ui_pages.whatif_cluster_page import render_whatif_cluster_page
from src.ui_pages.about_page import render_about_page

# 3. Inject Custom Styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


def main():
    # Sidebar Branding
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size: 2.2rem;">👥</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-top: 4px;">Borrower Segmentation</div>
        <div style="font-size: 0.72rem; color: #0D9488; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
            Unsupervised Machine Learning
        </div>
    </div>
    <hr style="margin: 0.75rem 0 1.25rem 0; border: 0; border-top: 1px solid #E2E8F0;">
    """, unsafe_allow_html=True)

    # Status badge in sidebar
    st.sidebar.markdown("""
    <div style="background-color: #F0FDFA; border: 1px solid #99F6E4; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.85rem; font-size: 0.82rem;">
        <div style="font-weight: 700; color: #0F766E;">⚡ Unsupervised K-Means Pipeline</div>
        <div style="color: #115E59; font-size: 0.75rem;">15 Financial Features • Standardized PCA</div>
    </div>
    """, unsafe_allow_html=True)

    # 7 Pure Unsupervised Pages
    pages = {
        "🏠 Home Dashboard": render_home_page,
        "📁 Upload Dataset": render_upload_page,
        "👥 Borrower Segmentation (K-Means)": render_clustering_page,
        "📊 Cluster Insights & Profiles": render_cluster_insights_page,
        "💰 Cost-Sensitive Decision Support": render_cost_support_page,
        "🔄 What-If & Counterfactual Recourse": render_whatif_cluster_page,
        "ℹ️ About Project": render_about_page,
    }

    page_options = list(pages.keys())
    saved_page = st.session_state.get("active_page")
    default_index = page_options.index(saved_page) if saved_page in page_options else 0

    selected_page_name = st.sidebar.radio(
        "Navigation Menu",
        options=page_options,
        index=default_index,
        label_visibility="collapsed"
    )
    st.session_state["active_page"] = selected_page_name

    st.sidebar.markdown("---")

    # App Info & Footnote
    st.sidebar.markdown("""
    <div style="padding-top: 1.5rem; font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">
        <strong>AIML Mini-Project</strong><br>
        Algorithm: K-Means Clustering<br>
        Dimensionality: 2D PCA Projection<br>
        Decision: Segment Cost Optimization<br>
        Recourse: Euclidean Counterfactuals<br>
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
        <div>Unsupervised Learning for Borrower Segmentation Using K-Means Clustering</div>
        <div>AIML Mini-Project • Python & Streamlit</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
