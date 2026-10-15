"""
about_page.py
-------------
Page 7: Academic documentation, ethical considerations, methodology, and limitations.
"""

import streamlit as st
from src.ui_pages.styles import render_disclaimer


def render_about_page():
    st.markdown('<div class="section-header">About the Project: Technical & Ethical Documentation</div>', unsafe_allow_html=True)

    st.subheader("1. Problem Statement & Motivation")
    st.markdown("""
    In the digital lending and fintech sector, algorithmic credit underwriting models evaluate loan applicants at high velocity. However, advanced models such as Gradient Boosting Machines and Random Forests operate as complex non-linear functions ("black boxes"). When a loan application is rejected or subjected to elevated interest rates, lenders are legally and ethically obligated to explain the governing reasons.

    This project demonstrates an **Explainable Machine Learning System** that pairs state-of-the-art predictive modeling with **SHapley Additive exPlanations (SHAP)**, providing mathematical transparency and local auditability for individual credit decisions.
    """)

    st.subheader("2. Dataset Source & Contextual Boundaries")
    st.markdown("""
    - **Dataset Origin:** The underlying dataset consists of **27,003 peer-to-peer loan records** sourced from LendingClub's historical lending operations (2007–2011).
    - **Geographical & Economic Context:** All records represent US-based consumer borrowing with prime/near-prime credit bureau tracking.
    - **Crucial Domain Boundary:** **This dataset is not representative of Indian microfinance, rural cooperative banking, or emerging-market fintech borrowers.** Applying models trained on US consumer credit records directly to non-US jurisdictions without localized retraining and calibration introduces severe domain shift.
    """)

    st.subheader("3. Feature Engineering & Leakage Prevention")
    st.markdown("""
    A fundamental requirement of real-world machine learning engineering is **temporal integrity**:
    - **Application-Time Features Only:** We restrict model inputs strictly to attributes present at the exact time of loan origination (loan amount, term, self-reported income, debt-to-income, and historical credit lines).
    - **Strict Elimination of Post-Origination Leakage:** 12 columns recording post-loan repayments, collections, and recoveries (`total_pymnt`, `recoveries`, `last_pymnt_amnt`, etc.) were systematically purged from training to ensure genuine out-of-sample validity.
    """)

    st.subheader("4. Supervised Machine Learning & Evaluation Methodology")
    st.markdown("""
    Four candidate classifiers were evaluated:
    1. **Logistic Regression:** Serves as the transparent linear baseline.
    2. **Decision Tree Classifier:** Captures non-linear decision splits.
    3. **Random Forest Classifier:** Primary ensemble model providing optimal ROC-AUC and continuous probability estimates.
    4. **XGBoost:** Gradient boosted tree benchmark.

    **Imbalance Strategy & Metric Hierarchy:**
    Due to the 14.73% minority default distribution, naive accuracy is discarded. Models are evaluated using **ROC-AUC, Recall (Sensitivity to defaults), Precision, and F1-score** computed on a held-out 20% stratified test set.
    """)

    st.subheader("5. Explainable AI with SHAP")
    st.markdown("""
    SHAP provides local feature attributions rooted in coalitional game theory:
    - **Efficiency Axiom:** The sum of SHAP attributions plus the population base rate equals the applicant's predicted default probability.
    - **Interpretability:** Deconstructs positive factors (increasing risk) and negative factors (mitigating risk) for regulatory **Adverse Action Notices**.
    """)

    st.subheader("6. Ethical Considerations & Responsible AI")
    st.markdown("""
    - **Fair Lending Compliance:** Credit models must prevent discriminatory disparate impact against protected demographic attributes (race, gender, religion, marital status).
    - **Data Minimization:** No sensitive personal identifiers (`id`, `url`, `desc`, `emp_title`, `zip_code`) were utilized in model fitting.
    - **Transparency Over Secrecy:** Black-box rejections without clear remediation guidance harm financial inclusion. Explainability provides actionable feedback for borrowers to improve credit health.
    """)

    st.subheader("7. Academic Disclaimer & Limitations")
    st.markdown(render_disclaimer(
        "This application is an educational capstone project developed for academic demonstration purposes. "
        "It carries no regulatory approval, no banking validation, and no fairness certification. "
        "It must not be deployed as an autonomous decision system in live financial environments."
    ), unsafe_allow_html=True)
