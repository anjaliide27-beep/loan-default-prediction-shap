# Explainable Loan Default Prediction Using Machine Learning and SHAP

An interactive, responsive financial analytics web application built with **Streamlit**, **Scikit-Learn**, and **SHAP** (SHapley Additive exPlanations) for transparent credit risk evaluation.

---

## 📌 Project Overview

Traditional credit scoring models often operate as "black boxes," obscuring the critical financial attributes that influence credit approval or denial. Under modern financial regulations (such as the Equal Credit Opportunity Act and Fair Credit Reporting Act), creditors are legally and ethically obligated to supply clear Adverse Action reasons when credit is denied.

This project delivers:
- **Application-Time Feature Engineering:** Purges 12 post-loan leakage variables (`total_pymnt`, `recoveries`, `last_pymnt_amnt`, etc.) to guarantee genuine out-of-sample validity.
- **Supervised Machine Learning Comparison:** Compares Logistic Regression, Decision Trees, Random Forests, and XGBoost with cost-sensitive class-imbalance compensation.
- **Explainable AI with SHAP:** Computes exact, local feature attributions using cooperative game theory (Shapley values) for both applicant-level waterfall charts and global feature importance.
- **What-If Counterfactual Simulator:** Allows users to interactively perturb financial parameters (income, debt-to-income, loan amount) and observe the resulting shifts in estimated default probability.

---

## 🛠️ Technology Stack

- **Python:** 3.10 to 3.14
- **Web Interface:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Explainable AI (XAI):** SHAP (`TreeExplainer`)
- **Data Visualization:** Plotly Graph Objects & Express
- **Model Serialization:** Joblib

---

## 📂 Project Structure

```
Explainable-Loan-Default/
│
├── data/
│   └── loan_train.csv          # LendingClub historical loan dataset (27,003 records)
│
├── models/                     # Saved model artifacts & metrics
│   ├── preprocessor.joblib     # Fitted ColumnTransformer (scaling & one-hot encoding)
│   ├── models.joblib           # Dictionary of trained classifiers
│   ├── rf_explainer.joblib     # Pre-fitted SHAP TreeExplainer
│   ├── metrics.json            # Cross-validated test metrics
│   └── metadata.json           # Feature names and dataset statistics
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Cleaning logic, leakage filters, ColumnTransformer
│   ├── train_model.py          # Stratified train/test split, model training & evaluation
│   ├── prediction.py           # Inference engine, input validation, risk categorization
│   ├── explainability.py       # Local waterfall plots, global importance, natural summaries
│   └── ui_pages/
│       ├── __init__.py
│       ├── styles.py           # Custom CSS, cards, and theme formatting
│       ├── home.py             # Page 1: Executive Dashboard & Dataset Summary
│       ├── eda.py              # Page 2: Exploratory Data Analysis & Correlations
│       ├── prediction_page.py  # Page 3: Interactive Applicant Scoring Form
│       ├── shap_page.py        # Page 4: Explainable AI with SHAP
│       ├── models_page.py      # Page 5: Model Evaluation & Metric Comparison
│       └── whatif_page.py      # Page 6: Counterfactual Scenario Simulator
│
├── app.py                      # Streamlit application entry point
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone or Navigate to Directory
```bash
cd "c:\Users\ACER\OneDrive\Documents\Desktop\AIML MINIPROJECT"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Dataset Placement
Ensure `loan_train.csv` is situated in the `data/` folder:
```
data/loan_train.csv
```

### 5. Train and Serialize Models
Run the training pipeline to generate models, compute test metrics, and fit the SHAP explainers:
```bash
python -m src.train_model
```

### 6. Launch the Web Application
Start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🌐 Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (ensure `data/loan_train.csv` and `models/` artifacts are included or generated via build script).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your GitHub repository and branch.
4. Set the **Main file path** to `app.py`.
5. Click **"Deploy!"**.

---

## 📊 Summary of Evaluated Models (Unseen Test Data)

| Model Name | ROC-AUC | Recall (Sensitivity) | Precision | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Primary)** | **0.7033** | **0.5849** | **0.2489** | **0.3492** | **0.6791** |
| **XGBoost Classifier** | 0.7058 | 0.5874 | 0.2518 | 0.3525 | 0.6823 |
| **Logistic Regression** | 0.7103 | 0.6352 | 0.2412 | 0.3496 | 0.6521 |
| **Decision Tree** | 0.6839 | 0.5912 | 0.2369 | 0.3383 | 0.6595 |

*Note: Models were trained using cost-sensitive class balancing to penalize false negatives (missed defaults) appropriately in an imbalanced 14.73% default environment.*

---

## ⚖️ Academic, Ethical & Regulatory Disclaimers

1. **Academic Demonstration Only:** This software is an educational prototype developed for an AIML academic capstone. It has received no regulatory certification, credit bureau approval, or commercial banking validation.
2. **Domain Boundaries:** The models were trained on US peer-to-peer lending records (LendingClub) and **are not representative of Indian microfinance borrowers or emerging-market fintech credit dynamics**.
3. **Attribution vs. Causation:** SHAP values represent mathematical feature contributions within the model function; they do not prove physical causality or guarantee future repayment performance.
