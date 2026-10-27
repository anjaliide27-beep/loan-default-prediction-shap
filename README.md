# Unsupervised Learning for Borrower Segmentation Using K-Means Clustering

An interactive, responsive financial analytics web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** for objective, label-free credit risk segmentation, cost-sensitive decision support, and actionable counterfactual recourse.

---

## 📌 Project Overview

Traditional credit scoring models rely heavily on historical loan outcome labels (`default` vs. `non-default`), which often inherit historical underwriting bias, macroeconomic noise, and regulatory friction. 

This project implements an **end-to-end unsupervised machine learning architecture** that segments borrowers strictly based on their intrinsic financial and debt-capacity profiles, completely independent of historical default labels:

- **Strict Label-Free Preprocessing:** Completely excludes target outcomes (`loan_status`), personal identifiers (`id`, `member_id`), and post-origination leakage columns (`total_pymnt`, `recoveries`, `last_pymnt_amnt`).
- **Standardized Multi-Dimensional Feature Space:** Cleans, imputes missing values, and normalizes 15 core financial attributes using `StandardScaler` to ensure unbiased Euclidean distance calculations.
- **K-Means Borrower Segmentation:** Partitions cohorts into cohesive borrower personas with user-configurable cluster counts ($k \in [2, 6]$, defaulting to 3), reproducible initialization (`random_state=42`, `n_init=10`), and silhouette score quantification.
- **Clustering Quality & Performance Evaluation:** Rigorously quantifies cluster compactness and separation using the Davies–Bouldin Index (DBI), Silhouette Coefficient, and Inertia (WCSS).
- **Cluster Insights & Statistical Profiling:** Generates comprehensive mean financial metric tables, standardized Z-score heatmaps, and objective, non-judgmental semantic personas (e.g., *Moderate-Principal • Balanced-Debt*, *Low-Principal • Low-Leverage*).
- **Cost-Sensitive Decision Support (Feature 1):** Translates borrower segments into operational portfolio servicing strategies, comparing flat uniform servicing against segment-tailored monitoring and user-customized intervention costs.
- **Actionable Counterfactual Recourse (Feature 5):** Calculates minimal, plausible adjustments to controllable financial attributes (loan amount, DTI, verified income) that transition a borrower into a different target segment, while strictly holding immutable credit history constant.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10 to 3.14
- **Web Interface:** Streamlit (Custom Responsive Theme & CSS)
- **Data Manipulation:** Pandas, NumPy
- **Unsupervised Machine Learning:** Scikit-Learn (`KMeans`, `StandardScaler`, `SimpleImputer`, `davies_bouldin_score`, `silhouette_score`)
- **Interactive Visualization:** Plotly Graph Objects & Express

---

## 📂 Project Structure

```
AIML-MINIPROJECT/
│
├── data/
│   └── loan_train.csv          # LendingClub loan dataset (27,003 records)
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data cleaning, financial ratio engineering, leakage guards
│   ├── clustering.py           # Core K-Means clustering, Davies-Bouldin evaluation, persona labeling
│   ├── cost_support.py         # Feature 1: Cost-Sensitive Decision Support & strategy comparison
│   ├── cluster_counterfactuals.py # Feature 5: Counterfactual Borrower Recourse & Euclidean distance
│   └── ui_pages/
│       ├── __init__.py
│       ├── styles.py           # Dashboard styling, metric cards, callouts
│       ├── home.py             # Page 1: Executive Dashboard & Feature Space
│       ├── upload_page.py      # Page 2: CSV Upload & Dynamic K-Means Pipeline
│       ├── clustering_page.py  # Page 3: Borrower Segmentation (K-Means)
│       ├── cluster_insights.py # Page 4: Cluster Insights & Relative Heatmaps
│       ├── cost_support_page.py# Page 5: Cost-Sensitive Decision Support
│       ├── whatif_cluster_page.py # Page 6: What-If & Counterfactual Recourse
│       └── about_page.py       # Page 7: Architectural Reference & Methodology
│
├── tests/
│   ├── __init__.py
│   └── test_application.py     # 14-point automated verification test suite
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

### 5. Run Automated Verification Tests
Execute the full test suite to verify all 14 unsupervised requirements:
```bash
python -m tests.test_application
```

### 6. Launch the Web Application
Start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 📊 Application Pages Overview

| # | Page | Functionality |
|---|------|---------------|
| 1 | **🏠 Home Dashboard** | Executive summary, portfolio metrics, feature catalog, and architecture overview. |
| 2 | **📁 Upload Dataset** | Upload custom CSV cohorts, preview features, configure $k$, and cluster immediately. |
| 3 | **👥 Borrower Segmentation (K-Means)** | Interactive $k$ slider (2–6), Davies–Bouldin Index, Silhouette Score, WCSS, and cluster profiles. |
| 4 | **📊 Cluster Insights & Profiles** | Segment persona cards, financial distributions (box plots), and standardized Z-score heatmaps. |
| 5 | **💰 Cost-Sensitive Decision Support** | Evaluates portfolio servicing costs across clusters; compares uniform vs. segment-tailored policies. |
| 6 | **🔄 What-If & Counterfactual Recourse** | Interactive parameter perturbation and algorithmic recommendations for transitioning clusters. |
| 7 | **ℹ️ About Project** | Mathematical formulas, distance metrics, regulatory compliance, and architectural documentation. |

---

## ⚖️ Regulatory Notice & Disclaimers

This software is an educational and analytical research platform developed for credit portfolio segmentation. Discovered clusters represent statistical mathematical proximities within standardized financial feature space and should not be used as the sole determinant for credit decisions without human review and regulatory compliance checks under the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA).
