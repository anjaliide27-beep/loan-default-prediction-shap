"""
data_processing.py
------------------
Module for data loading, preprocessing, cleaning, and transformation pipelines
for Explainable Loan Default Prediction.
"""

import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Explicit list of application-time predictive features
NUMERICAL_FEATURES = [
    'loan_amnt',
    'term_months',
    'int_rate_clean',
    'installment',
    'emp_length_years',
    'annual_inc',
    'dti',
    'delinq_2yrs',
    'inq_last_6mths',
    'open_acc',
    'pub_rec',
    'revol_bal',
    'revol_util_clean',
    'total_acc',
    'pub_rec_bankruptcies',
    'loan_to_income',
    'installment_to_income',
    'revol_to_income'
]

CATEGORICAL_FEATURES = [
    'grade',
    'home_ownership_clean',
    'verification_status',
    'purpose'
]

TARGET_COL = 'loan_status'

# Excluded columns cataloged by reason to avoid data leakage and arbitrary IDs
LEAKAGE_COLUMNS = [
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'last_credit_pull_d'
]

IDENTIFIER_COLUMNS = [
    'id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code'
]

REDUNDANT_POST_ORIGINATION_COLUMNS = [
    'funded_amnt', 'funded_amnt_inv', 'issue_d', 'sub_grade', 'addr_state',
    'earliest_cr_line', 'mths_since_last_delinq', 'mths_since_last_record'
]

# Human-friendly feature labels for display and SHAP explanations
FEATURE_LABELS = {
    'loan_amnt': 'Loan Amount ($)',
    'term_months': 'Loan Term (Months)',
    'int_rate_clean': 'Interest Rate (%)',
    'installment': 'Monthly Installment ($)',
    'emp_length_years': 'Employment Length (Years)',
    'annual_inc': 'Annual Income ($)',
    'dti': 'Debt-to-Income Ratio (DTI %)',
    'delinq_2yrs': 'Delinquencies in Past 2 Yrs',
    'inq_last_6mths': 'Inquiries in Past 6 Months',
    'open_acc': 'Open Credit Lines',
    'pub_rec': 'Public Derogatory Records',
    'revol_bal': 'Revolving Balance ($)',
    'revol_util_clean': 'Revolving Utilization (%)',
    'total_acc': 'Total Credit Lines',
    'pub_rec_bankruptcies': 'Bankruptcies',
    'loan_to_income': 'Loan-to-Income Ratio',
    'installment_to_income': 'Installment-to-Income Ratio',
    'revol_to_income': 'Revolving Debt-to-Income Ratio',
    'grade': 'Loan Grade',
    'home_ownership_clean': 'Home Ownership',
    'verification_status': 'Income Verification',
    'purpose': 'Loan Purpose'
}

EMP_LENGTH_MAP = {
    '< 1 year': 0.0,
    '1 year': 1.0,
    '2 years': 2.0,
    '3 years': 3.0,
    '4 years': 4.0,
    '5 years': 5.0,
    '6 years': 6.0,
    '7 years': 7.0,
    '8 years': 8.0,
    '9 years': 9.0,
    '10+ years': 10.0
}


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw LendingClub dataset into standard format for training/EDA.
    """
    data = df.copy()

    # Extract term in months as numerical (e.g., ' 36 months' -> 36.0)
    if 'term' in data.columns and 'term_months' not in data.columns:
        data['term_months'] = (
            data['term']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

    # Clean interest rate percentage (e.g., '13.23%' -> 13.23)
    if 'int_rate' in data.columns and 'int_rate_clean' not in data.columns:
        data['int_rate_clean'] = (
            data['int_rate']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
            .astype(float)
        )

    # Clean revolving line utilization (e.g., '62.70%' -> 62.70)
    if 'revol_util' in data.columns and 'revol_util_clean' not in data.columns:
        data['revol_util_clean'] = pd.to_numeric(
            data['revol_util'].astype(str).str.replace('%', '', regex=False).str.strip(),
            errors='coerce'
        )

    # Clean employment length to numerical years
    if 'emp_length' in data.columns and 'emp_length_years' not in data.columns:
        data['emp_length_years'] = data['emp_length'].map(EMP_LENGTH_MAP)

    # Standardize home ownership (group rare 'NONE' or 'OTHER' together)
    if 'home_ownership' in data.columns and 'home_ownership_clean' not in data.columns:
        data['home_ownership_clean'] = data['home_ownership'].replace({'NONE': 'OTHER'})

    # Financial ratio engineering
    if 'annual_inc' in data.columns and 'loan_amnt' in data.columns:
        data['loan_to_income'] = pd.to_numeric(data['loan_amnt'], errors='coerce') / (pd.to_numeric(data['annual_inc'], errors='coerce') + 1.0)
    if 'annual_inc' in data.columns and 'installment' in data.columns:
        data['installment_to_income'] = (pd.to_numeric(data['installment'], errors='coerce') * 12.0) / (pd.to_numeric(data['annual_inc'], errors='coerce') + 1.0)
    if 'annual_inc' in data.columns and 'revol_bal' in data.columns:
        data['revol_to_income'] = pd.to_numeric(data['revol_bal'], errors='coerce') / (pd.to_numeric(data['annual_inc'], errors='coerce') + 1.0)

    return data


def get_preprocessor() -> ColumnTransformer:
    """
    Creates and returns the Scikit-learn ColumnTransformer preprocessing pipeline.
    Prevents data leakage by encapsulating imputation and scaling.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        verbose_feature_names_out=True
    )
    return preprocessor


def get_friendly_feature_name(encoded_name: str) -> str:
    """
    Translates encoded ColumnTransformer feature names into clean, readable labels.
    Example: 'num__int_rate_clean' -> 'Interest Rate (%)'
             'cat__grade_A' -> 'Loan Grade: A'
             'cat__home_ownership_clean_RENT' -> 'Home Ownership: RENT'
    """
    if encoded_name.startswith('num__'):
        raw_col = encoded_name.replace('num__', '')
        return FEATURE_LABELS.get(raw_col, raw_col.replace('_', ' ').title())

    if encoded_name.startswith('cat__'):
        remainder = encoded_name.replace('cat__', '')
        for cat_col in CATEGORICAL_FEATURES:
            if remainder.startswith(cat_col + '_'):
                val = remainder[len(cat_col) + 1:]
                col_label = FEATURE_LABELS.get(cat_col, cat_col.replace('_', ' ').title())
                return f"{col_label}: {val}"

    return encoded_name.replace('_', ' ').title()


def format_user_input(input_dict: dict) -> pd.DataFrame:
    """
    Takes raw dictionary of user inputs from Streamlit form and formats it
    into a single-row DataFrame ready for the preprocessing pipeline.
    """
    df_row = pd.DataFrame([input_dict])
    cleaned_df = clean_raw_data(df_row)
    
    # Ensure all required features are present
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for col in expected_cols:
        if col not in cleaned_df.columns:
            cleaned_df[col] = np.nan

    return cleaned_df[expected_cols]
