"""
cluster_counterfactuals.py
--------------------------
Feature 5: Counterfactual Borrower Recommendations for K-Means Clustering.
Discovers minimal, plausible modifications to actionable financial features
(loan amount, DTI, annual income) that transition a borrower to a different cluster.
Applies the exact same fitted preprocessing scaler and KMeans model.
Strictly preserves immutable credit history records.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.clustering import predict_borrower_cluster

# Credit history features that are immutable and must NEVER be modified in recourse
IMMUTABLE_FEATURES = [
    'delinq_2yrs',
    'pub_rec',
    'pub_rec_bankruptcies',
    'inq_last_6mths',
    'emp_length_years'
]


def recalculate_installment(principal: float, annual_rate_pct: float, months: int = 36) -> float:
    """Computes monthly installment using the standard amortization formula."""
    if months <= 0 or principal <= 0:
        return 0.0
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        return round(principal / months, 2)
    inst = principal * (r * (1 + r)**months) / ((1 + r)**months - 1)
    return round(inst, 2)


def generate_cluster_counterfactuals(
    borrower_input: Dict[str, Any],
    fitted_pipeline: Dict[str, Any],
    target_cluster: Optional[int] = None,
    max_recommendations: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Searches a bounded grid of plausible actionable adjustments:
      - Lowering requested loan amount (-15%, -25%, -35%, -50%)
      - Reducing debt-to-income ratio (-3%, -6%, -10%, -15%)
      - Increasing verified annual income (+20%, +35%, +50%)
      - Joint adjustments (loan reduction + DTI reduction)

    Adheres strictly to:
      1. Immutable historical records are never altered.
      2. Derived features (installment, loan_to_income, etc.) are recalculated consistently.
      3. Uses the exact fitted scaler & KMeans model instance.
      4. Verifies actual assignment to the new cluster.
    """
    if "target_cluster_id" in kwargs and target_cluster is None:
        target_cluster = kwargs["target_cluster_id"]

    # 1. Evaluate current cluster assignment
    current_res = predict_borrower_cluster(fitted_pipeline, borrower_input)
    current_cluster_id = current_res["cluster_id"]
    current_cluster_name = current_res["cluster_name"]
    current_semantic = current_res["semantic_name"]

    orig_scaled = current_res["X_scaled"]

    base_loan = float(borrower_input.get("loan_amnt", 15000))
    base_dti = float(borrower_input.get("dti", 18.0))
    base_inc = float(borrower_input.get("annual_inc", 55000))
    base_rate = float(str(borrower_input.get("int_rate_clean", borrower_input.get("int_rate", 12.5))).replace("%", ""))
    term_str = str(borrower_input.get("term", "36"))
    term_months = 36 if "36" in term_str else 60

    # 2. Build candidate perturbations
    candidates = []

    # A. Loan Amount Reductions
    for pct_drop in [0.15, 0.25, 0.35, 0.50, 0.65, 0.75]:
        new_loan = round(base_loan * (1.0 - pct_drop), -2)
        if new_loan >= 1000 and new_loan < base_loan:
            cand = dict(borrower_input)
            cand["loan_amnt"] = new_loan
            cand["installment"] = recalculate_installment(new_loan, base_rate, term_months)
            candidates.append({
                "type": "Single Feature",
                "feature": "loan_amnt",
                "feature_label": "Loan Amount",
                "original_val": f"${base_loan:,.0f}",
                "suggested_val": f"${new_loan:,.0f}",
                "change_desc": f"Reduce requested loan amount by {int(pct_drop*100)}% (ask for ${new_loan:,.0f} instead of ${base_loan:,.0f})",
                "input_data": cand
            })

    # B. DTI Reductions (paying down non-loan revolving debt)
    for dti_drop in [3.0, 6.0, 10.0, 15.0, 20.0]:
        new_dti = max(2.0, base_dti - dti_drop)
        if new_dti < base_dti:
            cand = dict(borrower_input)
            cand["dti"] = round(new_dti, 1)
            candidates.append({
                "type": "Single Feature",
                "feature": "dti",
                "feature_label": "Debt-to-Income (DTI)",
                "original_val": f"{base_dti:.1f}%",
                "suggested_val": f"{new_dti:.1f}%",
                "change_desc": f"Lower DTI from {base_dti:.1f}% to {new_dti:.1f}% via existing credit card payoff",
                "input_data": cand
            })

    # C. Annual Income Increases (adding verified co-borrower)
    for inc_boost in [0.20, 0.35, 0.50, 0.75, 1.00, 1.50]:
        new_inc = round(base_inc * (1.0 + inc_boost), -2)
        cand = dict(borrower_input)
        cand["annual_inc"] = new_inc
        candidates.append({
            "type": "Single Feature",
            "feature": "annual_inc",
            "feature_label": "Annual Income",
            "original_val": f"${base_inc:,.0f}",
            "suggested_val": f"${new_inc:,.0f}",
            "change_desc": f"Include verified secondary income or co-borrower (+${new_inc - base_inc:,.0f}/yr)",
            "input_data": cand
        })

    # D. Revolving Credit Card Paydown (Reducing revol_util_clean & revol_bal)
    base_util = float(borrower_input.get("revol_util_clean", 60.0))
    base_bal = float(borrower_input.get("revol_bal", 12000.0))
    for util_drop in [20.0, 40.0, 60.0]:
        new_util = max(5.0, base_util - util_drop)
        if new_util < base_util:
            ratio = new_util / max(1.0, base_util)
            new_bal = round(base_bal * ratio, -2)
            cand = dict(borrower_input)
            cand["revol_util_clean"] = round(new_util, 1)
            cand["revol_bal"] = new_bal
            candidates.append({
                "type": "Single Feature",
                "feature": "revol_util_clean",
                "feature_label": "Revolving Credit Utilization",
                "original_val": f"{base_util:.1f}%",
                "suggested_val": f"{new_util:.1f}%",
                "change_desc": f"Pay down credit card balances to drop utilization from {base_util:.1f}% to {new_util:.1f}%",
                "input_data": cand
            })

    # E. Joint Action: Loan Reduction + DTI Payoff
    for p_drop, d_drop in [(0.25, 6.0), (0.40, 10.0), (0.55, 15.0), (0.70, 18.0)]:
        j_loan = round(base_loan * (1.0 - p_drop), -2)
        j_dti = max(2.0, base_dti - d_drop)
        if j_loan >= 1000 and (j_loan < base_loan or j_dti < base_dti):
            cand_j = dict(borrower_input)
            cand_j["loan_amnt"] = j_loan
            cand_j["dti"] = round(j_dti, 1)
            cand_j["installment"] = recalculate_installment(j_loan, base_rate, term_months)
            candidates.append({
                "type": "Joint Action",
                "feature": "loan_amnt + dti",
                "feature_label": "Loan Amount & DTI",
                "original_val": f"${base_loan:,.0f} & {base_dti:.1f}%",
                "suggested_val": f"${j_loan:,.0f} & {j_dti:.1f}%",
                "change_desc": f"Request ${j_loan:,.0f} (-{int(p_drop*100)}%) AND pay down debt to achieve {j_dti:.1f}% DTI",
                "input_data": cand_j
            })

    # 3. Test candidates against the exact fitted K-Means pipeline
    valid_transitions = []

    for c in candidates:
        eval_res = predict_borrower_cluster(fitted_pipeline, c["input_data"])
        new_c_id = eval_res["cluster_id"]
        new_c_name = eval_res["cluster_name"]
        new_semantic = eval_res["semantic_name"]
        new_scaled = eval_res["X_scaled"]

        # Transition Condition: Must actually land in a different cluster
        if new_c_id != current_cluster_id:
            if target_cluster is not None and new_c_id != target_cluster:
                continue

            # Compute Euclidean distance in normalized feature space
            delta_dist = float(np.linalg.norm(new_scaled - orig_scaled))

            c["new_cluster_id"] = new_c_id
            c["target_cluster"] = new_c_id
            c["new_cluster_name"] = new_c_name
            c["new_semantic"] = new_semantic
            c["current_cluster_id"] = current_cluster_id
            c["current_cluster_name"] = current_cluster_name
            c["current_semantic"] = current_semantic
            c["normalized_distance"] = round(delta_dist, 3)
            c["euclidean_distance"] = round(delta_dist, 3)
            c["proposed_values"] = c["input_data"]
            valid_transitions.append(c)

    # Sort transitions by minimal perturbation distance (easiest recourse first)
    valid_transitions.sort(key=lambda x: x["normalized_distance"])

    top_recs = valid_transitions[:max_recommendations]

    msg = None
    if len(top_recs) == 0:
        msg = (
            "No feasible counterfactual modification was found within realistic search bounds "
            f"that transitions this profile out of {current_cluster_name}. "
            "This borrower's placement is anchored by fixed credit profile dimensions."
        )

    return {
        "success": True,
        "current_cluster": current_cluster_id,
        "current_cluster_id": current_cluster_id,
        "current_cluster_name": current_cluster_name,
        "current_semantic": current_semantic,
        "total_candidates_evaluated": len(candidates),
        "valid_count": len(top_recs),
        "recommendations": top_recs,
        "message": msg
    }
