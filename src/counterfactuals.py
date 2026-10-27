"""
counterfactuals.py
------------------
Future Scope Item 5: Actionable Counterfactual Recourse in What-If Analysis.
Explores plausible, minimal perturbations to actionable financial attributes
(loan amount, DTI, annual income) to demonstrate how adjustments alter predicted classification.
Strictly leaves immutable historical records (delinquencies, inquiries, bankruptcies) unchanged.
"""

from typing import Dict, Any, List, Optional
from src.prediction import predict_loan_default, DEFAULT_LOW_THRESHOLD, DEFAULT_MED_THRESHOLD

# Immutable credit history features that must NEVER be altered in actionable recourse
IMMUTABLE_FEATURES = [
    'delinq_2yrs',
    'pub_rec',
    'pub_rec_bankruptcies',
    'inq_last_6mths',
    'emp_length'
]


def recalculate_installment(principal: float, annual_rate_pct: float, months: int) -> float:
    """Calculates updated monthly installment based on principal, rate, and term."""
    if months <= 0 or principal <= 0:
        return 0.0
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        return round(principal / months, 2)
    inst = principal * (r * (1 + r)**months) / ((1 + r)**months - 1)
    return round(inst, 2)


def generate_actionable_counterfactuals(
    current_input: Dict[str, Any],
    model_name: str = "Random Forest",
    decision_threshold: float = 0.35,
    max_recommendations: int = 5
) -> Dict[str, Any]:
    """
    Searches a bounded grid of plausible modifications to actionable features:
      - Lowering requested loan amount (-15%, -25%, -35%, -50%)
      - Reducing debt-to-income ratio (-4%, -8%, -12%, -16%)
      - Increasing verified annual income (+20%, +35%, +50%)
      - Joint adjustments (loan reduction + DTI reduction)

    Adheres to:
      1. Immutable features are strictly preserved.
      2. Derived features (installment, etc.) are recalculated consistently.
      3. Only candidates that change the classification or risk tier are returned.
      4. Honest messaging when no valid counterfactual is found within bounds.
      5. Clear explanation if borrower is already non-default / low risk.
    """
    # 1. Baseline prediction
    base_res = predict_loan_default(current_input, model_name=model_name)
    if not base_res["success"]:
        return {
            "success": False,
            "errors": base_res["errors"],
            "recommendations": []
        }

    base_proba = float(base_res["probability"])
    base_tier = base_res["risk_tier"]
    base_class = 1 if base_proba >= decision_threshold else 0
    base_class_label = "Default Risk (Flagged)" if base_class == 1 else "Repay (Approved)"

    base_loan = float(current_input.get("loan_amnt", 15000))
    base_dti = float(current_input.get("dti", 18.0))
    base_inc = float(current_input.get("annual_inc", 55000))
    base_rate = float(str(current_input.get("int_rate", 13.5)).replace("%", ""))
    term_str = str(current_input.get("term", "36 months"))
    term_months = 36 if "36" in term_str else 60

    is_favorably_classified = (base_class == 0 and base_tier == "Low")

    # If already low risk, we do not fabricate recourse recommendations (Requirement 12)
    if is_favorably_classified:
        return {
            "success": True,
            "base_proba": base_proba,
            "base_tier": base_tier,
            "base_class": base_class,
            "base_class_label": base_class_label,
            "decision_threshold": decision_threshold,
            "is_favorably_classified": True,
            "total_searched": 0,
            "valid_count": 0,
            "recommendations": [],
            "message": (
                f"This borrower is already favorably classified as Non-Default ({base_tier} Risk, "
                f"estimated default probability: {base_proba*100:.1f}% vs threshold {decision_threshold*100:.1f}%). "
                "Actionable recourse search is designed to identify paths to approval for borrowers facing "
                "elevated risk classifications. No financial concessions or adjustments are required for this profile."
            )
        }

    # 2. Build candidate perturbations for elevated risk borrowers
    candidates = []

    # A. Loan Amount Reductions
    for pct_drop in [0.15, 0.25, 0.35, 0.50]:
        new_loan = round(base_loan * (1.0 - pct_drop), -2)
        if new_loan >= 1000 and new_loan < base_loan:
            cand = dict(current_input)
            cand["loan_amnt"] = new_loan
            cand["installment"] = recalculate_installment(new_loan, base_rate, term_months)
            candidates.append({
                "type": "Single Feature",
                "feature": "loan_amnt",
                "feature_label": "Loan Amount",
                "original_val": f"${base_loan:,.0f}",
                "suggested_val": f"${new_loan:,.0f}",
                "change_desc": f"Reduce loan request by {int(pct_drop*100)}% (-${base_loan - new_loan:,.0f})",
                "input_data": cand
            })

    # B. DTI Reductions (paying down non-mortgage revolving balances before applying)
    for dti_drop in [4.0, 8.0, 12.0, 16.0]:
        new_dti = max(2.0, base_dti - dti_drop)
        if new_dti < base_dti:
            cand = dict(current_input)
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

    # C. Annual Income Increases (adding co-borrower / verified secondary earnings)
    for inc_boost in [0.20, 0.35, 0.50]:
        new_inc = round(base_inc * (1.0 + inc_boost), -2)
        cand = dict(current_input)
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

    # D. Joint Action: Moderate Loan Reduction + Moderate DTI Improvement
    for p_drop, d_drop in [(0.20, 5.0), (0.30, 8.0), (0.40, 10.0)]:
        j_loan = round(base_loan * (1.0 - p_drop), -2)
        j_dti = max(2.0, base_dti - d_drop)
        if j_loan >= 1000 and (j_loan < base_loan or j_dti < base_dti):
            cand_j = dict(current_input)
            cand_j["loan_amnt"] = j_loan
            cand_j["dti"] = round(j_dti, 1)
            cand_j["installment"] = recalculate_installment(j_loan, base_rate, term_months)
            candidates.append({
                "type": "Joint Action",
                "feature": "loan_amnt + dti",
                "feature_label": "Loan Amount & DTI",
                "original_val": f"${base_loan:,.0f} & {base_dti:.1f}%",
                "suggested_val": f"${j_loan:,.0f} & {j_dti:.1f}%",
                "change_desc": f"Request ${j_loan:,.0f} (-{int(p_drop*100)}%) AND pay down debt to reach {j_dti:.1f}% DTI",
                "input_data": cand_j
            })

    # 3. Evaluate candidates
    valid_recommendations = []

    for c in candidates:
        res = predict_loan_default(c["input_data"], model_name=model_name)
        if not res["success"]:
            continue

        c_proba = float(res["probability"])
        c_tier = res["risk_tier"]
        c_class = 1 if c_proba >= decision_threshold else 0
        c_class_label = "Default Risk (Flagged)" if c_class == 1 else "Repay (Approved)"

        # Check for classification change:
        # 1. Binary classification flip: was class 1 (default), now class 0 (approved)
        # 2. Risk tier transition: from High to Medium/Low, or from Medium to Low
        flipped = False
        if base_class == 1 and c_class == 0:
            flipped = True
        elif base_tier == "High" and c_tier in ["Medium", "Low"]:
            flipped = True
        elif base_tier == "Medium" and c_tier == "Low":
            flipped = True

        if flipped:
            c["new_proba"] = c_proba
            c["new_tier"] = c_tier
            c["new_class"] = c_class
            c["new_class_label"] = c_class_label
            c["base_proba"] = base_proba
            c["base_tier"] = base_tier
            c["base_class"] = base_class
            c["base_class_label"] = base_class_label
            c["proba_reduction_pct"] = (base_proba - c_proba) * 100.0
            valid_recommendations.append(c)

    # Sort candidates by greatest probability improvement
    valid_recommendations.sort(key=lambda x: x["proba_reduction_pct"], reverse=True)
    top_recommendations = valid_recommendations[:max_recommendations]

    msg = None
    if len(top_recommendations) == 0:
        msg = (
            "No valid counterfactual recommendation found within plausible search bounds. "
            "The applicant's elevated default risk is driven by immutable credit bureau factors "
            "(e.g., past delinquencies or public records) that cannot be altered through loan amount "
            "or DTI adjustments alone."
        )

    return {
        "success": True,
        "base_proba": base_proba,
        "base_tier": base_tier,
        "base_class": base_class,
        "base_class_label": base_class_label,
        "decision_threshold": decision_threshold,
        "is_favorably_classified": False,
        "total_searched": len(candidates),
        "valid_count": len(top_recommendations),
        "recommendations": top_recommendations,
        "message": msg
    }
