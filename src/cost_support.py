"""
cost_support.py
---------------
Feature 1: Cost-Sensitive Decision Support for Unsupervised Borrower Segmentation.
Evaluates portfolio management action costs across borrower clusters.
Compares alternative segment-tailored operational policies against uniform baselines.
Does NOT use default probabilities, classification thresholds, or supervised models.
"""

from typing import Dict, Any, List
import pandas as pd


# Default illustrative operational actions and per-borrower action costs ($)
DEFAULT_ACTION_PRESETS = {
    "Automated Low-Touch Servicing": 25.0,
    "Standard Account Monitoring": 100.0,
    "Enhanced Credit Health Counseling": 250.0,
    "Proactive Limit Review & Restructuring": 400.0
}


def calculate_segment_costs(
    cluster_counts: Dict[str, int],
    unit_costs: Dict[str, float]
) -> Dict[str, Any]:
    """
    Computes total operational expenditure for a specific strategy across clusters.
    """
    segment_details = []
    total_cost = 0.0
    total_borrowers = sum(cluster_counts.values())

    for cluster_name, count in cluster_counts.items():
        cost_per_borrower = float(unit_costs.get(cluster_name, 100.0))
        seg_total = count * cost_per_borrower
        total_cost += seg_total
        pct_of_total_cost = (seg_total / total_cost * 100.0) if total_cost > 0 else 0.0

        segment_details.append({
            "Cluster": cluster_name,
            "Borrower_Count": count,
            "Cost_per_Borrower ($)": cost_per_borrower,
            "Total_Segment_Cost ($)": seg_total,
            "Proportion_of_Borrowers (%)": round(count / max(1, total_borrowers) * 100.0, 1)
        })

    df = pd.DataFrame(segment_details)
    if total_cost > 0:
        df["Cost_Share (%)"] = (df["Total_Segment_Cost ($)"] / total_cost * 100.0).round(1)
    else:
        df["Cost_Share (%)"] = 0.0

    avg_cost = total_cost / max(1, total_borrowers)

    return {
        "total_cost": total_cost,
        "total_borrowers": total_borrowers,
        "average_cost_per_borrower": round(avg_cost, 2),
        "details_df": df
    }


def compare_operational_strategies(
    cluster_counts: Any = None,
    user_custom_costs: Any = None,
    uniform_unit_cost: float = 120.0,
    tailored_presets: Optional[Dict[str, float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compares three alternative borrower management strategies:
      1. Uniform Servicing Policy (flat rate across all borrowers)
      2. Segment-Tailored Operational Policy (differential based on leverage/profile)
      3. User-Configured Strategy (interactive sliders)
    """
    # Flexibility: Support distribution_df passed as keyword or positional
    dist_df = kwargs.get("distribution_df", None)
    if dist_df is not None:
        if "Cluster" in dist_df.columns and "Borrower_Count" in dist_df.columns:
            cluster_counts = dict(zip(dist_df["Cluster"], dist_df["Borrower_Count"]))
        elif "Cluster" in dist_df.columns and "Count" in dist_df.columns:
            cluster_counts = dict(zip(dist_df["Cluster"], dist_df["Count"]))

    if isinstance(cluster_counts, pd.DataFrame):
        if "Cluster" in cluster_counts.columns and "Borrower_Count" in cluster_counts.columns:
            cluster_counts = dict(zip(cluster_counts["Cluster"], cluster_counts["Borrower_Count"]))
        elif "Cluster" in cluster_counts.columns and "Count" in cluster_counts.columns:
            cluster_counts = dict(zip(cluster_counts["Cluster"], cluster_counts["Count"]))

    if cluster_counts is None:
        cluster_counts = {"Cluster 0": 300, "Cluster 1": 400, "Cluster 2": 300}

    # Normalize cluster keys if numeric
    normalized_counts = {}
    for k, v in cluster_counts.items():
        k_str = f"Cluster {k}" if isinstance(k, int) else str(k)
        normalized_counts[k_str] = int(v)
    cluster_counts = normalized_counts

    # Handle custom costs
    if user_custom_costs is None:
        user_custom_costs = kwargs.get("cluster_action_costs", {})
    if not isinstance(user_custom_costs, dict):
        user_custom_costs = {}

    normalized_costs = {}
    for k, v in user_custom_costs.items():
        k_str = f"Cluster {k}" if isinstance(k, int) else str(k)
        normalized_costs[k_str] = float(v)
    user_custom_costs = normalized_costs

    # Handle uniform cost
    if "c_baseline_uniform" in kwargs:
        uniform_unit_cost = float(kwargs["c_baseline_uniform"])

    clusters = list(cluster_counts.keys())

    # Strategy 1: Uniform Servicing Policy
    uniform_costs = {c: uniform_unit_cost for c in clusters}
    res_uniform = calculate_segment_costs(cluster_counts, uniform_costs)

    # Strategy 2: Segment-Tailored Policy
    if tailored_presets is None:
        tiered_rates = [40.0, 140.0, 320.0, 200.0, 100.0, 250.0]
        tailored_presets = {c: tiered_rates[i % len(tiered_rates)] for i, c in enumerate(clusters)}
    else:
        norm_tailored = {}
        for k, v in tailored_presets.items():
            k_str = f"Cluster {k}" if isinstance(k, int) else str(k)
            norm_tailored[k_str] = float(v)
        tailored_presets = norm_tailored
    res_tailored = calculate_segment_costs(cluster_counts, tailored_presets)

    # Strategy 3: User Custom Strategy
    res_custom = calculate_segment_costs(cluster_counts, user_custom_costs)

    # Comparative Summary
    summary_data = [
        {
            "Strategy": "Strategy A: Uniform Servicing",
            "Description": f"Flat operational allocation (${uniform_unit_cost:,.0f}/borrower) across all clusters",
            "Total_Estimated_Cost ($)": res_uniform["total_cost"],
            "Avg_Cost_Per_Borrower ($)": res_uniform["average_cost_per_borrower"]
        },
        {
            "Strategy": "Strategy B: Segment-Tailored Policy",
            "Description": "Differentiated interventions (automated low-touch to intensive counseling)",
            "Total_Estimated_Cost ($)": res_tailored["total_cost"],
            "Avg_Cost_Per_Borrower ($)": res_tailored["average_cost_per_borrower"]
        },
        {
            "Strategy": "Strategy C: User-Configured Strategy",
            "Description": "Interactive per-cluster cost parameters defined by user",
            "Total_Estimated_Cost ($)": res_custom["total_cost"],
            "Avg_Cost_Per_Borrower ($)": res_custom["average_cost_per_borrower"]
        }
    ]

    summary_df = pd.DataFrame(summary_data)
    min_row = summary_df.loc[summary_df["Total_Estimated_Cost ($)"].idxmin()]
    lowest_strategy = min_row["Strategy"]
    lowest_cost = float(min_row["Total_Estimated_Cost ($)"])

    recommendation = (
        f"Optimal operational strategy is {lowest_strategy} with an estimated portfolio cost "
        f"of ${lowest_cost:,.0f}."
    )

    return {
        "summary_df": summary_df,
        "comparison_df": summary_df,
        "lowest_cost_strategy": lowest_strategy,
        "lowest_estimated_cost": lowest_cost,
        "baseline_total_cost": res_uniform["total_cost"],
        "tailored_total_cost": res_tailored["total_cost"],
        "custom_total_cost": res_custom["total_cost"],
        "recommendation": recommendation,
        "uniform_results": res_uniform,
        "tailored_results": res_tailored,
        "custom_results": res_custom
    }

