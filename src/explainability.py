"""
explainability.py
-----------------
Explainable AI (XAI) engine using SHAP (SHapley Additive exPlanations)
for transparent, compliant, and intuitive credit-risk predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from src.data_processing import (
    get_friendly_feature_name,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

MODELS_DIR = "models"


def load_explainer_artifacts(models_dir: str = MODELS_DIR):
    """
    Loads saved explainers, metadata, and test samples.
    """
    rf_explainer_path = os.path.join(models_dir, "rf_explainer.joblib")
    metadata_path = os.path.join(models_dir, "metadata.json")
    sample_trans_path = os.path.join(models_dir, "test_sample_trans.joblib")

    if not os.path.exists(rf_explainer_path):
        raise FileNotFoundError(
            f"Explainer not found at {rf_explainer_path}. Run 'python -m src.train_model' first."
        )

    rf_explainer = joblib.load(rf_explainer_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    test_sample_trans = joblib.load(sample_trans_path) if os.path.exists(sample_trans_path) else None

    return rf_explainer, metadata, test_sample_trans


def compute_instance_shap(
    X_trans: np.ndarray,
    explainer=None,
    model_name: str = "Random Forest",
    models_dir: str = MODELS_DIR
) -> dict:
    """
    Computes SHAP values for a single prediction instance.
    Guarantees extraction of the Class 1 (Default) contribution.
    """
    if explainer is None:
        explainer, metadata, _ = load_explainer_artifacts(models_dir)
    else:
        with open(os.path.join(models_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

    friendly_names = metadata["friendly_feature_names"]

    # Compute SHAP explanation
    shap_raw = explainer.shap_values(X_trans)

    # Scikit-learn TreeExplainer produces (n_samples, n_features, 2) or list of 2 arrays
    if isinstance(shap_raw, list):
        class1_shap = shap_raw[1][0]
    elif len(shap_raw.shape) == 3:
        class1_shap = shap_raw[0, :, 1]
    else:
        class1_shap = shap_raw[0]

    # Base value (expected value for Class 1)
    if hasattr(explainer, "expected_value"):
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_val = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
        else:
            base_val = float(exp_val)
    else:
        base_val = 0.5

    shap_df = pd.DataFrame({
        "feature": friendly_names,
        "shap_value": class1_shap
    })

    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values(by="abs_shap", ascending=False).reset_index(drop=True)

    # Partition into risk-increasing vs risk-decreasing
    increasing_risk = shap_df[shap_df["shap_value"] > 0].copy()
    decreasing_risk = shap_df[shap_df["shap_value"] < 0].copy()

    return {
        "base_value": base_val,
        "shap_df": shap_df,
        "increasing_risk": increasing_risk,
        "decreasing_risk": decreasing_risk,
        "class1_shap": class1_shap
    }


def create_local_waterfall_plot(shap_info: dict, max_features: int = 10) -> go.Figure:
    """
    Constructs a responsive, modern Plotly waterfall chart showing feature attributions
    moving from the population baseline default rate to the applicant's predicted default rate.
    """
    shap_df = shap_info["shap_df"].head(max_features).copy()
    base_val = shap_info["base_value"]

    # Calculate 'Other features' residual
    other_shap = shap_info["shap_df"].iloc[max_features:]["shap_value"].sum()
    if abs(other_shap) > 0.0001:
        other_row = pd.DataFrame([{
            "feature": f"Other Features ({len(shap_info['shap_df']) - max_features} combined)",
            "shap_value": other_shap,
            "abs_shap": abs(other_shap)
        }])
        plot_df = pd.concat([shap_df, other_row], ignore_index=True)
    else:
        plot_df = shap_df.copy()

    # Create waterfall measures
    names = ["Baseline Population Avg"] + list(plot_df["feature"]) + ["Final Predicted Risk"]
    values = [base_val] + list(plot_df["shap_value"]) + [0]  # last is total
    measures = ["absolute"] + ["relative"] * len(plot_df) + ["total"]

    fig = go.Figure(go.Waterfall(
        name="SHAP Attribution",
        orientation="v",
        measure=measures,
        x=names,
        textposition="outside",
        text=[f"{v:+.1%}" if m == "relative" else f"{v:.1%}" for v, m in zip(values, measures)],
        y=values,
        connector={"line": {"color": "#94A3B8"}},
        decreasing={"marker": {"color": "#10B981"}},  # Green = reduces risk
        increasing={"marker": {"color": "#EF4444"}},  # Red = increases risk
        totals={"marker": {"color": "#0284C7"}}        # Navy / Blue = final
    ))

    fig.update_layout(
        title="<b>Local SHAP Explanation: Feature Contributions to Default Risk</b>",
        title_font={"size": 16, "family": "Inter, sans-serif", "color": "#0F172A"},
        waterfallgap=0.25,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=50, b=80),
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=11, color="#334155"),
            showgrid=False
        ),
        yaxis=dict(
            title="Estimated Probability of Default",
            tickformat=".0%",
            gridcolor="#E2E8F0"
        ),
        height=480
    )

    return fig


def create_local_bar_chart(shap_info: dict, top_n: int = 10) -> go.Figure:
    """
    Creates a divergent horizontal bar chart for top positive and negative SHAP features.
    """
    shap_df = shap_info["shap_df"].head(top_n).sort_values(by="shap_value", ascending=True)

    colors = ["#EF4444" if val > 0 else "#10B981" for val in shap_df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Top {top_n} Factors Influencing Prediction</b>",
        title_font={"size": 15, "family": "Inter, sans-serif"},
        xaxis=dict(
            title="SHAP Value (Impact on Default Probability)",
            zeroline=True,
            zerolinecolor="#64748B",
            zerolinewidth=1.5,
            gridcolor="#F1F5F9"
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#1E293B"),
            automargin=True
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=40, t=50, b=40),
        height=420
    )
    return fig


def create_global_importance_chart(metadata: dict, top_n: int = 15) -> go.Figure:
    """
    Creates the Global Feature Importance bar chart based on mean absolute SHAP values across test data.
    """
    friendly_names = metadata.get("friendly_feature_names", [])
    global_importances = metadata.get("global_importance_rf", [])

    if not friendly_names or not global_importances:
        return go.Figure()

    df_importance = pd.DataFrame({
        "feature": friendly_names,
        "importance": global_importances
    }).sort_values(by="importance", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=df_importance["importance"],
        y=df_importance["feature"],
        orientation='h',
        marker=dict(
            color=df_importance["importance"],
            colorscale='Tealgrn',
            showscale=False
        ),
        text=[f"{val:.3f}" for val in df_importance["importance"]],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Global Feature Importance (Mean |SHAP| across Test Set)</b>",
        title_font={"size": 15, "family": "Inter, sans-serif"},
        xaxis=dict(
            title="Mean Absolute SHAP Value",
            gridcolor="#F1F5F9"
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#1E293B"),
            automargin=True
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=40, t=50, b=40),
        height=480
    )
    return fig


def generate_plain_language_summary(shap_info: dict, raw_inputs: dict) -> str:
    """
    Generates a natural language, human-readable summary explaining the specific
    drivers behind an individual applicant's prediction.
    """
    inc = shap_info["increasing_risk"].head(3)
    dec = shap_info["decreasing_risk"].head(3)

    inc_factors = []
    for _, row in inc.iterrows():
        feat = row["feature"]
        inc_factors.append(f"**{feat}**")

    dec_factors = []
    for _, row in dec.iterrows():
        feat = row["feature"]
        dec_factors.append(f"**{feat}**")

    inc_str = ", ".join(inc_factors) if inc_factors else "None"
    dec_str = ", ".join(dec_factors) if dec_factors else "None"

    summary = (
        f"**Explanation Summary:**\n\n"
        f"• **Risk-Increasing Drivers:** The top factors pushing the predicted default probability **higher** "
        f"are {inc_str}. These indicators signaled elevated credit risk compared to historical non-defaulting borrowers.\n\n"
        f"• **Risk-Decreasing Drivers:** Conversely, factors exerting the strongest downward pressure on default risk "
        f"include {dec_str}. These protective borrower traits helped moderate the final default estimate.\n\n"
        f"*Note: SHAP values quantify statistical contribution to the machine learning model's output; "
        f"they indicate mathematical attribution, not real-world causation.*"
    )
    return summary
