"""
cost_support_page.py
--------------------
Page 5: Cost-Sensitive Decision Support for Unsupervised Borrower Segmentation.
Adapts cost-sensitive optimization to portfolio operational servicing strategies.
Compares alternative segment-tailored action policies against uniform baselines.
Does NOT use default probabilities or classification thresholds.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.clustering import run_kmeans_segmentation
from src.cost_support import compare_operational_strategies, DEFAULT_ACTION_PRESETS
from src.ui_pages.styles import render_disclaimer, render_metric_card

DATA_PATH = os.path.join("data", "loan_train.csv")


def render_cost_support_page():
    st.markdown('<div class="section-header">💰 Cost-Sensitive Decision Support</div>', unsafe_allow_html=True)
    st.markdown("""
    Optimize portfolio management expenditures by tailoring operational actions to borrower segments. 
    Instead of applying an expensive, one-size-fits-all monitoring process across the entire portfolio, 
    credit managers can assign **differentiated operational interventions** 
    (e.g., automated low-touch servicing for low-debt borrowers vs. enhanced counseling for high-leverage borrowers) 
    to evaluate portfolio operational trade-offs.
    """)

    # Ensure clustering results are active
    if "clustering_results" not in st.session_state:
        target_df = st.session_state.get("uploaded_df")
        if target_df is None:
            if not os.path.exists(DATA_PATH):
                st.error("Baseline dataset not found. Please upload a dataset.")
                return
            target_df = pd.read_csv(DATA_PATH, low_memory=False)

        with st.spinner("Generating borrower segmentation profile..."):
            st.session_state["clustering_results"] = run_kmeans_segmentation(target_df, n_clusters=3)

    results = st.session_state["clustering_results"]
    dist_df = results["distribution_df"]
    cluster_counts = dict(zip(dist_df["Cluster"], dist_df["Borrower_Count"]))
    n_clusters = results["n_clusters"]
    semantic_labels = results.get("semantic_labels", {})

    st.markdown("### 1. Configure Illustrative Action Costs per Segment")
    st.markdown("""
    Define the estimated operational cost ($ per borrower) for administering the servicing strategy 
    assigned to each borrower segment:
    """)

    user_costs = {}
    cols = st.columns(min(n_clusters, 4))
    default_cost_suggestions = [35.0, 150.0, 320.0, 200.0, 100.0, 250.0]

    for i, c_name in enumerate(cluster_counts.keys()):
        col_idx = i % len(cols)
        seg_desc = semantic_labels.get(c_name, c_name)
        with cols[col_idx]:
            user_costs[c_name] = st.number_input(
                f"{c_name} Action Cost ($):",
                min_value=5.0,
                max_value=1500.0,
                value=float(default_cost_suggestions[i % len(default_cost_suggestions)]),
                step=10.0,
                help=f"Operational intervention cost per borrower in {seg_desc}."
            )
            st.caption(f"**Profile:** {seg_desc.split(': ')[-1] if ': ' in seg_desc else seg_desc}")

    with st.expander("⚙️ Configure Baseline Uniform Servicing Cost"):
        uniform_cost = st.slider(
            "Baseline Uniform Policy Cost ($ per borrower):",
            min_value=20.0,
            max_value=500.0,
            value=120.0,
            step=10.0,
            help="Cost per borrower if the same standard review process is applied uniformly across all borrowers."
        )

    # Compute comparative strategy analysis
    comp_res = compare_operational_strategies(
        cluster_counts=cluster_counts,
        user_custom_costs=user_costs,
        uniform_unit_cost=float(uniform_cost)
    )

    summary_df = comp_res["summary_df"]
    lowest_strat = comp_res["lowest_cost_strategy"]
    lowest_cost = comp_res["lowest_estimated_cost"]

    # Uniform Baseline Cost for comparison
    uniform_total = float(summary_df.loc[summary_df["Strategy"].str.contains("Uniform"), "Total_Estimated_Cost ($)"].values[0])
    cost_diff = uniform_total - lowest_cost
    pct_savings = (cost_diff / uniform_total * 100.0) if uniform_total > 0 else 0.0

    st.markdown("---")
    st.markdown("### 2. Strategy Cost Comparison & Optimization")

    # KPI Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card(
            "Lowest Cost Strategy",
            lowest_strat.split(":")[0],
            lowest_strat.split(": ")[-1],
            "#10B981"
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card(
            "Optimized Portfolio Cost",
            f"${lowest_cost:,.0f}",
            "Under lowest strategy",
            "#0EA5E9"
        ), unsafe_allow_html=True)
    with m3:
        st.markdown(render_metric_card(
            "Cost Variance vs. Uniform",
            f"${cost_diff:,.0f}",
            f"{pct_savings:+.1f}% operational difference",
            "#F59E0B"
        ), unsafe_allow_html=True)
    with m4:
        avg_cost = comp_res["custom_results"]["average_cost_per_borrower"]
        st.markdown(render_metric_card(
            "Custom Avg Cost / Borrower",
            f"${avg_cost:.2f}",
            f"Across {comp_res['custom_results']['total_borrowers']:,} accounts",
            "#8B5CF6"
        ), unsafe_allow_html=True)

    # Strategy Comparison Bar Chart
    col_chart, col_table = st.columns([1.3, 1.0])

    with col_chart:
        st.markdown("##### Total Estimated Portfolio Cost by Strategy")
        fig_bar = px.bar(
            summary_df,
            x="Strategy",
            y="Total_Estimated_Cost ($)",
            text=summary_df["Total_Estimated_Cost ($)"].apply(lambda v: f"${v:,.0f}"),
            color="Strategy",
            color_discrete_sequence=["#64748B", "#0F766E", "#0284C7"]
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            yaxis=dict(title="Total Estimated Cost ($)", gridcolor="#F1F5F9"),
            height=340,
            margin=dict(t=20, b=30, l=60, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        st.markdown("##### Comparative Strategy Summary")
        disp_summary = summary_df.copy()
        disp_summary["Total_Estimated_Cost ($)"] = disp_summary["Total_Estimated_Cost ($)"].apply(lambda x: f"${x:,.0f}")
        disp_summary["Avg_Cost_Per_Borrower ($)"] = disp_summary["Avg_Cost_Per_Borrower ($)"].apply(lambda x: f"${x:.2f}")
        st.dataframe(disp_summary[["Strategy", "Total_Estimated_Cost ($)", "Avg_Cost_Per_Borrower ($)"]].set_index("Strategy"), use_container_width=True)

    # Per-Segment Cost Breakdown for Custom Strategy
    st.markdown("### 3. Detailed Segment-Level Expenditure (Custom Strategy)")
    st.markdown("Breakdown of total operational spending across discovered borrower segments under your configured assumptions:")

    custom_details_df = comp_res["custom_results"]["details_df"].copy()
    custom_details_df["Descriptive_Segment"] = custom_details_df["Cluster"].map(semantic_labels)
    custom_details_df["Total_Segment_Cost ($)"] = custom_details_df["Total_Segment_Cost ($)"].apply(lambda x: f"${x:,.0f}")
    custom_details_df["Cost_per_Borrower ($)"] = custom_details_df["Cost_per_Borrower ($)"].apply(lambda x: f"${x:.2f}")
    custom_details_df["Borrower_Count"] = custom_details_df["Borrower_Count"].apply(lambda x: f"{x:,}")

    cols_order = ["Cluster", "Descriptive_Segment", "Borrower_Count", "Cost_per_Borrower ($)", "Total_Segment_Cost ($)", "Cost_Share (%)"]
    st.dataframe(custom_details_df[cols_order].set_index("Cluster"), use_container_width=True)

    st.markdown(render_disclaimer(
        "Important Operational Cost Assumptions: All action costs and servicing budgets are illustrative parameters "
        "defined by the user to evaluate portfolio resource allocation across K-Means segments. "
        "These figures are not validated accounting costs and do not predict credit loss rates or loan defaults."
    ), unsafe_allow_html=True)
