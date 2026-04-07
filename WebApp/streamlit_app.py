from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecast_engine import default_scenario_template, executive_summary, run_all_forecasts, valid_banks


st.set_page_config(page_title="UC2 Forecast Dashboard", layout="wide")


def _feature_table(feature_df: pd.DataFrame, quarter: str) -> pd.DataFrame:
    drop_cols = [column for column in ["Bank", "Quarter"] if column in feature_df.columns]
    features = feature_df.loc[feature_df["Quarter"] == quarter].drop(columns=drop_cols)
    table = features.T.reset_index()
    table.columns = ["Driver", "Value"]
    return table


def _forecast_plot(result_df: pd.DataFrame, show_bloomberg: bool = False):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x_labels = result_df["Quarter"].tolist()

    if show_bloomberg and "Bloomberg Consensus" in result_df.columns:
        ax.plot(x_labels, result_df["Bloomberg Consensus"], marker="o", linewidth=2, label="Bloomberg Consensus")
        ax.plot(x_labels, result_df["ML Base"], marker="o", linewidth=2.4, label="Model Base")
    else:
        ax.plot(x_labels, result_df["ML Base"], marker="o", linewidth=2.4, label="Base Forecast")
        if "TS Global Champion" in result_df.columns:
            ax.plot(
                x_labels,
                result_df["TS Global Champion"],
                marker="o",
                linewidth=1.8,
                linestyle="--",
                label="TS Global Champion",
            )

    ax.plot(x_labels, result_df["Bear"], linewidth=1.8, linestyle="--", label="Bear")
    ax.plot(x_labels, result_df["Bull"], linewidth=1.8, linestyle="--", label="Bull")
    if "95% CI Lower" in result_df.columns and "95% CI Upper" in result_df.columns:
        ax.fill_between(
            x_labels,
            result_df["95% CI Lower"].to_numpy(dtype=float),
            result_df["95% CI Upper"].to_numpy(dtype=float),
            alpha=0.15,
            label="95% CI",
        )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("PCL")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def _render_individual_bundle(bundle: dict, key_prefix: str, title: str) -> None:
    st.subheader(title)

    if bundle["skipped_banks"]:
        skipped_text = ", ".join(f"{bank}: {reason}" for bank, reason in bundle["skipped_banks"].items())
        st.caption(f"Skipped banks: {skipped_text}")

    if not bundle["banks"]:
        st.warning("No eligible banks were available for this model variant.")
        return

    latest_df = bundle["latest_quarter_summary_df"]
    if not latest_df.empty:
        st.markdown("**Latest Quarter Bank Comparison**")
        chart_df = latest_df.set_index("Bank")[["ML Base", "Bear", "Bull"]]
        st.bar_chart(chart_df)
        st.dataframe(latest_df, use_container_width=True, hide_index=True)

    bank = st.selectbox("Bank", bundle["banks"], key=f"{key_prefix}_bank")
    bank_bundle = bundle["per_bank"][bank]
    quarter = st.selectbox("Quarter", bank_bundle["result_df"]["Quarter"].tolist(), key=f"{key_prefix}_quarter")
    row = bank_bundle["result_df"].loc[bank_bundle["result_df"]["Quarter"] == quarter].iloc[0]

    st.caption(
        f"Winning model: {bank_bundle['metadata']['winner_model']} | "
        f"Best OOS RMSE: {bank_bundle['metadata']['best_rmse']:,.2f}"
    )
    metrics = st.columns(5)
    metrics[0].metric("Base", f"{row['ML Base']:,.1f}")
    metrics[1].metric("Bear", f"{row['Bear']:,.1f}")
    metrics[2].metric("Bull", f"{row['Bull']:,.1f}")
    metrics[3].metric("95% CI Low", f"{row['95% CI Lower']:,.1f}")
    metrics[4].metric("95% CI High", f"{row['95% CI Upper']:,.1f}")

    left, right = st.columns([1.35, 0.9])
    with left:
        chart_df = bank_bundle["result_df"].set_index("Quarter")[["ML Base", "Bear", "Bull", "95% CI Lower", "95% CI Upper"]]
        st.line_chart(chart_df)
        st.dataframe(bank_bundle["result_df"], use_container_width=True, hide_index=True)
        with st.expander("Bank model scorecard", expanded=False):
            st.dataframe(bank_bundle["scorecard_df"], use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Forecast Drivers**")
        st.dataframe(_feature_table(bank_bundle["feature_df"], quarter), use_container_width=True, hide_index=True)


def _render_customer_model(bundle: dict, key_prefix: str, title: str, show_bloomberg: bool = False) -> None:
    result_df = bundle["result_df"]
    quarter = st.selectbox("Quarter", result_df["Quarter"].tolist(), key=f"{key_prefix}_quarter")
    row = result_df.loc[result_df["Quarter"] == quarter].iloc[0]

    st.subheader(title)
    st.caption(
        f"ML Champion: {bundle['metadata']['ml_champion']}"
        + (
            f" | TS Champion: {bundle['metadata']['ts_champion']}"
            if bundle["metadata"].get("ts_champion")
            else ""
        )
    )

    if show_bloomberg:
        metrics = st.columns(6)
        metrics[0].metric("Bloomberg Base", f"{row['Bloomberg Consensus']:,.1f}")
        metrics[1].metric("ML Delta", f"{row['ML Residual Delta']:,.1f}")
        metrics[2].metric("Base", f"{row['ML Base']:,.1f}")
        metrics[3].metric("Bear", f"{row['Bear']:,.1f}")
        metrics[4].metric("Bull", f"{row['Bull']:,.1f}")
        metrics[5].metric("95% Range", f"{row['95% CI Lower']:,.1f} to {row['95% CI Upper']:,.1f}")
        chart_df = result_df.set_index("Quarter")[
            ["Bloomberg Consensus", "ML Base", "Bear", "Bull", "95% CI Lower", "95% CI Upper"]
        ]
    else:
        metrics = st.columns(5)
        metrics[0].metric("Base", f"{row['ML Base']:,.1f}")
        metrics[1].metric("Bear", f"{row['Bear']:,.1f}")
        metrics[2].metric("Bull", f"{row['Bull']:,.1f}")
        metrics[3].metric("95% CI Low", f"{row['95% CI Lower']:,.1f}")
        metrics[4].metric("95% CI High", f"{row['95% CI Upper']:,.1f}")
        if "TS Global Champion" in row and pd.notna(row["TS Global Champion"]):
            st.metric("TS Global Champion", f"{row['TS Global Champion']:,.1f}")
        chart_df = result_df.set_index("Quarter")[[column for column in result_df.columns if column != "Quarter"]]

    left, right = st.columns([1.35, 0.9])
    with left:
        st.markdown("**Forecast Chart**")
        st.pyplot(_forecast_plot(result_df, show_bloomberg=show_bloomberg), use_container_width=True)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Forecast Drivers**")
        st.dataframe(_feature_table(bundle["feature_df"], quarter), use_container_width=True, hide_index=True)


st.title("UC2 Forecast Dashboard")
st.write(
    "Edit the scenario assumptions directly in the app and run the forecast report. "
    "Results are shown immediately in the dashboard."
)

if "scenario_df" not in st.session_state:
    st.session_state["scenario_df"] = default_scenario_template()

scenario_df = st.session_state["scenario_df"].copy()

with st.expander("Edit Scenario Inputs", expanded=True):
    scenario_df = st.data_editor(
        scenario_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Bank": st.column_config.SelectboxColumn("Bank", options=list(valid_banks())),
            "Year": st.column_config.NumberColumn("Year", step=1, format="%d"),
            "Quarter": st.column_config.SelectboxColumn("Quarter", options=["Q1", "Q2", "Q3", "Q4"]),
            "Bloomberg": st.column_config.NumberColumn("Bloomberg", format="%.2f"),
            "GDP YoY Forecast": st.column_config.NumberColumn("GDP YoY Forecast", format="%.2f"),
            "Unemployment Rate YoY QUARTER": st.column_config.NumberColumn(
                "Unemployment Rate YoY QUARTER", format="%.2f"
            ),
            "Overnight Rate": st.column_config.NumberColumn("Overnight Rate", format="%.2f"),
        },
        key="scenario_editor",
    )

st.session_state["scenario_df"] = scenario_df.copy()

if st.button("Run Forecast Report", type="primary"):
    try:
        with st.spinner("Running forecast models and loading cached assets..."):
            results = run_all_forecasts(scenario_df)
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        st.stop()
    st.session_state["forecast_results"] = results

results = st.session_state.get("forecast_results")
if results is None:
    st.info("Update the scenario table above and click Run Forecast Report.")
    st.stop()

st.success("Forecast report completed.")

st.subheader("Executive Summary")
for line in executive_summary(results):
    st.write(f"- {line}")

overview_tab, agg1_tab, agg2_tab, bbg_tab, ind_ns_tab, ind_sent_tab = st.tabs(
    [
        "Overview",
        "Aggregate Model 1",
        "Aggregate Model 2",
        "Bloomberg Model",
        "Individual Model (No Sentiment)",
        "Individual Model (With Sentiment)",
    ]
)

with overview_tab:
    latest_q = results["aggregate_model_1"]["result_df"]["Quarter"].iloc[-1]
    st.markdown(f"**Latest quarter in scenario:** {latest_q}")

    overview_cols = st.columns(3)
    for column, key, label in zip(
        overview_cols,
        ["aggregate_model_1", "aggregate_model_2", "bloomberg_model"],
        ["Aggregate Model 1", "Aggregate Model 2", "Bloomberg Model"],
    ):
        row = results[key]["result_df"].loc[results[key]["result_df"]["Quarter"] == latest_q].iloc[0]
        with column:
            st.markdown(f"**{label}**")
            st.metric("Base", f"{row['ML Base']:,.1f}")
            st.metric("Bear", f"{row['Bear']:,.1f}")
            st.metric("Bull", f"{row['Bull']:,.1f}")

    combined = pd.DataFrame(
        {
            "Quarter": results["aggregate_model_1"]["result_df"]["Quarter"],
            "Aggregate Model 1": results["aggregate_model_1"]["result_df"]["ML Base"],
            "Aggregate Model 2": results["aggregate_model_2"]["result_df"]["ML Base"],
            "Bloomberg Model": results["bloomberg_model"]["result_df"]["ML Base"],
        }
    ).set_index("Quarter")
    st.line_chart(combined)

    st.markdown("**Individual Bank Snapshot**")
    ind_left, ind_right = st.columns(2)
    with ind_left:
        st.markdown("No Sentiment")
        if not results["individual_no_sentiment"]["latest_quarter_summary_df"].empty:
            st.dataframe(
                results["individual_no_sentiment"]["latest_quarter_summary_df"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No eligible banks.")
    with ind_right:
        st.markdown("With Sentiment")
        if not results["individual_with_sentiment"]["latest_quarter_summary_df"].empty:
            st.dataframe(
                results["individual_with_sentiment"]["latest_quarter_summary_df"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No eligible banks.")

with agg1_tab:
    _render_customer_model(results["aggregate_model_1"], "agg_m1", "Aggregate Model 1")
with agg2_tab:
    _render_customer_model(results["aggregate_model_2"], "agg_m2", "Aggregate Model 2")
with bbg_tab:
    _render_customer_model(results["bloomberg_model"], "bbg", "Bloomberg Residual Model", show_bloomberg=True)
with ind_ns_tab:
    _render_individual_bundle(results["individual_no_sentiment"], "ind_ns", "Individual Model (No Sentiment)")
with ind_sent_tab:
    _render_individual_bundle(results["individual_with_sentiment"], "ind_sent", "Individual Model (With Sentiment)")
