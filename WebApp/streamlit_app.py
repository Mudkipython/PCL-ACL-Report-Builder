from __future__ import annotations

import hashlib
import io
import json

import pandas as pd
import streamlit as st

from forecast_engine import (
    default_scenario_template,
    executive_summary,
    parse_uploaded_scenario,
    results_excel_bytes,
    run_all_forecasts,
    template_csv_bytes,
    template_excel_bytes,
    valid_banks,
)


st.set_page_config(page_title="UC2 Forecast Dashboard", layout="wide")


def _feature_table(feature_df: pd.DataFrame, quarter: str) -> pd.DataFrame:
    drop_cols = [column for column in ["Bank", "Quarter"] if column in feature_df.columns]
    features = feature_df.loc[feature_df["Quarter"] == quarter].drop(columns=drop_cols)
    table = features.T.reset_index()
    table.columns = ["Driver", "Value"]
    return table


def _scenario_excel_bytes(scenario_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        scenario_df.to_excel(writer, index=False, sheet_name="Scenario Template")
    return output.getvalue()


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
        st.line_chart(chart_df)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Forecast Drivers**")
        st.dataframe(_feature_table(bundle["feature_df"], quarter), use_container_width=True, hide_index=True)


st.title("UC2 Forecast Dashboard")
st.write(
    "A customer-facing forecasting app built from the notebook workflow. "
    "Start from the official default scenario, edit values directly in the app or in Excel, "
    "and then run the forecast report."
)

hero_left, hero_right = st.columns([1.2, 1])
with hero_left:
    st.markdown("**Recommended workflow**")
    st.markdown(
        "1. Download the official scenario template.\n"
        "2. Update macro assumptions in the app or in Excel.\n"
        "3. Optionally upload the official workbook or exported scenario template.\n"
        "4. Run the forecast report."
    )
with hero_right:
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "CSV Template",
        data=template_csv_bytes(),
        file_name="uc2_scenario_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl2.download_button(
        "Excel Template",
        data=template_excel_bytes(),
        file_name="uc2_scenario_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption(
        "Expected columns: Bank, Year, Quarter, Bloomberg, GDP YoY Forecast, "
        "Unemployment Rate YoY QUARTER, Overnight Rate"
    )

if "scenario_editor_version" not in st.session_state:
    st.session_state["scenario_editor_version"] = 0
if "working_scenario_df" not in st.session_state:
    st.session_state["working_scenario_df"] = default_scenario_template()
    st.session_state["scenario_source"] = "Official default scenario"

def _load_working_scenario(dataframe: pd.DataFrame, source_label: str) -> None:
    st.session_state["working_scenario_df"] = dataframe.copy()
    st.session_state["scenario_source"] = source_label
    st.session_state["scenario_editor_version"] += 1
    st.session_state.pop("forecast_results", None)


uploaded = st.file_uploader(
    "Optional: upload the official scenario template or the full workbook",
    type=["csv", "xlsx", "xls"],
)

action_left, action_right = st.columns([1, 1.2])
with action_left:
    if st.button("Reset to Official Default Scenario", use_container_width=True):
        _load_working_scenario(default_scenario_template(), "Official default scenario")
with action_right:
    st.download_button(
        "Download Current Scenario as Excel",
        data=_scenario_excel_bytes(st.session_state["working_scenario_df"]),
        file_name="uc2_current_scenario.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

if uploaded is not None:
    uploaded_bytes = uploaded.getvalue()
    current_signature = (
        uploaded.name,
        uploaded.size,
        hashlib.blake2b(uploaded_bytes, digest_size=16).hexdigest(),
    )
    if st.session_state.get("uploaded_signature") != current_signature:
        st.session_state["uploaded_signature"] = current_signature
        try:
            loaded_df = parse_uploaded_scenario(uploaded.name, uploaded_bytes)
        except Exception as exc:
            st.error(f"Upload could not be parsed, keeping the current scenario instead: {exc}")
        else:
            _load_working_scenario(loaded_df, f"Uploaded file: {uploaded.name}")
            st.success(f"Loaded scenario from {uploaded.name}")

scenario_df = st.session_state["working_scenario_df"].copy()
st.caption(
    f"Scenario source: {st.session_state.get('scenario_source', 'Official default scenario')}. "
    "You can keep historical actuals as-is and only update forward-looking forecast rows."
)

with st.expander("Scenario preview", expanded=False):
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

with st.expander("Edit Scenario in App", expanded=True):
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
        key=f"scenario_editor_{st.session_state['scenario_editor_version']}",
    )
st.session_state["working_scenario_df"] = scenario_df.copy()

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
    st.info("Edit the default scenario or upload a workbook/template, then click Run Forecast Report.")
    st.stop()

st.success("Forecast report completed.")

report_col1, report_col2 = st.columns(2)
report_col1.download_button(
    "Download Excel Report",
    data=results_excel_bytes(results),
    file_name="uc2_forecast_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
report_col2.download_button(
    "Download Technical JSON",
    data=json.dumps(results["json_payload"], ensure_ascii=False, indent=2),
    file_name="uc2_forecast_payload.json",
    mime="application/json",
    use_container_width=True,
)

st.subheader("Executive Summary")
for line in executive_summary(results):
    st.write(f"- {line}")

overview_tab, agg1_tab, agg2_tab, bbg_tab, ind_ns_tab, ind_sent_tab, tech_tab = st.tabs(
    [
        "Overview",
        "Aggregate Model 1",
        "Aggregate Model 2",
        "Bloomberg Model",
        "Individual Model (No Sentiment)",
        "Individual Model (With Sentiment)",
        "Technical",
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
with tech_tab:
    st.markdown("**Scenario Input**")
    st.dataframe(results["scenario_df"], use_container_width=True, hide_index=True)
    with st.expander("Technical JSON", expanded=False):
        st.json(results["json_payload"])
