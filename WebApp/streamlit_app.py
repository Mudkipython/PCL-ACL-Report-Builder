from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from forecast_engine import default_quarter_template, parse_uploaded_scenario, run_all_forecasts


st.set_page_config(page_title="UC2 Forecast Dashboard", layout="wide")


def _load_input_table() -> pd.DataFrame:
    uploaded = st.sidebar.file_uploader("上传场景文件（CSV/XLSX）", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        return parse_uploaded_scenario(uploaded.name, uploaded.getvalue())
    return default_quarter_template()


def _render_bundle(bundle: dict, key_prefix: str) -> None:
    result_df = bundle["result_df"]
    feature_df = bundle["feature_df"]
    quarter = st.selectbox("选择季度", result_df["Quarter"].tolist(), key=f"{key_prefix}_quarter")
    row = result_df.loc[result_df["Quarter"] == quarter].iloc[0]
    features = feature_df.loc[feature_df["Quarter"] == quarter].drop(columns=["Quarter"])
    feature_table = features.T.reset_index()
    feature_table.columns = ["Feature", "Value"]

    st.caption(
        f"ML Champion: {bundle['metadata']['ml_champion']}"
        + (
            f" | TS Champion: {bundle['metadata']['ts_champion']}"
            if bundle["metadata"].get("ts_champion")
            else ""
        )
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Base", f"{row['ML Base']:,.1f}")
    metric_columns[1].metric("Bear", f"{row['Bear']:,.1f}")
    metric_columns[2].metric("Bull", f"{row['Bull']:,.1f}")
    metric_columns[3].metric("95% CI Low", f"{row['95% CI Lower']:,.1f}")
    metric_columns[4].metric("95% CI High", f"{row['95% CI Upper']:,.1f}")

    if "TS Global Champion" in row and pd.notna(row["TS Global Champion"]):
        st.metric("TS Global Champion", f"{row['TS Global Champion']:,.1f}")

    left, right = st.columns([1.2, 1])
    with left:
        chart_df = result_df.set_index("Quarter")[[column for column in result_df.columns if column != "Quarter"]]
        st.line_chart(chart_df)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Features")
        st.dataframe(feature_table, use_container_width=True, hide_index=True)
        st.subheader("JSON")
        st.json(bundle["payload"][quarter])


def _render_bloomberg_bundle(bundle: dict) -> None:
    result_df = bundle["result_df"]
    feature_df = bundle["feature_df"]
    quarter = st.selectbox("选择季度", result_df["Quarter"].tolist(), key="bloomberg_quarter")
    row = result_df.loc[result_df["Quarter"] == quarter].iloc[0]
    features = feature_df.loc[feature_df["Quarter"] == quarter].drop(columns=["Quarter"])
    feature_table = features.T.reset_index()
    feature_table.columns = ["Feature", "Value"]

    st.caption(f"ML Champion: {bundle['metadata']['ml_champion']}")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Bloomberg", f"{row['Bloomberg Consensus']:,.1f}")
    metric_columns[1].metric("ML Delta", f"{row['ML Residual Delta']:,.1f}")
    metric_columns[2].metric("Base", f"{row['ML Base']:,.1f}")
    metric_columns[3].metric("Bear", f"{row['Bear']:,.1f}")
    metric_columns[4].metric("Bull", f"{row['Bull']:,.1f}")
    metric_columns[5].metric("95% CI", f"{row['95% CI Lower']:,.1f} ~ {row['95% CI Upper']:,.1f}")

    left, right = st.columns([1.2, 1])
    with left:
        chart_df = result_df.set_index("Quarter")[
            ["Bloomberg Consensus", "ML Base", "Bear", "Bull", "95% CI Lower", "95% CI Upper"]
        ]
        st.line_chart(chart_df)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Features")
        st.dataframe(feature_table, use_container_width=True, hide_index=True)
        st.subheader("JSON")
        st.json(bundle["payload"][quarter])


st.title("UC2 宏观场景预测面板")
st.write(
    "这个页面把两个 `ipynb` 的训练逻辑封装成一个 Streamlit 前端。"
    "客户可以直接修改未来季度的宏观假设，系统会返回 Aggregate Model 1、Aggregate Model 2 和 Bloomberg Residual Model 的预测。"
)
st.info("首次运行会加载并缓存 notebook 训练单元，通常需要几秒。后续交互会明显更快。")

input_df = _load_input_table()

edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
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

run_button = st.button("运行预测", type="primary")

if run_button:
    try:
        results = run_all_forecasts(edited_df)
    except Exception as exc:
        st.error(f"预测失败：{exc}")
    else:
        st.success("预测完成。")

        st.download_button(
            "下载 JSON",
            data=json.dumps(results["json_payload"], ensure_ascii=False, indent=2),
            file_name="uc2_forecast_payload.json",
            mime="application/json",
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Aggregate Model 1", "Aggregate Model 2", "Bloomberg Model", "Full JSON"]
        )

        with tab1:
            _render_bundle(results["aggregate_model_1"], "agg_m1")
        with tab2:
            _render_bundle(results["aggregate_model_2"], "agg_m2")
        with tab3:
            _render_bloomberg_bundle(results["bloomberg_model"])
        with tab4:
            st.json(results["json_payload"])
