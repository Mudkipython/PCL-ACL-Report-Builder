from __future__ import annotations

import contextlib
import io
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX

try:
    from pmdarima import auto_arima

    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False


BASE_DIR = Path(__file__).resolve().parent
WORKBOOK_PATH = BASE_DIR / "Complete_UC2_Forecast_Data - Use Case 2 - Modeller.xlsx"
AGGREGATE_NOTEBOOK = BASE_DIR / "Aggregate - Use Case 2 - Modeller.ipynb"
BLOOMBERG_NOTEBOOK = BASE_DIR / "Bloomberg - Use Case 2 - Modeller.ipynb"
QUARTER_TO_DATE = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
DEFAULT_PLAYBACK_CURVE = np.array([0.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0], dtype=float)


def _patched_notebook_code(code: str) -> str:
    return (
        code.replace("except ImportError:", "except Exception:")
        .replace("Complete_UC2_Forecast_Data.xlsx", str(WORKBOOK_PATH))
    )


def _execute_notebook_cells(notebook_path: Path, cell_indices: list[int]) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    namespace: dict[str, Any] = {"display": lambda *args, **kwargs: None}
    notebook = nbformat.read(notebook_path, as_version=4)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for index in cell_indices:
            code = _patched_notebook_code(notebook.cells[index].source)
            exec(compile(code, f"{notebook_path.name}#cell{index}", "exec"), namespace)

    return namespace


def _quarter_strings(index: pd.DatetimeIndex) -> list[str]:
    return [f"{date.year} Q{date.quarter}" for date in index]


def _quarter_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        df["Year"].astype(int).astype(str)
        + "-"
        + df["Quarter"].map(QUARTER_TO_DATE)
    )


def _clean_scenario_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Year",
        "Quarter",
        "Bloomberg",
        "GDP YoY Forecast",
        "Unemployment Rate YoY QUARTER",
        "Overnight Rate",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Scenario data is missing columns: {', '.join(missing)}")

    scenario_df = df[required].copy()
    scenario_df["Year"] = pd.to_numeric(scenario_df["Year"], errors="coerce").astype("Int64")
    scenario_df["Quarter"] = scenario_df["Quarter"].astype(str).str.upper().str.strip()

    for column in required[2:]:
        scenario_df[column] = pd.to_numeric(scenario_df[column], errors="coerce")

    scenario_df = scenario_df.dropna().copy()
    scenario_df["Year"] = scenario_df["Year"].astype(int)

    allowed_quarters = set(QUARTER_TO_DATE)
    invalid_quarters = sorted(set(scenario_df["Quarter"]) - allowed_quarters)
    if invalid_quarters:
        raise ValueError(f"Unsupported quarter labels: {', '.join(invalid_quarters)}")

    return scenario_df.sort_values(["Year", "Quarter"]).reset_index(drop=True)


def _load_sheet1() -> pd.DataFrame:
    return pd.read_excel(WORKBOOK_PATH, sheet_name="Sheet1")


def default_quarter_template() -> pd.DataFrame:
    sheet1 = _load_sheet1()
    template = (
        sheet1.groupby(["Year", "Quarter"], as_index=False)
        .agg(
            {
                "Bloomberg": "sum",
                "GDP YoY Forecast": "mean",
                "Unemployment Rate YoY QUARTER": "mean",
                "Overnight Rate": "mean",
            }
        )
        .sort_values(["Year", "Quarter"])
        .reset_index(drop=True)
    )
    return template


def parse_uploaded_scenario(upload_name: str, data: bytes) -> pd.DataFrame:
    suffix = Path(upload_name).suffix.lower()
    buffer = io.BytesIO(data)

    if suffix == ".csv":
        uploaded = pd.read_csv(buffer)
    elif suffix in {".xlsx", ".xls"}:
        uploaded = pd.read_excel(buffer)
    else:
        raise ValueError("Only CSV and Excel files are supported.")

    return _clean_scenario_df(uploaded)


def _forecast_sentiment(history: pd.Series, steps: int) -> np.ndarray:
    clean_history = history.dropna().astype(float).to_numpy()
    if steps <= 0:
        return np.array([], dtype=float)
    if len(clean_history) == 0:
        return np.zeros(steps, dtype=float)

    try:
        if HAS_PMDARIMA:
            return np.asarray(
                auto_arima(clean_history, seasonal=False, suppress_warnings=True).predict(n_periods=steps),
                dtype=float,
            )
        return np.asarray(ARIMA(clean_history, order=(1, 0, 0)).fit().forecast(steps=steps), dtype=float)
    except Exception:
        return np.repeat(clean_history[-1], steps).astype(float)


def _playback_metrics(model_df: pd.DataFrame, horizon: int) -> tuple[float, float, np.ndarray]:
    pre_covid = model_df.loc["2019-12-31", "PCL_t"] if "2019-12-31" in model_df.index else 1500.0
    covid_peak = model_df.loc["2020-09-30", "PCL_t"] if "2020-09-30" in model_df.index else 6000.0
    pre_recovery = model_df.loc["2021-03-31", "PCL_t"] if "2021-03-31" in model_df.index else 1500.0
    recovery_trough = model_df.loc["2022-06-30", "PCL_t"] if "2022-06-30" in model_df.index else 0.0

    covid_shock = max(0.0, float(covid_peak - pre_covid))
    recovery_release = max(0.0, float(pre_recovery - recovery_trough))
    playback_curve = DEFAULT_PLAYBACK_CURVE[:horizon]
    return covid_shock, recovery_release, playback_curve


def _build_bloomberg_model(model_name: str):
    if "Elastic Net" in model_name or "Lasso" in model_name:
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=TimeSeriesSplit(n_splits=3))
    if "Ridge" in model_name:
        return RidgeCV(alphas=np.logspace(-2, 4, 50), cv=TimeSeriesSplit(n_splits=3))
    if "Random Forest" in model_name:
        return RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    return GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
    )


@lru_cache(maxsize=1)
def load_aggregate_context() -> dict[str, Any]:
    return _execute_notebook_cells(AGGREGATE_NOTEBOOK, [4, 5, 9, 10])


@lru_cache(maxsize=1)
def load_bloomberg_context() -> dict[str, Any]:
    return _execute_notebook_cells(BLOOMBERG_NOTEBOOK, [5, 6])


def _forecast_aggregate_model(
    scenario_df: pd.DataFrame,
    model_number: int,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    grouped = (
        scenario_df.groupby(["Year", "Quarter"], as_index=False)
        .agg(
            {
                "GDP YoY Forecast": "mean",
                "Unemployment Rate YoY QUARTER": "mean",
                "Overnight Rate": "mean",
            }
        )
        .sort_values(["Year", "Quarter"])
        .reset_index(drop=True)
    )
    forecast_index = _quarter_index(grouped)
    quarter_labels = _quarter_strings(forecast_index)
    horizon = len(grouped)

    if model_number == 1:
        model_df = ctx["model1_df"]
        train_valid_df = ctx["train_valid_df1"]
        selected_features = ctx["selected_features_m1"]
        best_ml_name = ctx["best_ml_m1"]
        global_best_name = ctx["global_best_m1"]
        final_table = ctx["final_table1"]
        exog_features = ctx["exog_m1"]
        pure_exog = ctx["pure_exog_m1"]
        future_pool = pd.DataFrame(index=range(horizon))
        future_pool["Fcst_GDP_YoY_for_t+1"] = grouped["GDP YoY Forecast"].to_numpy()
        future_pool["Fcst_Unemp_Q_for_t+1"] = grouped["Unemployment Rate YoY QUARTER"].to_numpy()
        future_pool["Fcst_BoC_for_t+1"] = grouped["Overnight Rate"].to_numpy()
        future_pool["Sent_BART_t"] = _forecast_sentiment(model_df["Sent_BART_t"], horizon)
        future_pool["M1_BART_x_GDP"] = future_pool["Sent_BART_t"] * future_pool["Fcst_GDP_YoY_for_t+1"]
        future_pool["M1_BART_x_Unemp"] = future_pool["Sent_BART_t"] * future_pool["Fcst_Unemp_Q_for_t+1"]
        future_pool["M1_BART_x_BoC"] = future_pool["Sent_BART_t"] * future_pool["Fcst_BoC_for_t+1"]
    else:
        model_df = ctx["model2_df"]
        train_valid_df = ctx["train_valid_df2"]
        selected_features = ctx["selected_features_m2"]
        best_ml_name = ctx["best_ml_m2"]
        global_best_name = ctx["global_best_m2"]
        final_table = ctx["final_table2"]
        exog_features = ctx["exog_m2"]
        pure_exog = ctx["pure_exog_m2"]
        future_pool = pd.DataFrame(index=range(horizon))
        future_pool["Rfsh_Fcst_GDP_YoY_1_month_prior_to_t+1"] = grouped["GDP YoY Forecast"].to_numpy()
        future_pool["Unemp_1_month_prior _to_t+1"] = grouped["Unemployment Rate YoY QUARTER"].to_numpy()
        future_pool["Rfsh_Fcst_BoC_1_month_prior_to_t+1"] = grouped["Overnight Rate"].to_numpy()
        future_pool["Sent_BART_t"] = _forecast_sentiment(model_df["Sent_BART_t"], horizon)
        future_pool["M2_BART_x_GDP"] = future_pool["Sent_BART_t"] * grouped["GDP YoY Forecast"].to_numpy()
        future_pool["M2_BART_x_Unemp"] = future_pool["Sent_BART_t"] * grouped["Unemployment Rate YoY QUARTER"].to_numpy()
        future_pool["M2_BART_x_BoC"] = future_pool["Sent_BART_t"] * grouped["Overnight Rate"].to_numpy()

    future_pool["PCL_t"] = np.nan

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_valid_df[selected_features])
    target = train_valid_df["PCL_t+1"]
    champion_model = ctx["sklearn"].base.clone(ctx["ml_models_dict"][best_ml_name])
    champion_model.fit(scaled_train, target)

    best_rmse = float(final_table.loc[final_table["Model"] == best_ml_name, "OOS RMSE"].iloc[0])
    last_known_pcl = float(model_df["PCL_t"].dropna().iloc[-1])
    current_pcl = last_known_pcl
    pcl_base_ml: list[float] = []

    for row_index in range(horizon):
        future_pool.loc[row_index, "PCL_t"] = current_pcl
        row = future_pool.iloc[row_index : row_index + 1][selected_features]
        prediction = float(champion_model.predict(scaler.transform(row))[0])
        pcl_base_ml.append(prediction)
        current_pcl = prediction

    base_values = np.asarray(pcl_base_ml, dtype=float)
    covid_shock, recovery_release, playback_curve = _playback_metrics(model_df, horizon)
    bear_values = base_values + covid_shock * playback_curve
    bull_values = base_values - recovery_release * playback_curve
    lower_values = base_values - 1.96 * best_rmse
    upper_values = base_values + 1.96 * best_rmse

    ts_values: np.ndarray | None = None
    if global_best_name != best_ml_name:
        current_exog = pure_exog if "Pure Macro" in global_best_name else exog_features
        try:
            if "VARMAX" in global_best_name:
                ts_values = (
                    VARMAX(
                        train_valid_df[["PCL_t+1"] + ctx["features_ts_hist"]],
                        exog=train_valid_df[current_exog],
                        order=(1, 0),
                    )
                    .fit(disp=False)
                    .forecast(steps=horizon, exog=future_pool[current_exog])
                    .iloc[:, 0]
                    .to_numpy()
                )
            elif "VAR" in global_best_name:
                ts_values = (
                    VAR(train_valid_df[["PCL_t+1"] + ctx["features_ts_hist"]])
                    .fit(maxlags=1)
                    .forecast(
                        train_valid_df[["PCL_t+1"] + ctx["features_ts_hist"]].to_numpy()[-1:],
                        steps=horizon,
                    )[:, 0]
                )
            elif "ARIMAX" in global_best_name:
                ts_values = np.asarray(
                    ARIMA(
                        train_valid_df["PCL_t+1"].to_numpy(),
                        exog=train_valid_df[current_exog].to_numpy(),
                        order=(1, 0, 0),
                    )
                    .fit()
                    .forecast(steps=horizon, exog=future_pool[current_exog].to_numpy()),
                    dtype=float,
                )
            elif "SARIMA" in global_best_name:
                ts_values = np.asarray(
                    ARIMA(train_valid_df["PCL_t+1"].to_numpy(), order=(1, 0, 0)).fit().forecast(steps=horizon),
                    dtype=float,
                )
        except Exception:
            ts_values = None

    result_df = pd.DataFrame(
        {
            "Quarter": quarter_labels,
            "ML Base": base_values,
            "Bear": bear_values,
            "Bull": bull_values,
            "95% CI Lower": lower_values,
            "95% CI Upper": upper_values,
        }
    )
    if ts_values is not None:
        result_df["TS Global Champion"] = np.asarray(ts_values, dtype=float)

    feature_df = future_pool[selected_features].copy()
    feature_df.insert(0, "Quarter", quarter_labels)

    payload: dict[str, Any] = {}
    for _, row in result_df.iterrows():
        quarter = row["Quarter"]
        features = feature_df.loc[feature_df["Quarter"] == quarter].iloc[0].to_dict()
        features.pop("Quarter", None)
        payload[quarter] = {
            "model": f"Aggregate Model {model_number}",
            "champions": {
                "ml_base": best_ml_name,
                "global_ts": global_best_name if ts_values is not None else None,
            },
            "inputs": features,
            "forecast": {
                "base": round(float(row["ML Base"]), 1),
                "bear": round(float(row["Bear"]), 1),
                "bull": round(float(row["Bull"]), 1),
                "ci_lower": round(float(row["95% CI Lower"]), 1),
                "ci_upper": round(float(row["95% CI Upper"]), 1),
                "ts_global_champion": round(float(row["TS Global Champion"]), 1)
                if "TS Global Champion" in row and pd.notna(row["TS Global Champion"])
                else None,
            },
        }

    return {
        "name": f"Aggregate Model {model_number}",
        "metadata": {
            "ml_champion": best_ml_name,
            "ts_champion": global_best_name if ts_values is not None else None,
        },
        "result_df": result_df.round(1),
        "feature_df": feature_df.round(4),
        "payload": payload,
    }


def _forecast_bloomberg_model(scenario_df: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, Any]:
    grouped = (
        scenario_df.groupby(["Year", "Quarter"], as_index=False)
        .agg(
            {
                "Bloomberg": "sum",
                "GDP YoY Forecast": "mean",
                "Unemployment Rate YoY QUARTER": "mean",
                "Overnight Rate": "mean",
            }
        )
        .sort_values(["Year", "Quarter"])
        .reset_index(drop=True)
    )
    forecast_index = _quarter_index(grouped)
    quarter_labels = _quarter_strings(forecast_index)
    horizon = len(grouped)

    future_pool = pd.DataFrame(index=range(horizon))
    future_pool["CloseIn_GDP"] = grouped["GDP YoY Forecast"].to_numpy()
    future_pool["Unemp_1_month_prior _to_t+1"] = grouped["Unemployment Rate YoY QUARTER"].to_numpy()
    future_pool["Rfsh_Fcst_BoC_1_month_prior_to_t+1"] = grouped["Overnight Rate"].to_numpy()
    future_pool["Sent_BART_t"] = _forecast_sentiment(ctx["model_df"]["Sent_BART_t"], horizon)
    future_pool["BART_x_CloseIn_Unemp"] = future_pool["Sent_BART_t"] * future_pool["Unemp_1_month_prior _to_t+1"]
    future_pool["BART_x_CloseIn_BoC"] = future_pool["Sent_BART_t"] * future_pool["Rfsh_Fcst_BoC_1_month_prior_to_t+1"]
    future_pool["BART_x_CloseIn_GDP"] = future_pool["Sent_BART_t"] * future_pool["CloseIn_GDP"]

    scaler = StandardScaler()
    selected_features = ctx["selected_features_ml"]
    train_valid_df = ctx["train_valid_df"]
    scaled_train = scaler.fit_transform(train_valid_df[selected_features])
    champion_model = _build_bloomberg_model(ctx["best_ml_model_name"])
    champion_model.fit(scaled_train, train_valid_df["PCL_Residual_t+1"])

    bbg_base = grouped["Bloomberg"].to_numpy(dtype=float)
    residual_delta = champion_model.predict(scaler.transform(future_pool[selected_features]))
    base_values = bbg_base + residual_delta

    covid_shock, recovery_release, playback_curve = _playback_metrics(ctx["model_df"], horizon)
    bear_values = base_values + covid_shock * playback_curve
    bull_values = base_values - recovery_release * playback_curve
    ci_spread = 1.96 * float(ctx["best_rmse"])
    lower_values = base_values - ci_spread
    upper_values = base_values + ci_spread

    result_df = pd.DataFrame(
        {
            "Quarter": quarter_labels,
            "Bloomberg Consensus": bbg_base,
            "ML Residual Delta": residual_delta,
            "ML Base": base_values,
            "Bear": bear_values,
            "Bull": bull_values,
            "95% CI Lower": lower_values,
            "95% CI Upper": upper_values,
        }
    )

    feature_df = future_pool[selected_features].copy()
    feature_df.insert(0, "Quarter", quarter_labels)

    payload: dict[str, Any] = {}
    for _, row in result_df.iterrows():
        quarter = row["Quarter"]
        features = feature_df.loc[feature_df["Quarter"] == quarter].iloc[0].to_dict()
        features.pop("Quarter", None)
        payload[quarter] = {
            "model": "Bloomberg Residual Model",
            "champions": {"ml_base": ctx["best_ml_model_name"]},
            "inputs": {
                "Bloomberg": round(float(row["Bloomberg Consensus"]), 1),
                **{key: round(float(value), 4) for key, value in features.items()},
            },
            "forecast": {
                "ml_delta": round(float(row["ML Residual Delta"]), 1),
                "base": round(float(row["ML Base"]), 1),
                "bear": round(float(row["Bear"]), 1),
                "bull": round(float(row["Bull"]), 1),
                "ci_lower": round(float(row["95% CI Lower"]), 1),
                "ci_upper": round(float(row["95% CI Upper"]), 1),
            },
        }

    return {
        "name": "Bloomberg Residual Model",
        "metadata": {"ml_champion": ctx["best_ml_model_name"]},
        "result_df": result_df.round(1),
        "feature_df": feature_df.round(4),
        "payload": payload,
    }


def run_all_forecasts(scenario_df: pd.DataFrame) -> dict[str, Any]:
    clean_df = _clean_scenario_df(scenario_df)
    aggregate_ctx = load_aggregate_context()
    bloomberg_ctx = load_bloomberg_context()

    aggregate_model_1 = _forecast_aggregate_model(clean_df, 1, aggregate_ctx)
    aggregate_model_2 = _forecast_aggregate_model(clean_df, 2, aggregate_ctx)
    bloomberg_model = _forecast_bloomberg_model(clean_df, bloomberg_ctx)

    combined_json = {
        "input_scenarios": clean_df.to_dict(orient="records"),
        "aggregate_model_1": aggregate_model_1["payload"],
        "aggregate_model_2": aggregate_model_2["payload"],
        "bloomberg_model": bloomberg_model["payload"],
    }

    return {
        "scenario_df": clean_df,
        "aggregate_model_1": aggregate_model_1,
        "aggregate_model_2": aggregate_model_2,
        "bloomberg_model": bloomberg_model,
        "json_payload": combined_json,
    }
