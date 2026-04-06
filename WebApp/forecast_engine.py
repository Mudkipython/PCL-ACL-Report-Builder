from __future__ import annotations

import contextlib
import io
import urllib3
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import r2_score
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
INDIVIDUAL_NO_SENTIMENT_NOTEBOOK = BASE_DIR / "Individual bank - Use Case 2 -Modeller.ipynb"
INDIVIDUAL_WITH_SENTIMENT_NOTEBOOK = BASE_DIR / "Individual bank with sentiment data - Use case 2 - Modeller.ipynb"
QUARTER_TO_DATE = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
DEFAULT_PLAYBACK_CURVE = np.array([0.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0], dtype=float)
CACHE_DIR = BASE_DIR / ".cache"
AGGREGATE_CACHE_PATH = CACHE_DIR / "aggregate_context.joblib"
BLOOMBERG_CACHE_PATH = CACHE_DIR / "bloomberg_context.joblib"
INDIVIDUAL_NO_SENTIMENT_CACHE_PATH = CACHE_DIR / "individual_no_sentiment_context.joblib"
INDIVIDUAL_WITH_SENTIMENT_CACHE_PATH = CACHE_DIR / "individual_with_sentiment_context.joblib"
NOT_OPENSSL_WARNING = getattr(urllib3.exceptions, "NotOpenSSLWarning", Warning)
CACHE_VERSION = 2
SCENARIO_COLUMNS = [
    "Bank",
    "Year",
    "Quarter",
    "Bloomberg",
    "GDP YoY Forecast",
    "Unemployment Rate YoY QUARTER",
    "Overnight Rate",
]


def _patched_notebook_code(code: str) -> str:
    return (
        code.replace("except ImportError:", "except Exception:")
        .replace("Complete_UC2_Forecast_Data.xlsx", str(WORKBOOK_PATH))
    )


def _source_signature() -> dict[str, float]:
    return {
        "cache_version": CACHE_VERSION,
        "engine_mtime": Path(__file__).stat().st_mtime,
        "workbook_mtime": WORKBOOK_PATH.stat().st_mtime,
        "aggregate_notebook_mtime": AGGREGATE_NOTEBOOK.stat().st_mtime,
        "bloomberg_notebook_mtime": BLOOMBERG_NOTEBOOK.stat().st_mtime,
        "individual_no_sentiment_notebook_mtime": INDIVIDUAL_NO_SENTIMENT_NOTEBOOK.stat().st_mtime,
        "individual_with_sentiment_notebook_mtime": INDIVIDUAL_WITH_SENTIMENT_NOTEBOOK.stat().st_mtime,
    }


def _execute_notebook_cells(notebook_path: Path, cell_indices: list[int]) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=NOT_OPENSSL_WARNING)
    namespace: dict[str, Any] = {"display": lambda *args, **kwargs: None}
    notebook = nbformat.read(notebook_path, as_version=4)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for index in cell_indices:
            code = _patched_notebook_code(notebook.cells[index].source)
            exec(compile(code, f"{notebook_path.name}#cell{index}", "exec"), namespace)

    return namespace


def _quarter_strings(index: pd.DatetimeIndex) -> list[str]:
    return [f"{date.year} Q{date.quarter}" for date in index]


def _quarter_label_to_timestamp(label: str) -> pd.Timestamp:
    year_str, quarter_str = label.split()
    return pd.Timestamp(f"{int(year_str)}-{QUARTER_TO_DATE[quarter_str]}")


def _quarter_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        df["Year"].astype(int).astype(str)
        + "-"
        + df["Quarter"].map(QUARTER_TO_DATE)
    )


@lru_cache(maxsize=1)
def valid_banks() -> tuple[str, ...]:
    try:
        sheet = pd.read_excel(WORKBOOK_PATH, sheet_name="Sheet1")
        banks = sorted(sheet["Bank"].dropna().astype(str).str.strip().unique().tolist())
        if banks:
            return tuple(banks)
    except Exception:
        pass
    return ("BMO", "BNS", "CIBC", "RBC", "TD")


def _clean_scenario_df(df: pd.DataFrame) -> pd.DataFrame:
    incoming = list(df.columns)
    missing = [column for column in SCENARIO_COLUMNS if column not in incoming]
    extra = [column for column in incoming if column not in SCENARIO_COLUMNS]
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing columns: {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected columns: {', '.join(extra)}")
        raise ValueError("Scenario template mismatch: " + "; ".join(problems))

    scenario_df = df[SCENARIO_COLUMNS].copy()
    scenario_df["Bank"] = scenario_df["Bank"].astype(str).str.strip().str.upper()
    scenario_df["Year"] = pd.to_numeric(scenario_df["Year"], errors="coerce").astype("Int64")
    scenario_df["Quarter"] = scenario_df["Quarter"].astype(str).str.upper().str.strip()

    for column in SCENARIO_COLUMNS[3:]:
        scenario_df[column] = pd.to_numeric(scenario_df[column], errors="coerce")

    scenario_df = scenario_df.dropna().copy()
    scenario_df["Year"] = scenario_df["Year"].astype(int)

    allowed_banks = set(valid_banks())
    invalid_banks = sorted(set(scenario_df["Bank"]) - allowed_banks)
    if invalid_banks:
        raise ValueError(f"Unsupported banks: {', '.join(invalid_banks)}")

    allowed_quarters = set(QUARTER_TO_DATE)
    invalid_quarters = sorted(set(scenario_df["Quarter"]) - allowed_quarters)
    if invalid_quarters:
        raise ValueError(f"Unsupported quarter labels: {', '.join(invalid_quarters)}")

    dupes = scenario_df.duplicated(subset=["Bank", "Year", "Quarter"], keep=False)
    if dupes.any():
        dup_df = scenario_df.loc[dupes, ["Bank", "Year", "Quarter"]].drop_duplicates()
        dup_str = ", ".join(
            f"{row.Bank}-{row.Year}-{row.Quarter}" for row in dup_df.itertuples(index=False)
        )
        raise ValueError(f"Duplicate bank-quarter rows found: {dup_str}")

    return scenario_df.sort_values(["Bank", "Year", "Quarter"]).reset_index(drop=True)


def _load_sheet1() -> pd.DataFrame:
    return pd.read_excel(WORKBOOK_PATH, sheet_name="Sheet1")


def default_quarter_template() -> pd.DataFrame:
    sheet1 = _load_sheet1()
    template = sheet1[SCENARIO_COLUMNS].copy()
    template["Bank"] = template["Bank"].astype(str).str.strip().str.upper()
    template["Year"] = pd.to_numeric(template["Year"], errors="coerce").astype("Int64")
    template = template.dropna().copy()
    template["Year"] = template["Year"].astype(int)
    return template.sort_values(["Bank", "Year", "Quarter"]).reset_index(drop=True)


def default_scenario_template() -> pd.DataFrame:
    return default_quarter_template()


def template_csv_bytes() -> bytes:
    return default_quarter_template().to_csv(index=False).encode("utf-8")


def template_excel_bytes() -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        default_quarter_template().to_excel(writer, index=False, sheet_name="Scenario Template")
    return output.getvalue()


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


def _build_sentiment_bundle(history: pd.Series) -> dict[str, Any]:
    clean_history = history.dropna().astype(float).to_numpy()
    if len(clean_history) == 0:
        return {"type": "constant", "last_value": 0.0}

    if HAS_PMDARIMA:
        try:
            return {
                "type": "auto_arima",
                "model": auto_arima(clean_history, seasonal=False, suppress_warnings=True),
                "last_value": float(clean_history[-1]),
            }
        except Exception:
            pass

    try:
        return {
            "type": "arima",
            "model": ARIMA(clean_history, order=(1, 0, 0)).fit(),
            "last_value": float(clean_history[-1]),
        }
    except Exception:
        return {"type": "constant", "last_value": float(clean_history[-1])}


def _predict_sentiment(bundle: dict[str, Any], steps: int) -> np.ndarray:
    if steps <= 0:
        return np.array([], dtype=float)
    if bundle["type"] == "auto_arima":
        try:
            return np.asarray(bundle["model"].predict(n_periods=steps), dtype=float)
        except Exception:
            return np.repeat(bundle["last_value"], steps).astype(float)
    if bundle["type"] == "arima":
        try:
            return np.asarray(bundle["model"].forecast(steps=steps), dtype=float)
        except Exception:
            return np.repeat(bundle["last_value"], steps).astype(float)
    return np.repeat(bundle["last_value"], steps).astype(float)


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


def _clean_num(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s in {"—", "", "-", "NA", "N/A", "nan", "NaN", "None", "null", "NULL", "Unnamed:23", "Unnamed:39"}:
        return np.nan
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    s = (
        s.replace(",", "")
        .replace("$", "")
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace('"', "")
        .replace("\u200b", "")
        .replace("\xa0", "")
    )
    try:
        return float(s)
    except Exception:
        return np.nan


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def _unique_keep_order(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def _load_individual_raw_df() -> pd.DataFrame:
    raw_df = pd.read_excel(WORKBOOK_PATH, sheet_name="Forecast Data", header=0)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    if "Bank" not in raw_df.columns and str(raw_df.iloc[0, 0]).strip() == "Bank":
        raw_df.columns = [str(c).strip() for c in raw_df.iloc[0].tolist()]
        raw_df = raw_df.iloc[1:].reset_index(drop=True)

    required = {"Bank", "Year", "Quarter"}
    if not required.issubset(set(raw_df.columns)):
        raise ValueError("Workbook parsing failed for individual-bank models.")

    cols_to_clean = [
        "PCL_t+1",
        "PCL_t",
        "Sent_BART_t",
        "Fcst_GDP_YoY_for_t+1",
        "Fcst_Unemp_A_for_t+1",
        "Fcst_BoC_for_t+1",
        "GDP_YoY_t",
        "Overnight_Rate_t",
    ]
    for column in cols_to_clean:
        if column in raw_df.columns:
            raw_df[column] = raw_df[column].apply(_clean_num)

    raw_df["Bank"] = raw_df["Bank"].astype(str).str.strip().str.upper()
    raw_df["Year"] = pd.to_numeric(raw_df["Year"], errors="coerce")
    raw_df["Quarter"] = raw_df["Quarter"].astype(str).str.strip().str.upper()
    raw_df = raw_df.dropna(subset=["Bank", "Year", "Quarter"]).copy()
    raw_df["Year"] = raw_df["Year"].astype(int)
    raw_df = raw_df[raw_df["Bank"].isin(valid_banks())].copy()
    raw_df = raw_df.sort_values(by=["Bank", "Year", "Quarter"]).reset_index(drop=True)
    return raw_df


def _select_features_train_only(
    train_df: pd.DataFrame,
    candidate_features: list[str],
    fallback_features: list[str],
    target_col: str = "PCL_t+1",
) -> list[str]:
    candidate_features = [c for c in _unique_keep_order(candidate_features) if c in train_df.columns]
    fallback_features = [c for c in _unique_keep_order(fallback_features) if c in train_df.columns]
    fs_df = train_df.dropna(subset=candidate_features + [target_col]).copy()

    if len(candidate_features) == 0 or len(fs_df) < 6:
        return fallback_features

    n_splits_fs = min(3, len(fs_df) - 1)
    if n_splits_fs < 2:
        return fallback_features

    try:
        tscv_fs = TimeSeriesSplit(n_splits=n_splits_fs)
        scaler_fs = StandardScaler()
        x_fs = scaler_fs.fit_transform(fs_df[candidate_features])
        y_fs = fs_df[target_col].values

        elastic_fs = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            cv=tscv_fs,
            random_state=42,
        )
        elastic_fs.fit(x_fs, y_fs)

        selected = [feat for feat, coef in zip(candidate_features, elastic_fs.coef_) if abs(coef) > 1e-4]
        selected = _unique_keep_order(selected)
        if len(selected) <= 1:
            return fallback_features
        return selected
    except Exception:
        return fallback_features


def _build_individual_ml_models(tscv: TimeSeriesSplit) -> dict[str, Any]:
    return {
        "06. ML: Lasso": LassoCV(alphas=np.logspace(-2, 4, 50), cv=tscv),
        "07. ML: Ridge": RidgeCV(alphas=np.logspace(-2, 4, 50)),
        "08. ML: Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        "09. ML: Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "10. ML: Elastic Net": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=tscv, random_state=42),
    }


def _make_individual_future_frame(
    scenario_bank_df: pd.DataFrame,
    sentiment_bundle: dict[str, Any] | None,
    horizon: int,
) -> tuple[pd.DataFrame, list[str]]:
    bank_df = scenario_bank_df.copy().sort_values(["Year", "Quarter"]).head(horizon).reset_index(drop=True)
    future_dates = _quarter_index(bank_df)
    labels = _quarter_strings(future_dates)

    future_pool = pd.DataFrame(index=range(len(bank_df)))
    future_pool["Fcst_GDP_YoY_for_t+1"] = bank_df["GDP YoY Forecast"].to_numpy(dtype=float)
    future_pool["Fcst_Unemp_A_for_t+1"] = bank_df["Unemployment Rate YoY QUARTER"].to_numpy(dtype=float)
    future_pool["Fcst_BoC_for_t+1"] = bank_df["Overnight Rate"].to_numpy(dtype=float)
    future_pool["GDP_YoY_t"] = bank_df["GDP YoY Forecast"].to_numpy(dtype=float)
    future_pool["Overnight_Rate_t"] = bank_df["Overnight Rate"].to_numpy(dtype=float)
    future_pool["Sent_BART_t"] = (
        _predict_sentiment(sentiment_bundle, len(bank_df)) if sentiment_bundle is not None else np.nan
    )
    future_pool["PCL_t"] = np.nan
    return future_pool, labels


def _fit_individual_winner_bundle(
    model_name: str,
    train_valid_df: pd.DataFrame,
    selected_features: list[str],
    features_ts_hist: list[str],
) -> dict[str, Any]:
    y_full = pd.to_numeric(train_valid_df["PCL_t+1"], errors="coerce").dropna().copy()
    last_level = float(y_full.iloc[-1]) if len(y_full) > 0 else np.nan
    exog_features = _unique_keep_order([feature for feature in selected_features if feature != "PCL_t"])

    if model_name == "01. Baseline: Naive (Last Qtr)":
        return {"type": "baseline", "model_name": model_name, "last_level": last_level}

    if model_name == "02. TS: SARIMA (Blind)":
        try:
            if HAS_PMDARIMA:
                model = auto_arima(y_full, seasonal=False, suppress_warnings=True, error_action="ignore")
                return {"type": "sarima_auto", "model_name": model_name, "model": model, "last_level": last_level}
            model = ARIMA(y_full.values, order=(1, 0, 0)).fit()
            return {"type": "sarima", "model_name": model_name, "model": model, "last_level": last_level}
        except Exception:
            return {"type": "baseline", "model_name": model_name, "last_level": last_level}

    if model_name == "03. TS: VAR (Blind)":
        try:
            var_cols = _unique_keep_order(["PCL_t+1"] + [c for c in features_ts_hist if c in train_valid_df.columns])
            var_input = train_valid_df[var_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
            if len(var_input) > 5 and var_input.shape[1] >= 2:
                model = VAR(var_input).fit(maxlags=1)
                return {
                    "type": "var",
                    "model_name": model_name,
                    "model": model,
                    "last_state": var_input.values[-model.k_ar :],
                    "last_level": last_level,
                }
        except Exception:
            pass
        return {"type": "baseline", "model_name": model_name, "last_level": last_level}

    if model_name == "04. TS: ARIMAX (Macro-Aware)":
        try:
            needed = _unique_keep_order(["PCL_t+1"] + exog_features)
            fit_df = train_valid_df[needed].apply(pd.to_numeric, errors="coerce").dropna().copy()
            if len(fit_df) >= 6 and len(exog_features) > 0:
                model = ARIMA(fit_df["PCL_t+1"].values, exog=fit_df[exog_features].values, order=(1, 0, 0)).fit()
                return {
                    "type": "arimax",
                    "model_name": model_name,
                    "model": model,
                    "exog_features": exog_features,
                    "last_level": last_level,
                }
        except Exception:
            pass
        return {"type": "baseline", "model_name": model_name, "last_level": last_level}

    if model_name == "05. TS: VARMAX (Macro-Aware)":
        try:
            endog_cols = _unique_keep_order(["PCL_t+1"] + [c for c in features_ts_hist if c in train_valid_df.columns])
            fit_endog = train_valid_df[endog_cols].apply(pd.to_numeric, errors="coerce")
            fit_exog = train_valid_df[exog_features].apply(pd.to_numeric, errors="coerce")
            valid_idx = fit_endog.dropna().index.intersection(fit_exog.dropna().index)
            fit_endog = fit_endog.loc[valid_idx]
            fit_exog = fit_exog.loc[valid_idx]
            if len(fit_endog) >= 6 and len(exog_features) > 0:
                model = VARMAX(fit_endog, exog=fit_exog, order=(1, 0)).fit(disp=False, maxiter=200)
                return {
                    "type": "varmax",
                    "model_name": model_name,
                    "model": model,
                    "exog_features": exog_features,
                    "last_level": last_level,
                }
        except Exception:
            pass
        return {"type": "baseline", "model_name": model_name, "last_level": last_level}

    try:
        ml_df = train_valid_df[selected_features + ["PCL_t+1"]].apply(pd.to_numeric, errors="coerce").dropna().copy()
        if len(ml_df) < 6 or len(selected_features) == 0:
            return {"type": "baseline", "model_name": model_name, "last_level": last_level}
        n_splits = min(3, len(ml_df) - 1)
        if n_splits < 2:
            return {"type": "baseline", "model_name": model_name, "last_level": last_level}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        ml_models = _build_individual_ml_models(tscv)
        model = ml_models[model_name]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(ml_df[selected_features])
        fitted = model.fit(x_train, ml_df["PCL_t+1"].values)
        return {
            "type": "ml",
            "model_name": model_name,
            "model": fitted,
            "scaler": scaler,
            "selected_features": selected_features,
            "last_level": last_level,
        }
    except Exception:
        return {"type": "baseline", "model_name": model_name, "last_level": last_level}


def _build_individual_scorecard(prediction_records: list[dict[str, Any]]) -> pd.DataFrame:
    prediction_df = pd.DataFrame(prediction_records)
    rows: list[dict[str, Any]] = []
    for model_name, group in prediction_df.groupby("Model"):
        actual = group["Actual ($M)"].to_numpy(dtype=float)
        pred = group["Predicted ($M)"].to_numpy(dtype=float)
        rows.append(
            {
                "Model": model_name,
                "OOS R2": r2_score(actual, pred) if len(actual) >= 2 else np.nan,
                "OOS MAPE": _safe_mape(actual, pred),
                "OOS RMSE": _safe_rmse(actual, pred),
                "N_OOS": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["OOS RMSE", "OOS MAPE"]).reset_index(drop=True)


def _one_step_individual_prediction(
    model_name: str,
    train_df: pd.DataFrame,
    test_row: pd.DataFrame,
    selected_features: list[str],
    features_ts_hist: list[str],
) -> float:
    y_train = pd.to_numeric(train_df["PCL_t+1"], errors="coerce").dropna().copy()
    last_level = float(y_train.iloc[-1]) if len(y_train) > 0 else np.nan
    exog_features = _unique_keep_order([feature for feature in selected_features if feature != "PCL_t"])

    try:
        if model_name == "01. Baseline: Naive (Last Qtr)":
            return float(pd.to_numeric(train_df["PCL_t"], errors="coerce").dropna().iloc[-1])

        if model_name == "02. TS: SARIMA (Blind)":
            if HAS_PMDARIMA:
                model = auto_arima(y_train, seasonal=False, suppress_warnings=True, error_action="ignore")
                return float(model.predict(n_periods=1)[0])
            return float(ARIMA(y_train.values, order=(1, 0, 0)).fit().forecast(steps=1)[0])

        if model_name == "03. TS: VAR (Blind)":
            var_cols = _unique_keep_order(["PCL_t+1"] + [c for c in features_ts_hist if c in train_df.columns])
            var_input = train_df[var_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
            if len(var_input) > 5 and var_input.shape[1] >= 2:
                fit = VAR(var_input).fit(maxlags=1)
                return float(fit.forecast(var_input.values[-fit.k_ar :], steps=1)[0, 0])
            return last_level

        if model_name == "04. TS: ARIMAX (Macro-Aware)":
            needed = _unique_keep_order(["PCL_t+1"] + exog_features)
            fit_df = train_df[needed].apply(pd.to_numeric, errors="coerce").dropna().copy()
            if len(fit_df) >= 6 and len(exog_features) > 0:
                fit = ARIMA(fit_df["PCL_t+1"].values, exog=fit_df[exog_features].values, order=(1, 0, 0)).fit()
                test_exog = test_row[exog_features].apply(pd.to_numeric, errors="coerce").to_numpy()
                return float(fit.forecast(steps=1, exog=test_exog)[0])
            return last_level

        if model_name == "05. TS: VARMAX (Macro-Aware)":
            endog_cols = _unique_keep_order(["PCL_t+1"] + [c for c in features_ts_hist if c in train_df.columns])
            fit_endog = train_df[endog_cols].apply(pd.to_numeric, errors="coerce")
            fit_exog = train_df[exog_features].apply(pd.to_numeric, errors="coerce")
            valid_idx = fit_endog.dropna().index.intersection(fit_exog.dropna().index)
            fit_endog = fit_endog.loc[valid_idx]
            fit_exog = fit_exog.loc[valid_idx]
            if len(fit_endog) >= 6 and len(exog_features) > 0:
                fit = VARMAX(fit_endog, exog=fit_exog, order=(1, 0)).fit(disp=False, maxiter=200)
                test_exog = test_row[exog_features].apply(pd.to_numeric, errors="coerce")
                return float(fit.forecast(steps=1, exog=test_exog).iloc[0, 0])
            return last_level

        ml_df = train_df[selected_features + ["PCL_t+1"]].apply(pd.to_numeric, errors="coerce").dropna().copy()
        if len(ml_df) < 6 or len(selected_features) == 0:
            return last_level
        n_splits = min(3, len(ml_df) - 1)
        if n_splits < 2:
            return last_level
        tscv = TimeSeriesSplit(n_splits=n_splits)
        ml_models = _build_individual_ml_models(tscv)
        model = ml_models[model_name]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(ml_df[selected_features])
        model.fit(x_train, ml_df["PCL_t+1"].values)
        x_test = scaler.transform(test_row[selected_features].apply(pd.to_numeric, errors="coerce"))
        return float(model.predict(x_test)[0])
    except Exception:
        return last_level


def _build_individual_variant_cache(variant_name: str) -> dict[str, Any]:
    raw_df = _load_individual_raw_df()
    if variant_name == "individual_no_sentiment":
        fill_cols = ["PCL_t", "Fcst_GDP_YoY_for_t+1", "Fcst_Unemp_A_for_t+1", "Fcst_BoC_for_t+1"]
        candidate_features = ["PCL_t", "Fcst_GDP_YoY_for_t+1", "Fcst_Unemp_A_for_t+1", "Fcst_BoC_for_t+1"]
        fallback_features = ["PCL_t", "Fcst_GDP_YoY_for_t+1"]
        lower_bound = pd.Timestamp("2021-12-31")
        include_sentiment = False
    else:
        fill_cols = ["PCL_t", "Sent_BART_t", "Fcst_GDP_YoY_for_t+1", "Fcst_Unemp_A_for_t+1", "Fcst_BoC_for_t+1"]
        candidate_features = ["PCL_t", "Sent_BART_t", "Fcst_GDP_YoY_for_t+1", "Fcst_Unemp_A_for_t+1", "Fcst_BoC_for_t+1"]
        fallback_features = ["PCL_t", "Sent_BART_t", "Fcst_GDP_YoY_for_t+1"]
        lower_bound = None
        include_sentiment = True

    for column in fill_cols:
        if column in raw_df.columns:
            raw_df[column] = raw_df.groupby("Bank")[column].transform(lambda x: x.ffill().bfill())

    raw_df["date_idx"] = pd.to_datetime(
        raw_df["Year"].astype(int).astype(str) + "-" + raw_df["Quarter"].map(QUARTER_TO_DATE),
        errors="coerce",
    )
    raw_df = raw_df.dropna(subset=["date_idx"]).copy()
    raw_df = raw_df[raw_df["date_idx"] <= "2026-03-31"].copy()
    if lower_bound is not None:
        raw_df = raw_df[raw_df["date_idx"] >= lower_bound].copy()

    features_ts_hist = ["GDP_YoY_t", "Overnight_Rate_t"]
    bank_cache: dict[str, Any] = {}
    skipped_banks: dict[str, str] = {}

    for bank in valid_banks():
        bank_df = raw_df[raw_df["Bank"] == bank].copy()
        if bank_df.empty:
            skipped_banks[bank] = "No workbook history found."
            continue
        bank_df = bank_df.set_index("date_idx").sort_index()

        keep_cols = _unique_keep_order(["Bank", "PCL_t+1", "PCL_t", "Year", "Quarter"] + candidate_features + features_ts_hist)
        keep_cols = [column for column in keep_cols if column in bank_df.columns]
        model_df = bank_df[keep_cols].copy()
        train_valid_df = model_df.dropna(subset=["PCL_t+1"]).copy()

        if len(train_valid_df) < 8 or len(train_valid_df) <= 4:
            skipped_banks[bank] = "Insufficient history after notebook filters."
            continue

        prediction_records: list[dict[str, Any]] = []
        test_indices = list(range(len(train_valid_df) - 4, len(train_valid_df)))

        for t in test_indices:
            train_df = train_valid_df.iloc[:t].copy()
            test_row = train_valid_df.iloc[t : t + 1].copy()
            selected_features = _select_features_train_only(train_df, candidate_features, fallback_features)
            actual = float(pd.to_numeric(test_row["PCL_t+1"], errors="coerce").iloc[0])

            for model_name in [
                "01. Baseline: Naive (Last Qtr)",
                "02. TS: SARIMA (Blind)",
                "03. TS: VAR (Blind)",
                "04. TS: ARIMAX (Macro-Aware)",
                "05. TS: VARMAX (Macro-Aware)",
                "06. ML: Lasso",
                "07. ML: Ridge",
                "08. ML: Random Forest",
                "09. ML: Gradient Boosting",
                "10. ML: Elastic Net",
            ]:
                pred = _one_step_individual_prediction(model_name, train_df, test_row, selected_features, features_ts_hist)
                prediction_records.append(
                    {
                        "Bank": bank,
                        "Quarter": f"{test_row.index[0].year} Q{test_row.index[0].quarter}",
                        "Model": model_name,
                        "Predicted ($M)": pred,
                        "Actual ($M)": actual,
                    }
                )

        scorecard_df = _build_individual_scorecard(prediction_records)
        best_row = scorecard_df.iloc[0]
        selected_features_final = _select_features_train_only(train_valid_df, candidate_features, fallback_features)
        winner_bundle = _fit_individual_winner_bundle(best_row["Model"], train_valid_df, selected_features_final, features_ts_hist)

        bank_cache[bank] = {
            "bank": bank,
            "variant": variant_name,
            "model_df": model_df[["PCL_t"] + (["Sent_BART_t"] if include_sentiment and "Sent_BART_t" in model_df.columns else [])].copy(),
            "sentiment_bundle": _build_sentiment_bundle(model_df["Sent_BART_t"]) if include_sentiment and "Sent_BART_t" in model_df.columns else None,
            "selected_features": selected_features_final,
            "scorecard_df": scorecard_df,
            "prediction_df": pd.DataFrame(prediction_records),
            "winner_model": str(best_row["Model"]),
            "best_rmse": float(best_row["OOS RMSE"]),
            "winner_bundle": winner_bundle,
            "playback": _playback_metrics(model_df, len(DEFAULT_PLAYBACK_CURVE)),
            "last_known_pcl": float(pd.to_numeric(model_df["PCL_t"], errors="coerce").dropna().iloc[-1]),
        }

    cache = {
        "signature": _source_signature(),
        "variant_name": variant_name,
        "features_ts_hist": features_ts_hist,
        "banks": sorted(bank_cache.keys()),
        "skipped_banks": skipped_banks,
        "banks_data": bank_cache,
    }
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = (
        INDIVIDUAL_NO_SENTIMENT_CACHE_PATH
        if variant_name == "individual_no_sentiment"
        else INDIVIDUAL_WITH_SENTIMENT_CACHE_PATH
    )
    joblib.dump(cache, cache_path)
    return cache


def _forecast_individual_variant(scenario_df: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, Any]:
    per_bank: dict[str, Any] = {}
    combined_rows: list[pd.DataFrame] = []
    payload: dict[str, Any] = {}

    for bank in ctx["banks"]:
        bank_scenario = scenario_df[scenario_df["Bank"] == bank].copy().sort_values(["Year", "Quarter"])
        if bank_scenario.empty:
            continue

        bank_ctx = ctx["banks_data"][bank]
        future_pool, quarter_labels = _make_individual_future_frame(bank_scenario, bank_ctx["sentiment_bundle"], horizon=4)
        winner_bundle = bank_ctx["winner_bundle"]
        horizon = len(quarter_labels)
        base_values: np.ndarray

        if winner_bundle["type"] == "baseline":
            base_values = np.repeat(bank_ctx["last_known_pcl"], horizon).astype(float)
        elif winner_bundle["type"] == "sarima_auto":
            base_values = np.asarray(winner_bundle["model"].predict(n_periods=horizon), dtype=float)
        elif winner_bundle["type"] == "sarima":
            base_values = np.asarray(winner_bundle["model"].forecast(steps=horizon), dtype=float)
        elif winner_bundle["type"] == "var":
            base_values = np.asarray(
                winner_bundle["model"].forecast(winner_bundle["last_state"], steps=horizon)[:, 0],
                dtype=float,
            )
        elif winner_bundle["type"] == "arimax":
            base_values = np.asarray(
                winner_bundle["model"].forecast(
                    steps=horizon,
                    exog=future_pool[winner_bundle["exog_features"]].to_numpy(),
                ),
                dtype=float,
            )
        elif winner_bundle["type"] == "varmax":
            base_values = winner_bundle["model"].forecast(
                steps=horizon,
                exog=future_pool[winner_bundle["exog_features"]],
            ).iloc[:, 0].to_numpy(dtype=float)
        else:
            current_pcl = bank_ctx["last_known_pcl"]
            preds: list[float] = []
            selected_features = winner_bundle["selected_features"]
            for row_index in range(horizon):
                future_pool.loc[row_index, "PCL_t"] = current_pcl
                row = future_pool.iloc[row_index : row_index + 1][selected_features]
                pred = float(winner_bundle["model"].predict(winner_bundle["scaler"].transform(row))[0])
                preds.append(pred)
                current_pcl = pred
            base_values = np.asarray(preds, dtype=float)

        covid_shock, recovery_release, full_curve = bank_ctx["playback"]
        playback_curve = full_curve[:horizon]
        bear_values = base_values + covid_shock * playback_curve
        bull_values = base_values - recovery_release * playback_curve
        lower_values = base_values - 1.96 * bank_ctx["best_rmse"]
        upper_values = base_values + 1.96 * bank_ctx["best_rmse"]

        result_df = pd.DataFrame(
            {
                "Bank": bank,
                "Quarter": quarter_labels,
                "Winning Model": bank_ctx["winner_model"],
                "ML Base": base_values,
                "Bear": bear_values,
                "Bull": bull_values,
                "95% CI Lower": lower_values,
                "95% CI Upper": upper_values,
            }
        ).round(1)

        feature_df = future_pool[bank_ctx["selected_features"]].copy()
        feature_df.insert(0, "Quarter", quarter_labels)
        feature_df.insert(0, "Bank", bank)
        feature_df = feature_df.round(4)

        bank_payload: dict[str, Any] = {}
        for _, row in result_df.iterrows():
            quarter = row["Quarter"]
            feature_row = feature_df.loc[feature_df["Quarter"] == quarter].iloc[0].to_dict()
            feature_row.pop("Bank", None)
            feature_row.pop("Quarter", None)
            bank_payload[quarter] = {
                "bank": bank,
                "model": bank_ctx["winner_model"],
                "inputs": feature_row,
                "forecast": {
                    "base": round(float(row["ML Base"]), 1),
                    "bear": round(float(row["Bear"]), 1),
                    "bull": round(float(row["Bull"]), 1),
                    "ci_lower": round(float(row["95% CI Lower"]), 1),
                    "ci_upper": round(float(row["95% CI Upper"]), 1),
                },
            }

        per_bank[bank] = {
            "metadata": {
                "winner_model": bank_ctx["winner_model"],
                "best_rmse": round(bank_ctx["best_rmse"], 2),
                "selected_features": bank_ctx["selected_features"],
            },
            "result_df": result_df,
            "feature_df": feature_df,
            "scorecard_df": bank_ctx["scorecard_df"].round(4),
            "payload": bank_payload,
        }
        payload[bank] = bank_payload
        combined_rows.append(result_df)

    combined_result_df = (
        pd.concat(combined_rows, axis=0).reset_index(drop=True) if combined_rows else pd.DataFrame()
    )
    winner_summary_df = pd.DataFrame(
        [
            {
                "Bank": bank,
                "Winning Model": per_bank[bank]["metadata"]["winner_model"],
                "Best OOS RMSE": per_bank[bank]["metadata"]["best_rmse"],
                "Selected Features": ", ".join(per_bank[bank]["metadata"]["selected_features"]),
            }
            for bank in per_bank
        ]
    )

    latest_quarter_summary = pd.DataFrame()
    if not combined_result_df.empty:
        latest_quarter = max(combined_result_df["Quarter"].unique().tolist(), key=_quarter_label_to_timestamp)
        latest_quarter_summary = combined_result_df[combined_result_df["Quarter"] == latest_quarter].copy()

    return {
        "name": ctx["variant_name"],
        "banks": list(per_bank.keys()),
        "skipped_banks": ctx["skipped_banks"],
        "per_bank": per_bank,
        "combined_result_df": combined_result_df,
        "winner_summary_df": winner_summary_df,
        "latest_quarter_summary_df": latest_quarter_summary,
        "payload": payload,
    }


def _build_aggregate_cache() -> dict[str, Any]:
    ctx = _execute_notebook_cells(AGGREGATE_NOTEBOOK, [4, 5, 9, 10])

    aggregate_cache: dict[str, Any] = {"signature": _source_signature(), "models": {}}
    for model_number in (1, 2):
        model_df = ctx[f"model{model_number}_df"]
        train_valid_df = ctx[f"train_valid_df{model_number}"]
        selected_features = ctx[f"selected_features_m{model_number}"]
        best_ml_name = ctx[f"best_ml_m{model_number}"]
        global_best_name = ctx[f"global_best_m{model_number}"]
        final_table = ctx[f"final_table{model_number}"]
        exog_features = ctx[f"exog_m{model_number}"]
        pure_exog = ctx[f"pure_exog_m{model_number}"]

        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_valid_df[selected_features])
        champion_model = ctx["sklearn"].base.clone(ctx["ml_models_dict"][best_ml_name])
        champion_model.fit(scaled_train, train_valid_df["PCL_t+1"])

        playback = _playback_metrics(model_df, len(DEFAULT_PLAYBACK_CURVE))
        best_rmse = float(final_table.loc[final_table["Model"] == best_ml_name, "OOS RMSE"].iloc[0])
        last_known_pcl = float(model_df["PCL_t"].dropna().iloc[-1])

        ts_bundle: dict[str, Any] | None = None
        if global_best_name != best_ml_name:
            ts_type = "unknown"
            current_exog = pure_exog if "Pure Macro" in global_best_name else exog_features
            try:
                if "VARMAX" in global_best_name:
                    ts_type = "varmax"
                    ts_model = VARMAX(
                        train_valid_df[["PCL_t+1"] + ctx["features_ts_hist"]],
                        exog=train_valid_df[current_exog],
                        order=(1, 0),
                    ).fit(disp=False)
                    ts_bundle = {"type": ts_type, "model": ts_model, "exog_features": current_exog}
                elif "VAR" in global_best_name:
                    ts_type = "var"
                    ts_model = VAR(train_valid_df[["PCL_t+1"] + ctx["features_ts_hist"]]).fit(maxlags=1)
                    ts_bundle = {
                        "type": ts_type,
                        "model": ts_model,
                        "last_state": train_valid_df[["PCL_t+1"] + ctx["features_ts_hist"]].to_numpy()[-1:],
                    }
                elif "ARIMAX" in global_best_name:
                    ts_type = "arimax"
                    ts_model = ARIMA(
                        train_valid_df["PCL_t+1"].to_numpy(),
                        exog=train_valid_df[current_exog].to_numpy(),
                        order=(1, 0, 0),
                    ).fit()
                    ts_bundle = {"type": ts_type, "model": ts_model, "exog_features": current_exog}
                elif "SARIMA" in global_best_name:
                    ts_type = "sarima"
                    ts_model = ARIMA(train_valid_df["PCL_t+1"].to_numpy(), order=(1, 0, 0)).fit()
                    ts_bundle = {"type": ts_type, "model": ts_model}
            except Exception:
                ts_bundle = None

        aggregate_cache["models"][model_number] = {
            "model_df": model_df[["PCL_t", "Sent_BART_t"]].copy(),
            "sentiment_bundle": _build_sentiment_bundle(model_df["Sent_BART_t"]),
            "selected_features": selected_features,
            "best_ml_name": best_ml_name,
            "global_best_name": global_best_name,
            "best_rmse": best_rmse,
            "last_known_pcl": last_known_pcl,
            "playback": playback,
            "scaler": scaler,
            "champion_model": champion_model,
            "ts_bundle": ts_bundle,
        }

    CACHE_DIR.mkdir(exist_ok=True)
    joblib.dump(aggregate_cache, AGGREGATE_CACHE_PATH)
    return aggregate_cache


def _build_bloomberg_cache() -> dict[str, Any]:
    ctx = _execute_notebook_cells(BLOOMBERG_NOTEBOOK, [5, 6])
    selected_features = ctx["selected_features_ml"]
    train_valid_df = ctx["train_valid_df"]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_valid_df[selected_features])
    champion_model = _build_bloomberg_model(ctx["best_ml_model_name"])
    champion_model.fit(scaled_train, train_valid_df["PCL_Residual_t+1"])

    bloomberg_cache = {
        "signature": _source_signature(),
        "selected_features": selected_features,
        "best_ml_model_name": ctx["best_ml_model_name"],
        "best_rmse": float(ctx["best_rmse"]),
        "model_df": ctx["model_df"][["PCL_t", "Sent_BART_t"]].copy(),
        "sentiment_bundle": _build_sentiment_bundle(ctx["model_df"]["Sent_BART_t"]),
        "playback": _playback_metrics(ctx["model_df"], len(DEFAULT_PLAYBACK_CURVE)),
        "scaler": scaler,
        "champion_model": champion_model,
    }
    CACHE_DIR.mkdir(exist_ok=True)
    joblib.dump(bloomberg_cache, BLOOMBERG_CACHE_PATH)
    return bloomberg_cache


def _load_cached_asset(cache_path: Path, builder) -> dict[str, Any]:
    if cache_path.exists():
        cache = joblib.load(cache_path)
        if cache.get("signature") == _source_signature():
            return cache
    return builder()


@lru_cache(maxsize=1)
def load_aggregate_context() -> dict[str, Any]:
    return _load_cached_asset(AGGREGATE_CACHE_PATH, _build_aggregate_cache)


@lru_cache(maxsize=1)
def load_bloomberg_context() -> dict[str, Any]:
    return _load_cached_asset(BLOOMBERG_CACHE_PATH, _build_bloomberg_cache)


@lru_cache(maxsize=1)
def load_individual_no_sentiment_context() -> dict[str, Any]:
    return _load_cached_asset(
        INDIVIDUAL_NO_SENTIMENT_CACHE_PATH,
        lambda: _build_individual_variant_cache("individual_no_sentiment"),
    )


@lru_cache(maxsize=1)
def load_individual_with_sentiment_context() -> dict[str, Any]:
    return _load_cached_asset(
        INDIVIDUAL_WITH_SENTIMENT_CACHE_PATH,
        lambda: _build_individual_variant_cache("individual_with_sentiment"),
    )


def _forecast_aggregate_model(
    scenario_df: pd.DataFrame,
    model_number: int,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    model_cache = ctx["models"][model_number]
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
        model_df = model_cache["model_df"]
        future_pool = pd.DataFrame(index=range(horizon))
        future_pool["Fcst_GDP_YoY_for_t+1"] = grouped["GDP YoY Forecast"].to_numpy()
        future_pool["Fcst_Unemp_Q_for_t+1"] = grouped["Unemployment Rate YoY QUARTER"].to_numpy()
        future_pool["Fcst_BoC_for_t+1"] = grouped["Overnight Rate"].to_numpy()
        future_pool["Sent_BART_t"] = _predict_sentiment(model_cache["sentiment_bundle"], horizon)
        future_pool["M1_BART_x_GDP"] = future_pool["Sent_BART_t"] * future_pool["Fcst_GDP_YoY_for_t+1"]
        future_pool["M1_BART_x_Unemp"] = future_pool["Sent_BART_t"] * future_pool["Fcst_Unemp_Q_for_t+1"]
        future_pool["M1_BART_x_BoC"] = future_pool["Sent_BART_t"] * future_pool["Fcst_BoC_for_t+1"]
    else:
        model_df = model_cache["model_df"]
        future_pool = pd.DataFrame(index=range(horizon))
        future_pool["Rfsh_Fcst_GDP_YoY_1_month_prior_to_t+1"] = grouped["GDP YoY Forecast"].to_numpy()
        future_pool["Unemp_1_month_prior _to_t+1"] = grouped["Unemployment Rate YoY QUARTER"].to_numpy()
        future_pool["Rfsh_Fcst_BoC_1_month_prior_to_t+1"] = grouped["Overnight Rate"].to_numpy()
        future_pool["Sent_BART_t"] = _predict_sentiment(model_cache["sentiment_bundle"], horizon)
        future_pool["M2_BART_x_GDP"] = future_pool["Sent_BART_t"] * grouped["GDP YoY Forecast"].to_numpy()
        future_pool["M2_BART_x_Unemp"] = future_pool["Sent_BART_t"] * grouped["Unemployment Rate YoY QUARTER"].to_numpy()
        future_pool["M2_BART_x_BoC"] = future_pool["Sent_BART_t"] * grouped["Overnight Rate"].to_numpy()

    future_pool["PCL_t"] = np.nan

    selected_features = model_cache["selected_features"]
    best_ml_name = model_cache["best_ml_name"]
    global_best_name = model_cache["global_best_name"]
    scaler = model_cache["scaler"]
    champion_model = model_cache["champion_model"]
    best_rmse = model_cache["best_rmse"]
    current_pcl = model_cache["last_known_pcl"]
    pcl_base_ml: list[float] = []

    for row_index in range(horizon):
        future_pool.loc[row_index, "PCL_t"] = current_pcl
        row = future_pool.iloc[row_index : row_index + 1][selected_features]
        prediction = float(champion_model.predict(scaler.transform(row))[0])
        pcl_base_ml.append(prediction)
        current_pcl = prediction

    base_values = np.asarray(pcl_base_ml, dtype=float)
    covid_shock, recovery_release, full_playback_curve = model_cache["playback"]
    playback_curve = full_playback_curve[:horizon]
    bear_values = base_values + covid_shock * playback_curve
    bull_values = base_values - recovery_release * playback_curve
    lower_values = base_values - 1.96 * best_rmse
    upper_values = base_values + 1.96 * best_rmse

    ts_values: np.ndarray | None = None
    if model_cache["ts_bundle"] is not None:
        ts_bundle = model_cache["ts_bundle"]
        try:
            if ts_bundle["type"] == "varmax":
                ts_values = ts_bundle["model"].forecast(
                    steps=horizon,
                    exog=future_pool[ts_bundle["exog_features"]],
                ).iloc[:, 0].to_numpy()
            elif ts_bundle["type"] == "var":
                ts_values = ts_bundle["model"].forecast(ts_bundle["last_state"], steps=horizon)[:, 0]
            elif ts_bundle["type"] == "arimax":
                ts_values = np.asarray(
                    ts_bundle["model"].forecast(
                        steps=horizon,
                        exog=future_pool[ts_bundle["exog_features"]].to_numpy(),
                    ),
                    dtype=float,
                )
            elif ts_bundle["type"] == "sarima":
                ts_values = np.asarray(ts_bundle["model"].forecast(steps=horizon), dtype=float)
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
    future_pool["Sent_BART_t"] = _predict_sentiment(ctx["sentiment_bundle"], horizon)
    future_pool["BART_x_CloseIn_Unemp"] = future_pool["Sent_BART_t"] * future_pool["Unemp_1_month_prior _to_t+1"]
    future_pool["BART_x_CloseIn_BoC"] = future_pool["Sent_BART_t"] * future_pool["Rfsh_Fcst_BoC_1_month_prior_to_t+1"]
    future_pool["BART_x_CloseIn_GDP"] = future_pool["Sent_BART_t"] * future_pool["CloseIn_GDP"]

    selected_features = ctx["selected_features"]
    scaler = ctx["scaler"]
    champion_model = ctx["champion_model"]

    bbg_base = grouped["Bloomberg"].to_numpy(dtype=float)
    residual_delta = champion_model.predict(scaler.transform(future_pool[selected_features]))
    base_values = bbg_base + residual_delta

    covid_shock, recovery_release, full_playback_curve = ctx["playback"]
    playback_curve = full_playback_curve[:horizon]
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


def _run_all_forecasts_internal(clean_df: pd.DataFrame) -> dict[str, Any]:
    aggregate_ctx = load_aggregate_context()
    bloomberg_ctx = load_bloomberg_context()
    individual_no_sentiment_ctx = load_individual_no_sentiment_context()
    individual_with_sentiment_ctx = load_individual_with_sentiment_context()

    aggregate_model_1 = _forecast_aggregate_model(clean_df, 1, aggregate_ctx)
    aggregate_model_2 = _forecast_aggregate_model(clean_df, 2, aggregate_ctx)
    bloomberg_model = _forecast_bloomberg_model(clean_df, bloomberg_ctx)
    individual_no_sentiment = _forecast_individual_variant(clean_df, individual_no_sentiment_ctx)
    individual_with_sentiment = _forecast_individual_variant(clean_df, individual_with_sentiment_ctx)

    combined_json = {
        "input_scenarios": clean_df.to_dict(orient="records"),
        "aggregate_model_1": aggregate_model_1["payload"],
        "aggregate_model_2": aggregate_model_2["payload"],
        "bloomberg_model": bloomberg_model["payload"],
        "individual_no_sentiment": individual_no_sentiment["payload"],
        "individual_with_sentiment": individual_with_sentiment["payload"],
    }

    return {
        "scenario_df": clean_df,
        "aggregate_model_1": aggregate_model_1,
        "aggregate_model_2": aggregate_model_2,
        "bloomberg_model": bloomberg_model,
        "individual_no_sentiment": individual_no_sentiment,
        "individual_with_sentiment": individual_with_sentiment,
        "json_payload": combined_json,
    }


@lru_cache(maxsize=32)
def _run_all_forecasts_cached(scenario_csv: str) -> dict[str, Any]:
    clean_df = pd.read_csv(io.StringIO(scenario_csv))
    return _run_all_forecasts_internal(clean_df)


def run_all_forecasts(scenario_df: pd.DataFrame) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=NOT_OPENSSL_WARNING)
    clean_df = _clean_scenario_df(scenario_df)
    scenario_csv = clean_df.to_csv(index=False)
    return _run_all_forecasts_cached(scenario_csv)


def results_excel_bytes(results: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results["scenario_df"].to_excel(writer, index=False, sheet_name="Scenario Input")
        results["aggregate_model_1"]["result_df"].to_excel(writer, index=False, sheet_name="Aggregate Model 1")
        results["aggregate_model_2"]["result_df"].to_excel(writer, index=False, sheet_name="Aggregate Model 2")
        results["bloomberg_model"]["result_df"].to_excel(writer, index=False, sheet_name="Bloomberg Model")
        results["individual_no_sentiment"]["winner_summary_df"].to_excel(
            writer, index=False, sheet_name="Ind No Sent Summary"
        )
        results["individual_no_sentiment"]["combined_result_df"].to_excel(
            writer, index=False, sheet_name="Ind No Sent Combined"
        )
        results["individual_with_sentiment"]["winner_summary_df"].to_excel(
            writer, index=False, sheet_name="Ind Sent Summary"
        )
        results["individual_with_sentiment"]["combined_result_df"].to_excel(
            writer, index=False, sheet_name="Ind Sent Combined"
        )
        for bank, bundle in results["individual_no_sentiment"]["per_bank"].items():
            bundle["result_df"].to_excel(writer, index=False, sheet_name=f"NS {bank}"[:31])
        for bank, bundle in results["individual_with_sentiment"]["per_bank"].items():
            bundle["result_df"].to_excel(writer, index=False, sheet_name=f"SENT {bank}"[:31])
    return output.getvalue()


def executive_summary(results: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    latest_q = results["aggregate_model_1"]["result_df"]["Quarter"].iloc[-1]

    for key, label in [
        ("aggregate_model_1", "Aggregate Model 1"),
        ("aggregate_model_2", "Aggregate Model 2"),
        ("bloomberg_model", "Bloomberg Model"),
    ]:
        row = results[key]["result_df"].loc[results[key]["result_df"]["Quarter"] == latest_q].iloc[0]
        summaries.append(
            f"{label} projects {latest_q} base PCL at {row['ML Base']:,.1f}, "
            f"with a bear case of {row['Bear']:,.1f} and a bull case of {row['Bull']:,.1f}."
        )

    for key, label in [
        ("individual_no_sentiment", "Individual model without sentiment"),
        ("individual_with_sentiment", "Individual model with sentiment"),
    ]:
        latest_df = results[key]["latest_quarter_summary_df"]
        if latest_df.empty:
            summaries.append(f"{label} produced no eligible bank forecasts for the uploaded scenario.")
            continue
        top_row = latest_df.sort_values("ML Base", ascending=False).iloc[0]
        summaries.append(
            f"{label} shows the highest latest-quarter base forecast for {top_row['Bank']} at "
            f"{top_row['ML Base']:,.1f}."
        )

    return summaries


def warmup_model_cache() -> None:
    load_aggregate_context()
    load_bloomberg_context()
    load_individual_no_sentiment_context()
    load_individual_with_sentiment_context()
