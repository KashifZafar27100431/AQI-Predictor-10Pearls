from __future__ import annotations

from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.aqi_scale import color_from_score, label_from_score
from aqi_predictor.services.modeling import ModelLoadError
from aqi_predictor.services.prediction import PredictionService
from aqi_predictor.services.storage import MODEL_ARTIFACT_FILE, MODEL_METADATA_FILE


logger = logging.getLogger(__name__)

SECRET_ENV_KEYS = (
    "OPENWEATHER_API_KEY",
    "HOPSWORKS_API_KEY",
    "HOPSWORKS_PROJECT",
    "MONGODB_URI",
    "MONGODB_DATABASE",
    "AQI_CITY",
    "AQI_LAT",
    "AQI_LON",
    "AQI_TIMEZONE",
    "AQI_FORECAST_HOURS",
    "AQI_MAX_FORECAST_HOURS",
    "AQI_ALLOW_LOCAL_MODEL_FALLBACK",
    "AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY",
    "AQI_USE_SAMPLE_DATA",
    "API_BASE_URL",
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        return


def _set_env_from_secret(key: str, value: object) -> None:
    if os.getenv(key) or value is None or isinstance(value, (dict, list, tuple, set)):
        return
    os.environ[key] = str(value)


def _load_streamlit_secrets() -> None:
    for key in SECRET_ENV_KEYS:
        try:
            value = st.secrets.get(key)
        except Exception:
            continue
        _set_env_from_secret(key, value)

    for section_name in ("env", "environment"):
        try:
            section = st.secrets.get(section_name, {})
        except Exception:
            continue
        if not hasattr(section, "items"):
            continue
        for key, value in section.items():
            if key in SECRET_ENV_KEYS:
                _set_env_from_secret(key, value)


@st.cache_data(ttl=300)
def _api_get(base_url: str, path: str, params: Optional[dict[str, Any]] = None) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}{path}", params=params, timeout=90)
    payload = response.json()
    if response.status_code >= 400:
        message = payload.get("message") if isinstance(payload, dict) else None
        raise RuntimeError(message or "API request failed.")
    return payload


@st.cache_data(ttl=600)
def _load_report_explainability() -> dict:
    path = ROOT / "reports" / "shap_summary.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        logger.exception("Failed to load precomputed explainability report.")
        return {}
    return payload if isinstance(payload, dict) else {}


def _show_model_load_diagnostics(exc: ModelLoadError, settings) -> None:
    logger.exception("AQI model loading failed.")
    st.error("The forecasting model is temporarily unavailable.")
    st.info("The app is configured for Hopsworks Model Registry serving. Check deployment secrets and model registry logs.")

    model_dir = settings.model_dir
    diagnostics = {
        "hopsworks_api_key_configured": bool(settings.hopsworks_api_key),
        "hopsworks_project_configured": bool(settings.hopsworks_project),
        "allow_local_model_fallback": settings.allow_local_model_fallback,
        "model_dir": str(model_dir),
        "local_metadata_exists": (model_dir / MODEL_METADATA_FILE).exists(),
        "local_model_exists": (model_dir / MODEL_ARTIFACT_FILE).exists(),
    }
    with st.expander("Deployment diagnostics"):
        st.json(diagnostics)


def _forecast_chart(frame: pd.DataFrame) -> go.Figure:
    x_column = "display_time" if "display_time" in frame.columns else "event_time"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame[x_column],
            y=frame["predicted_aqi_score"],
            mode="lines+markers",
            line={"color": "#2563eb", "width": 3},
            marker={"size": 5},
            name="Predicted AQI",
        )
    )
    bands = [
        (0, 50, "#dcfce7"),
        (50, 100, "#fef9c3"),
        (100, 150, "#ffedd5"),
        (150, 200, "#fee2e2"),
        (200, 300, "#f3e8ff"),
        (300, 500, "#fee2e2"),
    ]
    for low, high, color in bands:
        fig.add_hrect(y0=low, y1=high, line_width=0, fillcolor=color, opacity=0.35)
    fig.update_layout(
        height=430,
        margin={"l": 24, "r": 24, "t": 20, "b": 24},
        xaxis_title=None,
        yaxis_title="AQI score",
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def _daily_summary(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    time_column = "display_time" if "display_time" in daily.columns else "event_time"
    daily["date"] = pd.to_datetime(daily[time_column]).dt.date
    grouped = daily.groupby("date")["predicted_aqi_score"].agg(["min", "mean", "max"]).reset_index()
    grouped["category"] = grouped["max"].apply(label_from_score)
    return grouped


def _latest_observed_record(service: Optional[PredictionService] = None, api_base_url: Optional[str] = None) -> dict:
    try:
        if api_base_url:
            return _api_get(api_base_url, "/latest").get("latest") or {}
        if service is not None:
            return service.latest_payload().get("latest") or {}
    except Exception:
        logger.exception("Latest AQI lookup failed.")
    return {}


def _model_value(model: dict, primary: str, fallback: str = "", default: object = "unknown") -> object:
    value = model.get(primary)
    if value is None and fallback:
        value = model.get(fallback)
    return default if value is None else value


def _apply_report_explainability_fallback(model_info: dict) -> dict:
    explainability = model_info.get("explainability") or {}
    if isinstance(explainability, dict) and explainability.get("top_features"):
        return model_info
    report = _load_report_explainability()
    if not report.get("top_features"):
        return model_info
    updated = dict(model_info)
    updated["explainability"] = report
    updated["feature_importance"] = report.get("top_features", [])
    return updated


def _alert_label(level: str) -> str:
    return {
        "sensitive_groups": "Sensitive group alert",
        "unhealthy": "Health alert",
        "very_unhealthy": "Serious health alert",
        "hazardous": "Emergency hazardous alert",
    }.get(level, "No alert")


def _alert_severity(level: str) -> int:
    return {"sensitive_groups": 1, "unhealthy": 2, "very_unhealthy": 3, "hazardous": 4}.get(level, 0)


def _render_model_section(
    payload: dict,
    service: Optional[PredictionService] = None,
    api_base_url: Optional[str] = None,
) -> None:
    st.subheader("Model")
    model_summary = payload.get("model", {})
    model_info = {
        "model": {
            "metrics": model_summary.get("metrics", {}),
            "model_name": model_summary.get("name"),
            "serving_source": model_summary.get("serving_source"),
            "registry_version": model_summary.get("registry_version"),
            "trained_at": model_summary.get("trained_at"),
        },
        "data_freshness": {},
        "feature_importance": [],
        "explainability": {},
    }
    try:
        if api_base_url:
            model_info = _api_get(api_base_url, "/model-info")
        elif service is not None:
            model_info = service.model_info_payload()
    except Exception:
        logger.exception("AQI model explanation failed.")
        st.info("Feature importance is temporarily unavailable; prediction-time model metadata is shown below.")
    model_info = _apply_report_explainability_fallback(model_info)

    model = model_info.get("model", {})
    metrics = model.get("metrics", {})
    metric_cols = st.columns(5)
    metric_cols[0].metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
    metric_cols[1].metric("MAE", f"{metrics.get('mae', 0):.2f}")
    metric_cols[2].metric("R²", f"{metrics.get('r2', 0):.2f}")
    metric_cols[3].metric("Name", str(_model_value(model, "model_name", "name")))
    metric_cols[4].metric("Source", str(_model_value(model, "serving_source")))
    freshness = model_info.get("data_freshness", {})
    st.caption(
        f"Model trained: {_model_value(model, 'trained_at')} | "
        f"Registry version: {_model_value(model, 'registry_version', default='local')} | "
        f"Latest feature: {freshness.get('latest_feature_event_time_local') or 'unknown'} | "
        f"Freshness: {freshness.get('status', 'unknown')}"
    )

    importance = pd.DataFrame(model_info.get("feature_importance", []))
    if not importance.empty:
        explainability = model_info.get("explainability", {})
        method = explainability.get("method", "model_native_feature_importance")
        status = explainability.get("status", "unknown")
        computed_at = explainability.get("computed_at", "unknown")
        st.caption(f"Explainability: {method} ({status}), precomputed at {computed_at}")
        st.bar_chart(importance.set_index("feature")["importance"])


def main() -> None:
    _load_dotenv()
    st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
    _load_streamlit_secrets()
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; }
        [data-testid="stMetricValue"] { font-size: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    settings = get_settings()
    if settings.openweather_api_key and settings.use_sample_data:
        settings = replace(settings, use_sample_data=False)
    api_base_url = os.getenv("API_BASE_URL", "").strip().rstrip("/") or None
    service: Optional[PredictionService] = None if api_base_url else PredictionService(settings)

    st.title("Pearls AQI Predictor")
    st.caption(f"{settings.city} forecast")

    with st.sidebar:
        horizon = st.slider(
            "Forecast hours",
            min_value=24,
            max_value=settings.max_forecast_hours,
            value=min(settings.forecast_hours, settings.max_forecast_hours),
            step=12,
        )
        if api_base_url:
            sample = False
            st.caption("API-backed live mode")
        elif service and service.openweather.configured:
            sample = False
            st.caption("Live OpenWeather mode")
        else:
            sample = st.toggle("Sample mode", value=True)
        st.button("Refresh forecast", type="primary", width="stretch")

    try:
        if api_base_url:
            payload = _api_get(api_base_url, "/predict", params={"horizon": horizon})
        elif service is not None:
            payload = service.predict(horizon=horizon, sample=sample, store_predictions=False)
        else:
            raise RuntimeError("Prediction service is not configured.")
    except ModelLoadError as exc:
        _show_model_load_diagnostics(exc, settings)
        st.stop()
    except Exception:
        logger.exception("AQI prediction failed.")
        st.error("AQI prediction is temporarily unavailable. Please retry after the deployment is healthy.")
        st.stop()

    predictions = pd.DataFrame(payload["predictions"])
    if predictions.empty:
        st.warning("No predictions available.")
        st.stop()
    predictions["event_time"] = pd.to_datetime(predictions["event_time"], utc=True)
    if "event_time_local" in predictions.columns:
        predictions["display_time"] = pd.to_datetime(predictions["event_time_local"])
    else:
        predictions["display_time"] = predictions["event_time"]

    next_hour = predictions.iloc[0]
    current = _latest_observed_record(service=service, api_base_url=api_base_url)
    current_score = current.get("aqi_score")
    color_score = float(current_score) if current_score is not None else float(next_hour["predicted_aqi_score"])
    color = color_from_score(color_score)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current AQI", "N/A" if current_score is None else f"{float(current_score):.0f}")
    col2.metric("Next hour AQI", f"{next_hour['predicted_aqi_score']:.0f}")
    col3.metric("Category", current.get("aqi_category") or next_hour["aqi_category"])
    col4.metric("Primary pollutant", str(current.get("primary_pollutant") or next_hour.get("primary_pollutant", "unknown")).upper())
    col5.metric("Worst forecast AQI", f"{predictions['predicted_aqi_score'].max():.0f}")

    st.markdown(
        f"<div style='height:6px;background:{color};border-radius:3px;margin:4px 0 16px 0;'></div>",
        unsafe_allow_html=True,
    )
    if payload.get("is_live"):
        st.success("Live OpenWeather inputs")
    else:
        st.warning("Sample/offline inputs")
    st.plotly_chart(_forecast_chart(predictions), width="stretch")

    summary = _daily_summary(predictions)
    day_cols = st.columns(min(3, len(summary)))
    for idx, (_, row) in enumerate(summary.head(3).iterrows()):
        with day_cols[idx]:
            st.metric(str(row["date"]), f"{row['mean']:.0f}", f"max {row['max']:.0f} · {row['category']}")

    left, right = st.columns([2, 1])
    with left:
        pollutant_columns = [column for column in ["pm2_5", "pm10", "no2", "o3", "so2", "co"] if column in predictions]
        if pollutant_columns:
            st.line_chart(predictions.set_index("event_time")[pollutant_columns])
    with right:
        alerts = predictions[~predictions["alert_level"].isin(["normal"])]
        if alerts.empty:
            st.success("No unhealthy forecast hours.")
        else:
            worst_level = max(alerts["alert_level"], key=_alert_severity)
            worst_alert = _alert_label(worst_level)
            st.error(f"{len(alerts)} forecast alert hours. Highest: {worst_alert}.")
            display_alerts = alerts.copy()
            display_alerts["alert"] = display_alerts["alert_level"].map(_alert_label)
            st.dataframe(
                display_alerts[
                    ["event_time_local", "predicted_aqi_score", "aqi_category", "alert", "primary_pollutant"]
                ].head(8),
                width="stretch",
                hide_index=True,
            )

    _render_model_section(payload, service=service, api_base_url=api_base_url)


if __name__ == "__main__":
    main()
