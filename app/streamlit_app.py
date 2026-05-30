from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.aqi_scale import color_from_score, label_from_score
from aqi_predictor.services.prediction import PredictionService


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        return


def _forecast_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["event_time"],
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
    daily["date"] = pd.to_datetime(daily["event_time"]).dt.date
    grouped = daily.groupby("date")["predicted_aqi_score"].agg(["min", "mean", "max"]).reset_index()
    grouped["category"] = grouped["max"].apply(label_from_score)
    return grouped


def main() -> None:
    _load_dotenv()
    st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
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
    service = PredictionService(settings)

    st.title("Pearls AQI Predictor")
    st.caption(f"{settings.city} forecast")

    with st.sidebar:
        horizon = st.slider("Forecast hours", min_value=24, max_value=96, value=settings.forecast_hours, step=12)
        sample = st.toggle("Sample mode", value=settings.use_sample_data or not service.openweather.configured)
        st.button("Refresh forecast", type="primary", use_container_width=True)

    try:
        payload = service.predict(horizon=horizon, sample=sample)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    predictions = pd.DataFrame(payload["predictions"])
    if predictions.empty:
        st.warning("No predictions available.")
        st.stop()
    predictions["event_time"] = pd.to_datetime(predictions["event_time"])

    latest = predictions.iloc[0]
    color = color_from_score(float(latest["predicted_aqi_score"]))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Next hour AQI", f"{latest['predicted_aqi_score']:.0f}")
    col2.metric("Category", latest["aqi_category"])
    col3.metric("Primary pollutant", str(latest.get("primary_pollutant", "unknown")).upper())
    col4.metric("Worst forecast AQI", f"{predictions['predicted_aqi_score'].max():.0f}")

    st.markdown(
        f"<div style='height:6px;background:{color};border-radius:3px;margin:4px 0 16px 0;'></div>",
        unsafe_allow_html=True,
    )
    if payload.get("is_live"):
        st.success("Live OpenWeather inputs")
    else:
        st.warning("Sample/offline inputs")
    st.plotly_chart(_forecast_chart(predictions), use_container_width=True)

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
        unhealthy = predictions[predictions["alert_level"].isin(["unhealthy", "hazardous"])]
        if unhealthy.empty:
            st.success("No unhealthy forecast hours.")
        else:
            st.error(f"{len(unhealthy)} unhealthy forecast hours.")
            st.dataframe(
                unhealthy[["event_time", "predicted_aqi_score", "aqi_category", "primary_pollutant"]].head(8),
                use_container_width=True,
                hide_index=True,
            )

    try:
        model_info = service.model_info_payload()
        metrics = model_info["model"].get("metrics", {})
        st.subheader("Model")
        metric_cols = st.columns(4)
        metric_cols[0].metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
        metric_cols[1].metric("MAE", f"{metrics.get('mae', 0):.2f}")
        metric_cols[2].metric("R²", f"{metrics.get('r2', 0):.2f}")
        metric_cols[3].metric("Name", model_info["model"].get("model_name", "unknown"))

        importance = pd.DataFrame(model_info.get("feature_importance", []))
        if not importance.empty:
            st.bar_chart(importance.set_index("feature")["importance"])
    except Exception:
        pass


if __name__ == "__main__":
    main()
