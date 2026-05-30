from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from aqi_predictor.services.aqi_scale import (
    aqi_score_from_components,
    aqi_score_from_openweather_level,
    alert_level,
    label_from_score,
    openweather_level_from_score,
    primary_pollutant_from_components,
)


POLLUTANT_COLUMNS: List[str] = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
WEATHER_COLUMNS: List[str] = [
    "temp",
    "feels_like",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_deg",
    "clouds",
]
TIME_COLUMNS: List[str] = ["hour", "day", "month", "weekday", "is_weekend"]
LAG_COLUMNS: List[str] = ["aqi_lag_1h", "aqi_rolling_3h", "aqi_rolling_24h", "aqi_change_rate"]
FEATURE_COLUMNS: List[str] = (
    POLLUTANT_COLUMNS + WEATHER_COLUMNS + TIME_COLUMNS + LAG_COLUMNS + ["forecast_horizon"]
)
TARGET_COLUMN = "aqi_score"


def _utc_from_unix(unix_ts: int) -> pd.Timestamp:
    return pd.to_datetime(int(unix_ts), unit="s", utc=True)


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def pollution_payload_to_frame(
    payload: Dict[str, Any], city: str, lat: float, lon: float
) -> pd.DataFrame:
    rows = []
    for item in payload.get("list", []):
        main = item.get("main", {})
        components = item.get("components", {})
        ow_aqi = int(main.get("aqi", 0) or 0)
        numeric_components = {
            column: _safe_float(components.get(column)) for column in POLLUTANT_COLUMNS
        }
        component_aqi = aqi_score_from_components(numeric_components)
        aqi_score = (
            component_aqi
            if component_aqi is not None
            else aqi_score_from_openweather_level(ow_aqi)
        )
        row: Dict[str, Any] = {
            "city": city,
            "lat": float(lat),
            "lon": float(lon),
            "event_time": _utc_from_unix(item["dt"]),
            "unix_ts": int(item["dt"]),
            "ow_aqi": ow_aqi,
            "aqi_score": aqi_score,
            "aqi_category": label_from_score(aqi_score),
            "primary_pollutant": primary_pollutant_from_components(numeric_components),
        }
        for column in POLLUTANT_COLUMNS:
            row[column] = numeric_components[column]
        rows.append(row)
    return pd.DataFrame(rows)


def current_weather_payload_to_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    clouds = payload.get("clouds", {})
    row = {
        "event_time": _utc_from_unix(payload.get("dt", pd.Timestamp.utcnow().timestamp())),
        "temp": _safe_float(main.get("temp")),
        "feels_like": _safe_float(main.get("feels_like")),
        "pressure": _safe_float(main.get("pressure")),
        "humidity": _safe_float(main.get("humidity")),
        "wind_speed": _safe_float(wind.get("speed")),
        "wind_deg": _safe_float(wind.get("deg")),
        "clouds": _safe_float(clouds.get("all")),
    }
    return pd.DataFrame([row])


def forecast_weather_payload_to_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("list", []):
        main = item.get("main", {})
        wind = item.get("wind", {})
        clouds = item.get("clouds", {})
        rows.append(
            {
                "event_time": _utc_from_unix(item["dt"]),
                "temp": _safe_float(main.get("temp")),
                "feels_like": _safe_float(main.get("feels_like")),
                "pressure": _safe_float(main.get("pressure")),
                "humidity": _safe_float(main.get("humidity")),
                "wind_speed": _safe_float(wind.get("speed")),
                "wind_deg": _safe_float(wind.get("deg")),
                "clouds": _safe_float(clouds.get("all")),
            }
        )
    return pd.DataFrame(rows)


def merge_weather_features(pollution: pd.DataFrame, weather: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pollution.empty:
        return pollution
    frame = pollution.copy()
    if weather is None or weather.empty:
        for column in WEATHER_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame

    left = frame.sort_values("event_time")
    right = weather.sort_values("event_time")
    merged = pd.merge_asof(left, right, on="event_time", direction="nearest")
    for column in WEATHER_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    event_time = pd.to_datetime(result["event_time"], utc=True)
    result["event_time"] = event_time
    result["hour"] = event_time.dt.hour
    result["day"] = event_time.dt.day
    result["month"] = event_time.dt.month
    result["weekday"] = event_time.dt.weekday
    result["is_weekend"] = result["weekday"].isin([5, 6]).astype(int)
    return result


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.sort_values(["city", "event_time"]).copy()
    if TARGET_COLUMN not in result.columns:
        for column in LAG_COLUMNS:
            if column not in result.columns:
                result[column] = 0.0
        return result

    grouped = result.groupby("city", group_keys=False)
    lag = grouped[TARGET_COLUMN].shift(1)
    shifted = grouped[TARGET_COLUMN].shift(1)
    result["aqi_lag_1h"] = lag.fillna(result[TARGET_COLUMN])
    result["aqi_rolling_3h"] = shifted.groupby(result["city"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    result["aqi_rolling_24h"] = shifted.groupby(result["city"]).rolling(24, min_periods=1).mean().reset_index(level=0, drop=True)
    result["aqi_rolling_3h"] = result["aqi_rolling_3h"].fillna(result["aqi_lag_1h"])
    result["aqi_rolling_24h"] = result["aqi_rolling_24h"].fillna(result["aqi_lag_1h"])
    result["aqi_change_rate"] = (result[TARGET_COLUMN] - result["aqi_lag_1h"]).fillna(0.0)
    return result


def build_feature_frame(
    pollution: pd.DataFrame, weather: Optional[pd.DataFrame] = None, forecast_horizon: int = 0
) -> pd.DataFrame:
    frame = merge_weather_features(pollution, weather)
    frame = add_time_features(frame)
    frame = add_lag_features(frame)
    if "forecast_horizon" not in frame.columns:
        frame["forecast_horizon"] = forecast_horizon
    return normalize_numeric_features(frame)


def normalize_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = normalize_numeric_features(add_lag_features(add_time_features(frame)))
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    for column in required:
        if column not in result.columns:
            result[column] = np.nan
    result = result.dropna(subset=[TARGET_COLUMN])
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].fillna(result[FEATURE_COLUMNS].median(numeric_only=True))
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].fillna(0.0)
    return result


def ensure_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].fillna(0.0)
    return result[FEATURE_COLUMNS]


def recent_history_stats(history: pd.DataFrame) -> Dict[str, float]:
    if history is None or history.empty or TARGET_COLUMN not in history.columns:
        return {
            "aqi_lag_1h": 100.0,
            "aqi_rolling_3h": 100.0,
            "aqi_rolling_24h": 100.0,
            "aqi_change_rate": 0.0,
        }
    sorted_history = history.sort_values("event_time")
    values = pd.to_numeric(sorted_history[TARGET_COLUMN], errors="coerce").dropna()
    if values.empty:
        return recent_history_stats(pd.DataFrame())
    last = float(values.iloc[-1])
    previous = float(values.iloc[-2]) if len(values) > 1 else last
    return {
        "aqi_lag_1h": last,
        "aqi_rolling_3h": float(values.tail(3).mean()),
        "aqi_rolling_24h": float(values.tail(24).mean()),
        "aqi_change_rate": last - previous,
    }


def attach_prediction_metadata(frame: pd.DataFrame, predictions: Iterable[float]) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_aqi_score"] = [float(max(0.0, value)) for value in predictions]
    result["predicted_aqi_level"] = result["predicted_aqi_score"].apply(openweather_level_from_score)
    result["aqi_category"] = result["predicted_aqi_score"].apply(label_from_score)
    result["alert_level"] = result["predicted_aqi_score"].apply(alert_level)
    return result


def json_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return result.replace({np.nan: None}).to_dict(orient="records")
