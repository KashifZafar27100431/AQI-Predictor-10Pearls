from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.aqi_scale import label_from_score, openweather_level_from_score
from aqi_predictor.services.features import build_feature_frame


def _sample_raw_frame(settings: Settings, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    unix_seconds = (timestamps.astype("int64") // 10**9).astype(int)
    absolute_hours = np.asarray(unix_seconds / 3600.0, dtype=float)

    daily_cycle = np.sin(2 * np.pi * (timestamps.hour.to_numpy() / 24.0))
    weekly_cycle = np.cos(2 * np.pi * (timestamps.dayofweek.to_numpy() / 7.0))
    seasonal_cycle = np.sin(2 * np.pi * (absolute_hours / (24.0 * 45.0)))
    deterministic_noise = 6 * np.sin(absolute_hours * 0.73)

    pm2_5 = np.clip(42 + 18 * daily_cycle + 8 * weekly_cycle + 8 * seasonal_cycle + deterministic_noise, 4, 180)
    pm10 = np.clip(pm2_5 * 1.8 + 12 * np.cos(absolute_hours * 0.19), 8, 320)
    no2 = np.clip(28 + 10 * daily_cycle + 3 * np.sin(absolute_hours * 0.37), 1, 160)
    o3 = np.clip(55 - 12 * daily_cycle + 5 * np.cos(absolute_hours * 0.11), 4, 190)
    so2 = np.clip(12 + 5 * weekly_cycle + 2 * np.sin(absolute_hours * 0.17), 1, 90)
    co = np.clip(520 + pm2_5 * 18 + 50 * np.sin(absolute_hours * 0.07), 200, 6000)
    no = np.clip(no2 * 0.18, 0.0, 60)
    nh3 = np.clip(8 + 2 * np.sin(absolute_hours * 0.21), 0.1, 50)

    temp = 29 + 4 * np.sin(2 * np.pi * ((timestamps.hour.to_numpy() - 14) / 24.0))
    humidity = np.clip(70 - 0.8 * (temp - 29) + 5 * np.sin(absolute_hours * 0.05), 30, 95)
    pressure = 1008 + 3 * np.cos(absolute_hours * 0.04)
    wind_speed = np.clip(3.5 + 1.5 * np.sin(absolute_hours * 0.09), 0.2, 9)
    wind_deg = (210 + 35 * np.sin(absolute_hours * 0.03)) % 360
    clouds = np.clip(45 + 25 * np.cos(absolute_hours * 0.06), 0, 100)

    aqi_score = np.clip(0.72 * pm2_5 + 0.22 * pm10 + 0.1 * no2 + 20, 20, 260)
    ow_aqi = [openweather_level_from_score(value) for value in aqi_score]

    return pd.DataFrame(
        {
            "city": settings.city,
            "lat": settings.lat,
            "lon": settings.lon,
            "event_time": timestamps,
            "unix_ts": unix_seconds,
            "ow_aqi": ow_aqi,
            "aqi_score": aqi_score,
            "aqi_category": [label_from_score(value) for value in aqi_score],
            "primary_pollutant": "pm2_5",
            "co": co,
            "no": no,
            "no2": no2,
            "o3": o3,
            "so2": so2,
            "pm2_5": pm2_5,
            "pm10": pm10,
            "nh3": nh3,
            "temp": temp,
            "feels_like": temp + 2,
            "pressure": pressure,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_deg": wind_deg,
            "clouds": clouds,
        }
    )


def generate_sample_history(
    settings: Settings,
    hours: int = 24 * 60,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    end = end_time or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours - 1)
    timestamps = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    raw = _sample_raw_frame(settings, timestamps)
    return build_feature_frame(raw, None)


def generate_sample_forecast(settings: Settings, hours: int = 72) -> pd.DataFrame:
    history = generate_sample_history(settings, hours=24 * 14)
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    timestamps = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    forecast = build_feature_frame(_sample_raw_frame(settings, timestamps), None)
    forecast["forecast_horizon"] = np.arange(1, hours + 1)
    forecast["aqi_lag_1h"] = float(history["aqi_score"].iloc[-1])
    forecast["aqi_rolling_3h"] = float(history["aqi_score"].tail(3).mean())
    forecast["aqi_rolling_24h"] = float(history["aqi_score"].tail(24).mean())
    forecast["aqi_change_rate"] = float(history["aqi_score"].iloc[-1] - history["aqi_score"].iloc[-2])
    return forecast
