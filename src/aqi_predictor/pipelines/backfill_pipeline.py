from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.features import WEATHER_COLUMNS, build_feature_frame, pollution_payload_to_frame
from aqi_predictor.services.openweather import OpenWeatherClient
from aqi_predictor.services.sample_data import generate_sample_history
from aqi_predictor.services.storage import (
    FEATURES_FEATURE_GROUP,
    RAW_AIR_FEATURE_GROUP,
    RAW_WEATHER_FEATURE_GROUP,
    FeatureStoreClient,
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def _clean_hourly_frame(frame: pd.DataFrame, primary_key: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    clean["event_time"] = pd.to_datetime(clean["event_time"], utc=True).dt.floor("h")
    clean = clean.dropna(subset=["event_time"]).sort_values("event_time")
    key_columns = [column for column in primary_key if column in clean.columns]
    if key_columns:
        clean = clean.drop_duplicates(subset=key_columns, keep="last")
    return clean.reset_index(drop=True)


def _cached_weather_for_window(
    feature_store: FeatureStoreClient,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    cached = feature_store.read_feature_group(RAW_WEATHER_FEATURE_GROUP)
    if cached.empty or "event_time" not in cached.columns:
        return pd.DataFrame()
    cached["event_time"] = pd.to_datetime(cached["event_time"], utc=True)
    mask = (cached["event_time"] >= pd.Timestamp(start)) & (cached["event_time"] <= pd.Timestamp(end))
    return _clean_hourly_frame(cached.loc[mask], ["event_time"])


def _continuity_report(frame: pd.DataFrame, start: datetime, end: datetime) -> dict:
    if frame.empty:
        expected_hours = int((end - start).total_seconds() // 3600) + 1
        return {"expected_hours": expected_hours, "observed_hours": 0, "missing_hours": expected_hours}
    observed = pd.to_datetime(frame["event_time"], utc=True).dt.floor("h").drop_duplicates()
    expected = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(observed))
    return {
        "expected_hours": int(len(expected)),
        "observed_hours": int(len(observed)),
        "missing_hours": int(len(missing)),
    }


def _weather_coverage(features: pd.DataFrame) -> float:
    if features.empty:
        return 0.0
    available_columns = [column for column in WEATHER_COLUMNS if column in features.columns]
    if not available_columns:
        return 0.0
    return float(features[available_columns].notna().any(axis=1).mean())


def run(days: int = 60, sample: bool = False) -> dict:
    _load_dotenv()
    settings = get_settings()
    feature_store = FeatureStoreClient(settings)

    if sample or settings.use_sample_data:
        features = generate_sample_history(settings, hours=days * 24)
        raw_air = features.copy()
        raw_weather = features[["event_time", *WEATHER_COLUMNS]].copy()
        continuity = _continuity_report(features, features["event_time"].min(), features["event_time"].max())
        weather_strategy = "sample_weather"
    else:
        client = OpenWeatherClient(settings)
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=days)
        payload = client.historical_air_pollution(start, end)
        raw_air = pollution_payload_to_frame(payload, settings.city, settings.lat, settings.lon)
        raw_air = _clean_hourly_frame(raw_air, ["city", "event_time"])
        raw_weather = _cached_weather_for_window(feature_store, start, end)
        features = build_feature_frame(raw_air, raw_weather, timezone_name=settings.timezone)
        continuity = _continuity_report(features, start, end)
        coverage = _weather_coverage(features)
        weather_strategy = (
            "cached_hourly_weather"
            if coverage >= settings.max_weather_missing_fraction
            else "weather_sparse_training_excludes_high_missingness_columns"
        )

    feature_store.insert_feature_group(RAW_AIR_FEATURE_GROUP, raw_air, primary_key=["city", "event_time"])
    feature_store.insert_feature_group(RAW_WEATHER_FEATURE_GROUP, raw_weather, primary_key=["event_time"])
    feature_store.insert_feature_group(FEATURES_FEATURE_GROUP, features, primary_key=["city", "event_time"])
    return {
        "status": "ok",
        "city": settings.city,
        "days": days,
        "rows_inserted": int(len(features)),
        "weather_strategy": weather_strategy,
        "weather_coverage_fraction": _weather_coverage(features),
        "continuity": continuity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features.")
    parser.add_argument("--days", type=int, default=60, help="Number of historical days to backfill.")
    parser.add_argument("--sample", action="store_true", help="Use deterministic sample data.")
    args = parser.parse_args()
    print(json.dumps(run(days=args.days, sample=args.sample), indent=2))


if __name__ == "__main__":
    main()
