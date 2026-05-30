from __future__ import annotations

import argparse
import json

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.features import (
    build_feature_frame,
    current_weather_payload_to_frame,
    json_records,
    pollution_payload_to_frame,
)
from aqi_predictor.services.openweather import OpenWeatherClient
from aqi_predictor.services.sample_data import generate_sample_history
from aqi_predictor.services.storage import (
    FEATURES_FEATURE_GROUP,
    RAW_AIR_FEATURE_GROUP,
    RAW_WEATHER_FEATURE_GROUP,
    FeatureStoreClient,
    MongoStore,
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def run(sample: bool = False) -> dict:
    _load_dotenv()
    settings = get_settings()
    feature_store = FeatureStoreClient(settings)
    mongo = MongoStore(settings)

    if sample or settings.use_sample_data:
        features = generate_sample_history(settings, hours=48).tail(1)
        raw_air = features[
            [
                "city",
                "lat",
                "lon",
                "event_time",
                "unix_ts",
                "ow_aqi",
                "aqi_score",
                "aqi_category",
                "primary_pollutant",
                "co",
                "no",
                "no2",
                "o3",
                "so2",
                "pm2_5",
                "pm10",
                "nh3",
            ]
        ].copy()
        raw_weather = features[
            ["event_time", "temp", "feels_like", "pressure", "humidity", "wind_speed", "wind_deg", "clouds"]
        ].copy()
    else:
        client = OpenWeatherClient(settings)
        raw_air = pollution_payload_to_frame(
            client.current_air_pollution(), settings.city, settings.lat, settings.lon
        )
        raw_weather = current_weather_payload_to_frame(client.current_weather())
        features = build_feature_frame(raw_air, raw_weather, timezone_name=settings.timezone)

    feature_store.insert_feature_group(RAW_AIR_FEATURE_GROUP, raw_air, primary_key=["city", "event_time"])
    feature_store.insert_feature_group(RAW_WEATHER_FEATURE_GROUP, raw_weather, primary_key=["event_time"])
    feature_store.insert_feature_group(FEATURES_FEATURE_GROUP, features, primary_key=["city", "event_time"])

    latest_records = json_records(features.tail(1))
    if latest_records:
        mongo.upsert_latest(latest_records[0])

    return {
        "status": "ok",
        "rows_inserted": int(len(features)),
        "city": settings.city,
        "latest": latest_records[0] if latest_records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hourly AQI feature pipeline.")
    parser.add_argument("--sample", action="store_true", help="Use deterministic sample data.")
    args = parser.parse_args()
    print(json.dumps(run(sample=args.sample), indent=2))


if __name__ == "__main__":
    main()
