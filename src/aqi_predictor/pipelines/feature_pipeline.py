from __future__ import annotations

import argparse
import json
import logging

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


logger = logging.getLogger(__name__)


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
    logger.info(
        "hourly_feature_pipeline_started city=%s timezone=%s sample=%s latest_only=true",
        settings.city,
        settings.timezone,
        sample or settings.use_sample_data,
    )

    if sample or settings.use_sample_data:
        logger.info("hourly_feature_pipeline_sample_data_started hours=48 selected_rows=1")
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
        logger.info("hourly_feature_pipeline_sample_data_success rows=%s", len(features))
    else:
        client = OpenWeatherClient(settings)
        logger.info("openweather_current_air_pollution_started city=%s", settings.city)
        raw_air = pollution_payload_to_frame(
            client.current_air_pollution(), settings.city, settings.lat, settings.lon
        )
        logger.info("openweather_current_air_pollution_success rows=%s", len(raw_air))
        logger.info("openweather_current_weather_started city=%s", settings.city)
        raw_weather = current_weather_payload_to_frame(client.current_weather())
        logger.info("openweather_current_weather_success rows=%s", len(raw_weather))
        logger.info("hourly_feature_build_started raw_air_rows=%s raw_weather_rows=%s", len(raw_air), len(raw_weather))
        features = build_feature_frame(raw_air, raw_weather, timezone_name=settings.timezone)
        logger.info("hourly_feature_build_success rows=%s", len(features))

    logger.info("feature_group_insert_sequence_started groups=3")
    feature_store.insert_feature_group(RAW_AIR_FEATURE_GROUP, raw_air, primary_key=["city", "event_time"])
    feature_store.insert_feature_group(RAW_WEATHER_FEATURE_GROUP, raw_weather, primary_key=["event_time"])
    feature_store.insert_feature_group(FEATURES_FEATURE_GROUP, features, primary_key=["city", "event_time"])
    logger.info("feature_group_insert_sequence_success groups=3 rows=%s", len(features))

    latest_records = json_records(features.tail(1))
    if latest_records:
        logger.info("mongodb_latest_upsert_started configured=%s", mongo.configured)
        mongo.upsert_latest(latest_records[0])
        logger.info("mongodb_latest_upsert_finished configured=%s", mongo.configured)

    logger.info("hourly_feature_pipeline_success city=%s rows_inserted=%s", settings.city, len(features))
    return {
        "status": "ok",
        "rows_inserted": int(len(features)),
        "city": settings.city,
        "latest": latest_records[0] if latest_records else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run the hourly AQI feature pipeline.")
    parser.add_argument("--sample", action="store_true", help="Use deterministic sample data.")
    args = parser.parse_args()
    try:
        print(json.dumps(run(sample=args.sample), indent=2))
    except Exception:
        logger.exception("hourly_feature_pipeline_failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
