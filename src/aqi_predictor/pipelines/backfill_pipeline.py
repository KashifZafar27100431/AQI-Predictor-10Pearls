from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.features import build_feature_frame, pollution_payload_to_frame
from aqi_predictor.services.openweather import OpenWeatherClient
from aqi_predictor.services.sample_data import generate_sample_history
from aqi_predictor.services.storage import (
    FEATURES_FEATURE_GROUP,
    RAW_AIR_FEATURE_GROUP,
    FeatureStoreClient,
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def run(days: int = 60, sample: bool = False) -> dict:
    _load_dotenv()
    settings = get_settings()
    feature_store = FeatureStoreClient(settings)

    if sample or settings.use_sample_data:
        features = generate_sample_history(settings, hours=days * 24)
        raw_air = features.copy()
    else:
        client = OpenWeatherClient(settings)
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=days)
        payload = client.historical_air_pollution(start, end)
        raw_air = pollution_payload_to_frame(payload, settings.city, settings.lat, settings.lon)
        features = build_feature_frame(raw_air, pd.DataFrame())

    feature_store.insert_feature_group(RAW_AIR_FEATURE_GROUP, raw_air, primary_key=["city", "event_time"])
    feature_store.insert_feature_group(FEATURES_FEATURE_GROUP, features, primary_key=["city", "event_time"])
    return {"status": "ok", "city": settings.city, "days": days, "rows_inserted": int(len(features))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features.")
    parser.add_argument("--days", type=int, default=60, help="Number of historical days to backfill.")
    parser.add_argument("--sample", action="store_true", help="Use deterministic sample data.")
    args = parser.parse_args()
    print(json.dumps(run(days=args.days, sample=args.sample), indent=2))


if __name__ == "__main__":
    main()

