from __future__ import annotations

import argparse
import json

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.modeling import train_and_register, write_training_report
from aqi_predictor.services.sample_data import generate_sample_history
from aqi_predictor.services.storage import FEATURES_FEATURE_GROUP, FeatureStoreClient


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def run(sample_if_empty: bool = True) -> dict:
    _load_dotenv()
    settings = get_settings()
    feature_store = FeatureStoreClient(settings)
    features = feature_store.read_feature_group(FEATURES_FEATURE_GROUP)
    if features.empty and sample_if_empty:
        features = generate_sample_history(settings, hours=24 * 60)
        feature_store.insert_feature_group(FEATURES_FEATURE_GROUP, features, primary_key=["city", "event_time"])
    metadata = train_and_register(features, settings)
    report_path = write_training_report(settings, metadata)
    return {"status": "ok", "report": str(report_path), "model": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the AQI forecasting model.")
    parser.add_argument(
        "--no-sample-if-empty",
        action="store_true",
        help="Fail instead of generating sample data when the feature store is empty.",
    )
    args = parser.parse_args()
    print(json.dumps(run(sample_if_empty=not args.no_sample_if_empty), indent=2))


if __name__ == "__main__":
    main()

