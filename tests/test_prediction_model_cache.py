from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.features import FEATURE_COLUMNS
import aqi_predictor.services.prediction as prediction_module
from aqi_predictor.services.prediction import PredictionService


class ConstantModel:
    n_features_in_ = len(FEATURE_COLUMNS)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.repeat(42.0, len(frame))


class FakeFeatureStore:
    def read_feature_group(self, name: str, limit: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame()

    def insert_feature_group(self, name: str, frame: pd.DataFrame, **kwargs: Any) -> None:
        return None


class FakeMongoStore:
    def insert_predictions(self, records: list[dict]) -> None:
        return None


def test_model_info_reuses_prediction_model_bundle(monkeypatch):
    calls = 0

    def fake_load_model_bundle(settings: Settings):
        nonlocal calls
        calls += 1
        return ConstantModel(), {
            "model_name": "constant",
            "trained_at": "2026-05-31T00:00:00Z",
            "metrics": {"rmse": 1.0, "mae": 1.0, "r2": 0.0},
            "serving_source": "hopsworks_model_registry",
            "registry_version": 5,
            "feature_columns": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
        }

    monkeypatch.setattr(prediction_module, "load_model_bundle", fake_load_model_bundle)
    settings = Settings(use_sample_data=True, openweather_api_key=None, mongodb_uri=None)
    service = PredictionService(settings, feature_store=FakeFeatureStore(), mongo_store=FakeMongoStore())

    payload = service.predict(horizon=3, sample=True)
    model_info = service.model_info_payload()

    assert payload["model"]["name"] == "constant"
    assert model_info["model"]["model_name"] == "constant"
    assert calls == 1
