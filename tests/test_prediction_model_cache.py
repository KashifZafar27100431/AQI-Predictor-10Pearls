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


class CountingFeatureStore(FakeFeatureStore):
    def __init__(self):
        self.insert_count = 0

    def insert_feature_group(self, name: str, frame: pd.DataFrame, **kwargs: Any) -> None:
        self.insert_count += 1


class FakeMongoStore:
    def __init__(self):
        self.insert_count = 0

    def insert_predictions(self, records: list[dict]) -> None:
        self.insert_count += 1

    def list_alerts(self, limit: int = 20) -> list[dict]:
        return []


class FreshFeatureStore(FakeFeatureStore):
    def read_feature_group(self, name: str, limit: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_time": [pd.Timestamp.utcnow()],
                "aqi_score": [75.0],
                **{column: [0.0] for column in FEATURE_COLUMNS},
            }
        )


def _patch_constant_model(monkeypatch) -> None:
    def fake_load_model_bundle(settings: Settings):
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


def test_model_info_prefers_precomputed_explainability(monkeypatch):
    def fake_load_model_bundle(settings: Settings):
        return ConstantModel(), {
            "model_name": "constant",
            "trained_at": "2026-05-31T00:00:00Z",
            "metrics": {"rmse": 1.0, "mae": 1.0, "r2": 0.0},
            "serving_source": "hopsworks_model_registry",
            "registry_version": 5,
            "feature_columns": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
            "explainability": {
                "method": "shap",
                "status": "ok",
                "computed_at": "2026-05-31T00:00:00Z",
                "top_features": [{"feature": "pm2_5", "importance": 1.25}],
            },
        }

    def fail_live_importance(*args, **kwargs):
        raise AssertionError("Live explainability should not run when precomputed SHAP exists.")

    monkeypatch.setattr(prediction_module, "load_model_bundle", fake_load_model_bundle)
    monkeypatch.setattr(prediction_module, "feature_importance", fail_live_importance)
    settings = Settings(use_sample_data=True, openweather_api_key=None, mongodb_uri=None)
    service = PredictionService(settings, feature_store=FakeFeatureStore(), mongo_store=FakeMongoStore())

    model_info = service.model_info_payload()

    assert model_info["explainability"]["method"] == "shap"
    assert model_info["feature_importance"] == [{"feature": "pm2_5", "importance": 1.25}]


def test_model_info_includes_feature_freshness(monkeypatch):
    _patch_constant_model(monkeypatch)
    settings = Settings(use_sample_data=True, openweather_api_key=None, mongodb_uri=None)
    service = PredictionService(settings, feature_store=FreshFeatureStore(), mongo_store=FakeMongoStore())

    model_info = service.model_info_payload()

    assert model_info["data_freshness"]["status"] == "fresh"
    assert model_info["data_freshness"]["latest_feature_age_minutes"] is not None
    assert model_info["data_freshness"]["latest_feature_event_time_local"] is not None


def test_prediction_can_skip_persistence_for_dashboard_reruns(monkeypatch):
    _patch_constant_model(monkeypatch)
    settings = Settings(use_sample_data=True, openweather_api_key=None, mongodb_uri=None)
    feature_store = CountingFeatureStore()
    mongo_store = FakeMongoStore()
    service = PredictionService(settings, feature_store=feature_store, mongo_store=mongo_store)

    payload = service.predict(horizon=3, sample=True, store_predictions=False)

    assert payload["prediction_store"] == "not_persisted"
    assert feature_store.insert_count == 0
    assert mongo_store.insert_count == 0


def test_alerts_payload_falls_back_to_current_forecast_for_usg_alerts(monkeypatch):
    class SensitiveGroupModel(ConstantModel):
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.repeat(111.0, len(frame))

    def fake_load_model_bundle(settings: Settings):
        return SensitiveGroupModel(), {
            "model_name": "sensitive_group_model",
            "trained_at": "2026-05-31T00:00:00Z",
            "metrics": {"rmse": 1.0, "mae": 1.0, "r2": 0.0},
            "serving_source": "hopsworks_model_registry",
            "registry_version": 9,
            "feature_columns": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
        }

    monkeypatch.setattr(prediction_module, "load_model_bundle", fake_load_model_bundle)
    settings = Settings(
        forecast_hours=3,
        use_sample_data=True,
        openweather_api_key=None,
        mongodb_uri=None,
    )
    service = PredictionService(settings, feature_store=FakeFeatureStore(), mongo_store=FakeMongoStore())

    payload = service.alerts_payload(limit=10)

    assert payload["source"] == "current_forecast"
    assert len(payload["alerts"]) == 3
    assert {record["alert_level"] for record in payload["alerts"]} == {"sensitive_groups"}
    assert {record["aqi_category"] for record in payload["alerts"]} == {"Unhealthy for Sensitive Groups"}
