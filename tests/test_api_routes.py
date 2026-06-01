from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import app.flask_api as flask_api


class FakePredictionService:
    def __init__(self, settings):
        self.settings = settings
        self.openweather = SimpleNamespace(configured=True)
        self.last_limit = None
        self.last_store_predictions = None

    def latest_payload(self):
        return {"city": self.settings.city, "latest": None}

    def predict(self, horizon=None, sample=False, store_predictions=True):
        self.last_store_predictions = store_predictions
        return {
            "city": self.settings.city,
            "horizon_hours": horizon,
            "sample": sample,
            "store_predictions": store_predictions,
            "predictions": [],
        }

    def alerts_payload(self, limit=20):
        self.last_limit = limit
        return {"city": self.settings.city, "limit": limit, "alerts": []}

    def model_info_payload(self):
        return {
            "city": self.settings.city,
            "model": {"serving_source": "local_model_dir"},
            "data_freshness": {"timezone": self.settings.timezone},
            "feature_importance": [],
        }


def _client(monkeypatch):
    monkeypatch.setattr(flask_api, "PredictionService", FakePredictionService)
    monkeypatch.setenv("AQI_FORECAST_HOURS", "72")
    monkeypatch.setenv("AQI_MAX_FORECAST_HOURS", "72")
    monkeypatch.setenv("AQI_MAX_API_LIMIT", "500")
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://karachi-aqi-predictor-10pearls.streamlit.app,http://localhost:8501",
    )
    app = flask_api.create_app()
    return app.test_client()


def test_health_latest_alerts_and_model_info(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/").json["service"] == "Karachi AQI Predictor API"
    assert client.get("/health").status_code == 200
    assert client.get("/diagnostics").status_code == 200
    assert client.get("/latest").status_code == 200
    assert client.get("/alerts?limit=999").json["limit"] == 500
    assert client.get("/model-info").json["model"]["serving_source"] == "local_model_dir"


def test_predict_horizon_is_clamped_and_invalid_params_are_rejected(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/predict?horizon=999").json["horizon_hours"] == 72
    assert client.get("/predict?horizon=0").json["horizon_hours"] == 1
    assert client.get("/predict?horizon=72").json["store_predictions"] is False
    assert client.get("/predict?horizon=abc").status_code == 400
    assert client.get("/predict?sample=maybe").status_code == 400


def test_diagnostics_are_safe_and_include_runtime_dependency_checks(monkeypatch):
    monkeypatch.setattr(flask_api, "PredictionService", FakePredictionService)
    monkeypatch.setenv("OPENWEATHER_API_KEY", "secret-openweather")
    monkeypatch.setenv("HOPSWORKS_API_KEY", "secret-hopsworks")
    monkeypatch.setenv("HOPSWORKS_PROJECT", "project-name")
    monkeypatch.setenv("AQI_ALLOW_LOCAL_MODEL_FALLBACK", "false")
    monkeypatch.setenv("AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY", "true")

    response = flask_api.create_app().test_client().get("/diagnostics")
    payload = response.json

    assert response.status_code == 200
    assert payload["env"]["OPENWEATHER_API_KEY"] is True
    assert payload["env"]["HOPSWORKS_API_KEY"] is True
    assert "pyarrow" in payload["packages"]
    assert payload["configuration"]["local_model_fallback_enabled"] is False
    serialized = response.get_data(as_text=True)
    assert "secret-openweather" not in serialized
    assert "secret-hopsworks" not in serialized


def test_public_errors_hide_tracebacks_and_internal_messages(monkeypatch):
    class FailingPredictionService(FakePredictionService):
        def predict(self, horizon=None, sample=False, store_predictions=True):
            raise RuntimeError("Traceback: /private/path secret-token")

    monkeypatch.setattr(flask_api, "PredictionService", FailingPredictionService)
    client = flask_api.create_app().test_client()

    response = client.get("/predict?horizon=72")
    body = response.get_data(as_text=True)

    assert response.status_code == 503
    assert response.json["message"] == "Prediction service is temporarily unavailable."
    assert "Traceback" not in body
    assert "secret-token" not in body


def test_cors_uses_configured_origins(monkeypatch):
    client = _client(monkeypatch)

    allowed = client.get(
        "/health",
        headers={"Origin": "https://karachi-aqi-predictor-10pearls.streamlit.app"},
    )
    denied = client.get("/health", headers={"Origin": "https://evil.example"})

    assert (
        allowed.headers.get("Access-Control-Allow-Origin")
        == "https://karachi-aqi-predictor-10pearls.streamlit.app"
    )
    assert denied.headers.get("Access-Control-Allow-Origin") is None


def test_wildcard_cors_origin_is_ignored(monkeypatch):
    monkeypatch.setattr(flask_api, "PredictionService", FakePredictionService)
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    app = flask_api.create_app()
    client = app.test_client()

    response = client.get("/health", headers={"Origin": "https://evil.example"})

    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") is None


def test_module_and_vercel_entrypoints_expose_flask_app():
    import api.index as vercel_entrypoint

    assert flask_api.app.name == flask_api.__name__
    assert vercel_entrypoint.app is flask_api.app


def test_local_model_fallback_alias_disables_fallback(monkeypatch):
    from aqi_predictor.config.settings import Settings

    monkeypatch.delenv("AQI_ALLOW_LOCAL_MODEL_FALLBACK", raising=False)
    monkeypatch.setenv("LOCAL_MODEL_FALLBACK_ENABLED", "false")

    assert Settings().allow_local_model_fallback is False


def test_city_environment_aliases_are_supported(monkeypatch):
    from aqi_predictor.config.settings import Settings

    for key in ("AQI_CITY", "AQI_LAT", "AQI_LON", "AQI_TIMEZONE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CITY_NAME", "Karachi")
    monkeypatch.setenv("CITY_LAT", "24.8607")
    monkeypatch.setenv("CITY_LON", "67.0011")
    monkeypatch.setenv("TIMEZONE", "Asia/Karachi")

    settings = Settings()

    assert settings.city == "Karachi"
    assert settings.lat == 24.8607
    assert settings.lon == 67.0011
    assert settings.timezone == "Asia/Karachi"


def test_vercel_defaults_use_tmp_runtime_dirs(monkeypatch):
    from aqi_predictor.config.settings import Settings

    monkeypatch.delenv("AQI_LOCAL_DATA_DIR", raising=False)
    monkeypatch.delenv("AQI_MODEL_DIR", raising=False)
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings()

    assert settings.local_data_dir == Path("/tmp/aqi_predictor/data")
    assert settings.model_dir == Path("/tmp/aqi_predictor/models")


def test_vercel_relocates_relative_runtime_dirs_to_tmp(monkeypatch):
    from aqi_predictor.config.settings import Settings

    monkeypatch.setenv("AQI_LOCAL_DATA_DIR", "data/processed")
    monkeypatch.setenv("AQI_MODEL_DIR", "models/latest")
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings()

    assert settings.local_data_dir == Path("/tmp/aqi_predictor/data/processed")
    assert settings.model_dir == Path("/tmp/aqi_predictor/models/latest")
