from __future__ import annotations

from types import SimpleNamespace

import app.flask_api as flask_api


class FakePredictionService:
    def __init__(self, settings):
        self.settings = settings
        self.openweather = SimpleNamespace(configured=True)
        self.last_limit = None

    def latest_payload(self):
        return {"city": self.settings.city, "latest": None}

    def predict(self, horizon=None, sample=False):
        return {
            "city": self.settings.city,
            "horizon_hours": horizon,
            "sample": sample,
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
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:8501")
    app = flask_api.create_app()
    return app.test_client()


def test_health_latest_alerts_and_model_info(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/health").status_code == 200
    assert client.get("/latest").status_code == 200
    assert client.get("/alerts?limit=999").json["limit"] == 500
    assert client.get("/model-info").json["model"]["serving_source"] == "local_model_dir"


def test_predict_horizon_is_clamped_and_invalid_params_are_rejected(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/predict?horizon=999").json["horizon_hours"] == 72
    assert client.get("/predict?horizon=0").json["horizon_hours"] == 1
    assert client.get("/predict?horizon=abc").status_code == 400
    assert client.get("/predict?sample=maybe").status_code == 400


def test_cors_uses_configured_origins(monkeypatch):
    client = _client(monkeypatch)

    allowed = client.get("/health", headers={"Origin": "http://localhost:8501"})
    denied = client.get("/health", headers={"Origin": "https://evil.example"})

    assert allowed.headers.get("Access-Control-Allow-Origin") == "http://localhost:8501"
    assert denied.headers.get("Access-Control-Allow-Origin") is None


def test_wildcard_cors_origin_is_ignored(monkeypatch):
    monkeypatch.setattr(flask_api, "PredictionService", FakePredictionService)
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    app = flask_api.create_app()
    client = app.test_client()

    response = client.get("/health", headers={"Origin": "https://evil.example"})

    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") is None
