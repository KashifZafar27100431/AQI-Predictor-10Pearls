from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.prediction import PredictionService


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        return


def create_app():
    _load_dotenv()
    from flask import Flask, jsonify, request

    try:
        from flask_cors import CORS
    except Exception:
        CORS = None

    app = Flask(__name__)
    if CORS is not None:
        CORS(app)

    settings = get_settings()
    service = PredictionService(settings)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "city": settings.city})

    @app.get("/latest")
    def latest():
        return jsonify(service.latest_payload())

    @app.get("/predict")
    def predict():
        horizon = request.args.get("horizon", default=settings.forecast_hours, type=int)
        sample = request.args.get("sample", default="false").lower() in {"1", "true", "yes"}
        try:
            return jsonify(service.predict(horizon=horizon, sample=sample))
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.get("/alerts")
    def alerts():
        limit = request.args.get("limit", default=20, type=int)
        return jsonify(service.alerts_payload(limit=limit))

    @app.get("/model-info")
    def model_info():
        try:
            return jsonify(service.model_info_payload())
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    return app


if __name__ == "__main__":
    _load_dotenv()
    app = create_app()
    settings = get_settings()
    app.run(host=settings.flask_host, port=settings.flask_port)

