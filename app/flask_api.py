from __future__ import annotations

from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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

    settings = get_settings()
    app = Flask(__name__)
    app.config["DEBUG"] = False
    logging.basicConfig(level=logging.INFO)
    if CORS is not None:
        origins = [origin for origin in settings.allowed_origins_list if origin != "*"]
        if len(origins) != len(settings.allowed_origins_list):
            app.logger.warning("Wildcard CORS origin was ignored; configure explicit ALLOWED_ORIGINS.")
        if origins:
            CORS(app, resources={r"/*": {"origins": origins}})
    service = PredictionService(settings)

    class ApiInputError(ValueError):
        pass

    def _safe_error(message: str, status_code: int):
        return jsonify({"status": "error", "message": message}), status_code

    def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
        raw = request.args.get(name)
        if raw is None or raw == "":
            value = default
        else:
            try:
                value = int(raw)
            except ValueError as exc:
                raise ApiInputError(f"{name} must be an integer.") from exc
        return max(minimum, min(value, maximum))

    def _bool_arg(name: str, default: bool = False) -> bool:
        raw = request.args.get(name)
        if raw is None or raw == "":
            return default
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise ApiInputError(f"{name} must be a boolean.")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "city": settings.city, "timezone": settings.timezone})

    @app.get("/")
    def index():
        return jsonify(
            {
                "status": "ok",
                "service": "Karachi AQI Predictor API",
                "city": settings.city,
                "timezone": settings.timezone,
                "routes": ["/health", "/latest", "/predict?horizon=72", "/alerts", "/model-info"],
            }
        )

    @app.get("/latest")
    def latest():
        try:
            return jsonify(service.latest_payload())
        except Exception:
            app.logger.exception("Failed to load latest AQI payload.")
            return _safe_error("Latest AQI data is temporarily unavailable.", 503)

    @app.get("/predict")
    def predict():
        try:
            horizon = _bounded_int("horizon", settings.forecast_hours, 1, settings.max_forecast_hours)
            sample = _bool_arg("sample", default=False)
            return jsonify(service.predict(horizon=horizon, sample=sample))
        except ApiInputError as exc:
            return _safe_error(str(exc), 400)
        except Exception:
            app.logger.exception("Prediction request failed.")
            return _safe_error("Prediction service is temporarily unavailable.", 503)

    @app.get("/alerts")
    def alerts():
        try:
            limit = _bounded_int("limit", 20, 1, settings.max_api_limit)
            return jsonify(service.alerts_payload(limit=limit))
        except ApiInputError as exc:
            return _safe_error(str(exc), 400)
        except Exception:
            app.logger.exception("Alert request failed.")
            return _safe_error("Alert data is temporarily unavailable.", 503)

    @app.get("/model-info")
    def model_info():
        try:
            return jsonify(service.model_info_payload())
        except Exception:
            app.logger.exception("Model info request failed.")
            return _safe_error("Model information is temporarily unavailable.", 503)

    return app


app = create_app()


if __name__ == "__main__":
    _load_dotenv()
    settings = get_settings()
    app.run(host=settings.flask_host, port=settings.flask_port)
