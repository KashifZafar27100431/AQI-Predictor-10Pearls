from __future__ import annotations

from pathlib import Path
import importlib.util
import logging
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.modeling import get_last_model_load_status
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

    def _module_available(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def _env_present(*names: str) -> bool:
        import os

        return any(bool(os.getenv(name)) for name in names)

    def _check_value(ok: bool) -> str:
        return "ok" if ok else "missing"

    def _model_cache_status() -> dict:
        path = settings.model_dir
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return {"status": "ok", "path": str(path), "writable": True}
        except Exception:
            return {"status": "failed", "path": str(path), "writable": False}

    def _env_checks() -> dict:
        return {
            "OPENWEATHER_API_KEY": _env_present("OPENWEATHER_API_KEY"),
            "HOPSWORKS_API_KEY": _env_present("HOPSWORKS_API_KEY"),
            "HOPSWORKS_PROJECT": _env_present("HOPSWORKS_PROJECT"),
            "MONGODB_URI": _env_present("MONGODB_URI"),
            "MONGODB_DATABASE": _env_present("MONGODB_DATABASE"),
            "AQI_CITY": _env_present("AQI_CITY", "CITY_NAME"),
            "AQI_LAT": _env_present("AQI_LAT", "CITY_LAT"),
            "AQI_LON": _env_present("AQI_LON", "CITY_LON"),
            "AQI_TIMEZONE": _env_present("AQI_TIMEZONE", "TIMEZONE"),
            "HOPSWORKS_MODEL_VERSION": _env_present("HOPSWORKS_MODEL_VERSION"),
            "AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY": _env_present("AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY"),
            "AQI_ALLOW_LOCAL_MODEL_FALLBACK": _env_present(
                "AQI_ALLOW_LOCAL_MODEL_FALLBACK", "LOCAL_MODEL_FALLBACK_ENABLED"
            ),
            "ALLOWED_ORIGINS": _env_present("ALLOWED_ORIGINS"),
        }

    def _package_checks() -> dict:
        return {
            "flask": _module_available("flask"),
            "flask_cors": _module_available("flask_cors"),
            "hopsworks": _module_available("hopsworks"),
            "pyarrow": _module_available("pyarrow"),
            "pandas": _module_available("pandas"),
            "sklearn": _module_available("sklearn"),
            "joblib": _module_available("joblib"),
            "pymongo": _module_available("pymongo"),
        }

    def _readiness_payload() -> tuple[dict, int]:
        checks = {
            "env": "ok",
            "pyarrow": _check_value(_module_available("pyarrow")),
            "hopsworks_import": _check_value(_module_available("hopsworks")),
            "sklearn_joblib": _check_value(_module_available("sklearn") and _module_available("joblib")),
            "model_registry": "unknown",
            "model_cache": _model_cache_status()["status"],
            "openweather": "unknown",
            "configuration": "ok",
        }
        required_env = [
            "OPENWEATHER_API_KEY",
            "HOPSWORKS_API_KEY",
            "HOPSWORKS_PROJECT",
            "MONGODB_URI",
            "MONGODB_DATABASE",
            "AQI_CITY",
            "AQI_LAT",
            "AQI_LON",
            "AQI_TIMEZONE",
        ]
        env = _env_checks()
        if not all(env.get(key) for key in required_env):
            checks["env"] = "missing"
        if not settings.city or not settings.timezone or settings.lat == 0 or settings.lon == 0:
            checks["configuration"] = "failed"

        model_payload = {}
        try:
            metadata = service.model_metadata()
            checks["model_registry"] = "ok"
            model_payload = {
                "source": metadata.get("serving_source"),
                "version": metadata.get("registry_version"),
                "model_type": metadata.get("model_name") or metadata.get("model_type"),
                "trained_at": metadata.get("trained_at"),
            }
        except Exception as exc:
            app.logger.exception("Readiness model registry check failed.")
            checks["model_registry"] = getattr(exc, "category", "model_registry_unavailable")

        try:
            if service.openweather.configured:
                original_timeout = getattr(service.openweather, "timeout", None)
                original_retries = getattr(service.openweather, "max_retries", None)
                if original_timeout is not None:
                    service.openweather.timeout = min(original_timeout, 5)
                if original_retries is not None:
                    service.openweather.max_retries = 1
                try:
                    service.openweather.current_air_pollution()
                finally:
                    if original_timeout is not None:
                        service.openweather.timeout = original_timeout
                    if original_retries is not None:
                        service.openweather.max_retries = original_retries
                checks["openweather"] = "ok"
            else:
                checks["openweather"] = "missing"
        except Exception:
            app.logger.exception("Readiness OpenWeather check failed.")
            checks["openweather"] = "openweather_unavailable"

        failed = [value for value in checks.values() if value not in {"ok"}]
        status_code = 200 if not failed else 503
        return (
            {
                "status": "ready" if status_code == 200 else "not_ready",
                "city": settings.city,
                "timezone": settings.timezone,
                "checks": checks,
                "model": model_payload,
                "last_model_load": get_last_model_load_status(),
            },
            status_code,
        )

    def _diagnostics_payload() -> dict:
        model_cache = _model_cache_status()
        return {
            "status": "ok",
            "service": "Karachi AQI Predictor API",
            "python_version": platform.python_version(),
            "runtime": {
                "vercel": _env_present("VERCEL"),
                "serverless": _env_present("VERCEL", "AWS_LAMBDA_FUNCTION_NAME", "K_SERVICE"),
                "model_cache_dir": str(settings.model_dir),
                "data_cache_dir": str(settings.local_data_dir),
                "model_cache_writable": model_cache["writable"],
            },
            "packages": _package_checks(),
            "env": _env_checks(),
            "configuration": {
                "city": settings.city,
                "lat": settings.lat,
                "lon": settings.lon,
                "timezone": settings.timezone,
                "pinned_model_version": settings.hopsworks_model_version,
                "local_model_fallback_enabled": settings.allow_local_model_fallback,
                "hopsworks_registry_required": settings.require_hopsworks_model_registry,
                "max_forecast_hours": settings.max_forecast_hours,
            },
            "last_model_load": get_last_model_load_status(),
        }

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
                "routes": [
                    "/health",
                    "/ready",
                    "/diagnostics",
                    "/latest",
                    "/predict?horizon=72",
                    "/alerts",
                    "/model-info",
                ],
            }
        )

    @app.get("/diagnostics")
    def diagnostics():
        return jsonify(_diagnostics_payload())

    @app.get("/ready")
    def ready():
        payload, status_code = _readiness_payload()
        return jsonify(payload), status_code

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
            return jsonify(service.predict(horizon=horizon, sample=sample, store_predictions=False))
        except ApiInputError as exc:
            return _safe_error(str(exc), 400)
        except Exception:
            app.logger.exception("Prediction request failed.")
            return _safe_error("Prediction service is temporarily unavailable.", 503)

    @app.get("/alerts")
    def alerts():
        try:
            limit = _bounded_int("limit", 20, 1, settings.max_api_limit)
            include_forecast = _bool_arg("forecast", default=True)
            return jsonify(service.alerts_payload(limit=limit, include_forecast=include_forecast))
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
