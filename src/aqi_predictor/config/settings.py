from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Iterable, List, Optional


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_any(names: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw
    return default


def _local_model_fallback_enabled() -> bool:
    if os.getenv("AQI_ALLOW_LOCAL_MODEL_FALLBACK") is not None:
        return _bool_env("AQI_ALLOW_LOCAL_MODEL_FALLBACK", True)
    if os.getenv("LOCAL_MODEL_FALLBACK_ENABLED") is not None:
        return _bool_env("LOCAL_MODEL_FALLBACK_ENABLED", True)
    return not bool(os.getenv("VERCEL"))


def _runtime_path(env_name: str, local_default: str, serverless_default: str) -> Path:
    configured = os.getenv(env_name)
    if configured:
        path = Path(configured)
        if os.getenv("VERCEL") and not path.is_absolute():
            return Path("/tmp") / "aqi_predictor" / path
        return path
    if os.getenv("VERCEL"):
        return Path("/tmp") / "aqi_predictor" / serverless_default
    return Path(local_default)


def _int_env(name: str, default: int, aliases: Iterable[str] = ()) -> int:
    raw = _env_any((name, *aliases))
    if raw is None or raw == "":
        return default
    return int(raw)


def _float_env(name: str, default: float, aliases: Iterable[str] = ()) -> float:
    raw = _env_any((name, *aliases))
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    city: str = field(default_factory=lambda: _env_any(("AQI_CITY", "CITY_NAME"), "Karachi") or "Karachi")
    lat: float = field(default_factory=lambda: _float_env("AQI_LAT", 24.8607, aliases=("CITY_LAT",)))
    lon: float = field(default_factory=lambda: _float_env("AQI_LON", 67.0011, aliases=("CITY_LON",)))
    forecast_hours: int = field(default_factory=lambda: _int_env("AQI_FORECAST_HOURS", 72))
    timezone: str = field(
        default_factory=lambda: _env_any(("AQI_TIMEZONE", "TIMEZONE"), "Asia/Karachi") or "Asia/Karachi"
    )

    openweather_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENWEATHER_API_KEY") or None
    )
    openweather_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5"
        )
    )
    openweather_timeout_seconds: int = field(
        default_factory=lambda: _int_env("OPENWEATHER_TIMEOUT_SECONDS", 20)
    )
    openweather_max_retries: int = field(
        default_factory=lambda: _int_env("OPENWEATHER_MAX_RETRIES", 3)
    )
    openweather_retry_backoff_seconds: float = field(
        default_factory=lambda: _float_env("OPENWEATHER_RETRY_BACKOFF_SECONDS", 1.0)
    )

    hopsworks_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("HOPSWORKS_API_KEY") or None
    )
    hopsworks_project: Optional[str] = field(
        default_factory=lambda: os.getenv("HOPSWORKS_PROJECT") or None
    )

    mongodb_uri: Optional[str] = field(default_factory=lambda: os.getenv("MONGODB_URI") or None)
    mongodb_database: str = field(
        default_factory=lambda: os.getenv("MONGODB_DATABASE", "pearls_aqi")
    )

    local_data_dir: Path = field(
        default_factory=lambda: _runtime_path("AQI_LOCAL_DATA_DIR", "data/processed", "data")
    )
    model_dir: Path = field(
        default_factory=lambda: _runtime_path("AQI_MODEL_DIR", "models/latest", "models")
    )
    allow_local_model_fallback: bool = field(
        default_factory=_local_model_fallback_enabled
    )
    require_hopsworks_model_registry: bool = field(
        default_factory=lambda: _bool_env("AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY", bool(os.getenv("VERCEL")))
    )
    hopsworks_model_version: Optional[int] = field(
        default_factory=lambda: (
            _int_env("HOPSWORKS_MODEL_VERSION", 0)
            if os.getenv("HOPSWORKS_MODEL_VERSION")
            else None
        )
    )
    max_weather_missing_fraction: float = field(
        default_factory=lambda: _float_env("AQI_MAX_WEATHER_MISSING_FRACTION", 0.50)
    )
    use_sample_data: bool = field(
        default_factory=lambda: _bool_env("AQI_USE_SAMPLE_DATA", False)
    )
    max_forecast_hours: int = field(default_factory=lambda: _int_env("AQI_MAX_FORECAST_HOURS", 72))
    max_api_limit: int = field(default_factory=lambda: _int_env("AQI_MAX_API_LIMIT", 500))

    flask_host: str = field(default_factory=lambda: os.getenv("FLASK_HOST", "0.0.0.0"))
    flask_port: int = field(default_factory=lambda: _int_env("FLASK_PORT", 8000))
    allowed_origins: str = field(
        default_factory=lambda: os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
        )
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


def get_settings() -> Settings:
    return Settings()
