from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Optional


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    city: str = field(default_factory=lambda: os.getenv("AQI_CITY", "Karachi"))
    lat: float = field(default_factory=lambda: _float_env("AQI_LAT", 24.8607))
    lon: float = field(default_factory=lambda: _float_env("AQI_LON", 67.0011))
    forecast_hours: int = field(default_factory=lambda: _int_env("AQI_FORECAST_HOURS", 72))
    timezone: str = field(default_factory=lambda: os.getenv("AQI_TIMEZONE", "Asia/Karachi"))

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
        default_factory=lambda: Path(os.getenv("AQI_LOCAL_DATA_DIR", "data/processed"))
    )
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AQI_MODEL_DIR", "models/latest"))
    )
    use_sample_data: bool = field(
        default_factory=lambda: _bool_env("AQI_USE_SAMPLE_DATA", False)
    )

    flask_host: str = field(default_factory=lambda: os.getenv("FLASK_HOST", "0.0.0.0"))
    flask_port: int = field(default_factory=lambda: _int_env("FLASK_PORT", 8000))


def get_settings() -> Settings:
    return Settings()
