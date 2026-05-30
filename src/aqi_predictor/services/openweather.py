from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import requests
from requests.exceptions import RequestException

from aqi_predictor.config.settings import Settings


class OpenWeatherError(RuntimeError):
    """Raised when OpenWeather cannot return usable data."""


class OpenWeatherClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openweather_api_key
        self.base_url = settings.openweather_base_url.rstrip("/")
        self.timeout = settings.openweather_timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise OpenWeatherError("OPENWEATHER_API_KEY is not configured.")

        request_params: Dict[str, Any] = {
            "lat": self.settings.lat,
            "lon": self.settings.lon,
            "appid": self.api_key,
        }
        if params:
            request_params.update(params)

        try:
            response = requests.get(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                params=request_params,
                timeout=self.timeout,
            )
        except RequestException as exc:
            raise OpenWeatherError(
                f"OpenWeather request failed for endpoint '{endpoint.lstrip('/')}'."
            ) from exc
        if response.status_code >= 400:
            raise OpenWeatherError(
                f"OpenWeather returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()

    def current_air_pollution(self) -> Dict[str, Any]:
        return self._get("air_pollution")

    def forecast_air_pollution(self) -> Dict[str, Any]:
        return self._get("air_pollution/forecast")

    def historical_air_pollution(self, start: datetime, end: datetime) -> Dict[str, Any]:
        return self._get(
            "air_pollution/history",
            {"start": int(start.timestamp()), "end": int(end.timestamp())},
        )

    def current_weather(self) -> Dict[str, Any]:
        return self._get("weather", {"units": "metric"})

    def forecast_weather(self) -> Dict[str, Any]:
        return self._get("forecast", {"units": "metric"})
