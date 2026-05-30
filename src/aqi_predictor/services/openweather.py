from __future__ import annotations

from datetime import datetime
import logging
import time
from typing import Any, Dict, Optional

import requests
from requests.exceptions import RequestException

from aqi_predictor.config.settings import Settings


class OpenWeatherError(RuntimeError):
    """Raised when OpenWeather cannot return usable data."""


logger = logging.getLogger(__name__)


class OpenWeatherClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openweather_api_key
        self.base_url = settings.openweather_base_url.rstrip("/")
        self.timeout = settings.openweather_timeout_seconds
        self.max_retries = max(1, settings.openweather_max_retries)
        self.retry_backoff_seconds = max(0.0, settings.openweather_retry_backoff_seconds)

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

        endpoint_name = endpoint.lstrip("/")
        url = f"{self.base_url}/{endpoint_name}"
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, params=request_params, timeout=self.timeout)
            except RequestException as exc:
                last_error = exc
                logger.warning("OpenWeather request failed for %s on attempt %s.", endpoint_name, attempt)
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                continue

            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                logger.warning(
                    "OpenWeather returned retryable HTTP %s for %s on attempt %s.",
                    response.status_code,
                    endpoint_name,
                    attempt,
                )
                self._sleep_before_retry(attempt)
                continue

            if response.status_code >= 400:
                raise OpenWeatherError(
                    f"OpenWeather returned HTTP {response.status_code} for endpoint '{endpoint_name}'."
                )

            try:
                payload = response.json()
            except ValueError as exc:
                last_error = exc
                logger.warning("OpenWeather returned malformed JSON for %s on attempt %s.", endpoint_name, attempt)
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise OpenWeatherError(
                    f"OpenWeather returned malformed JSON for endpoint '{endpoint_name}'."
                ) from exc

            if not isinstance(payload, dict):
                raise OpenWeatherError(
                    f"OpenWeather returned an unexpected payload for endpoint '{endpoint_name}'."
                )
            return payload

        raise OpenWeatherError(
            f"OpenWeather request failed for endpoint '{endpoint_name}' after {self.max_retries} attempts."
        ) from None

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        time.sleep(self.retry_backoff_seconds * attempt)

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
