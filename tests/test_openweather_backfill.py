from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from requests.exceptions import ConnectionError

from aqi_predictor.config.settings import Settings
from aqi_predictor.pipelines.backfill_pipeline import _clean_hourly_frame, _continuity_report
from aqi_predictor.services.openweather import OpenWeatherClient, OpenWeatherError


class FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def test_openweather_retries_retryable_http_status(monkeypatch):
    calls = []
    responses = [FakeResponse(500), FakeResponse(200, {"list": []})]

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr("aqi_predictor.services.openweather.requests.get", fake_get)
    monkeypatch.setattr("aqi_predictor.services.openweather.time.sleep", lambda seconds: None)

    settings = Settings(
        openweather_api_key="test-key",
        openweather_max_retries=2,
        openweather_retry_backoff_seconds=0,
    )
    assert OpenWeatherClient(settings).current_air_pollution() == {"list": []}
    assert len(calls) == 2


def test_openweather_request_errors_do_not_expose_api_key(monkeypatch):
    def fake_get(*args, **kwargs):
        raise ConnectionError("full url would normally include appid=test-secret")

    monkeypatch.setattr("aqi_predictor.services.openweather.requests.get", fake_get)
    monkeypatch.setattr("aqi_predictor.services.openweather.time.sleep", lambda seconds: None)
    settings = Settings(
        openweather_api_key="test-secret",
        openweather_max_retries=1,
        openweather_retry_backoff_seconds=0,
    )

    with pytest.raises(OpenWeatherError) as exc_info:
        OpenWeatherClient(settings).current_air_pollution()

    assert "test-secret" not in str(exc_info.value)
    assert "appid" not in str(exc_info.value)


def test_backfill_cleaning_deduplicates_and_reports_missing_hours():
    frame = pd.DataFrame(
        [
            {"city": "Karachi", "event_time": "2026-05-30T00:15:00Z", "aqi_score": 50},
            {"city": "Karachi", "event_time": "2026-05-30T00:45:00Z", "aqi_score": 60},
            {"city": "Karachi", "event_time": "2026-05-30T02:00:00Z", "aqi_score": 70},
        ]
    )

    clean = _clean_hourly_frame(frame, ["city", "event_time"])
    report = _continuity_report(
        clean,
        datetime(2026, 5, 30, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 30, 2, tzinfo=timezone.utc),
    )

    assert len(clean) == 2
    assert clean.iloc[0]["aqi_score"] == 60
    assert report == {"expected_hours": 3, "observed_hours": 2, "missing_hours": 1}
