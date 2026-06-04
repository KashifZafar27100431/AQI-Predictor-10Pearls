from __future__ import annotations

from types import MethodType
import time

import pandas as pd
import pytest

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.storage import FeatureStoreClient, HopsworksOperationTimeout, _run_with_timeout


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["Karachi"],
            "event_time": [pd.Timestamp("2026-06-04T00:00:00Z")],
            "aqi_score": [75.0],
        }
    )


class FakeFeatureGroup:
    def __init__(self):
        self.insert_calls = []

    def insert(self, frame, **kwargs):
        self.insert_calls.append(kwargs)
        return None


class LegacyFeatureGroup:
    def __init__(self):
        self.insert_calls = []

    def insert(self, frame, **kwargs):
        self.insert_calls.append(kwargs)
        if "write_options" in kwargs:
            raise TypeError("legacy insert signature")
        return None


class FlakyFeatureGroup:
    def __init__(self):
        self.insert_calls = 0

    def insert(self, frame, **kwargs):
        self.insert_calls += 1
        if self.insert_calls == 1:
            raise RuntimeError("temporary hopsworks failure")
        return None


class FakeFeatureStore:
    def __init__(self, feature_group):
        self.feature_group = feature_group
        self.get_or_create_calls = []

    def get_or_create_feature_group(self, **kwargs):
        self.get_or_create_calls.append(kwargs)
        return self.feature_group


def _client_with_fake_feature_store(tmp_path, feature_store):
    settings = Settings(
        local_data_dir=tmp_path,
        hopsworks_api_key="configured",
        hopsworks_project="project",
        hopsworks_feature_store_timeout_seconds=5,
        hopsworks_feature_store_max_retries=2,
        hopsworks_insert_wait_for_job=False,
    )
    client = FeatureStoreClient(settings)
    client._hopsworks_feature_store = MethodType(lambda self: feature_store, client)
    return client


def test_hopsworks_insert_does_not_wait_for_materialization_job(tmp_path):
    feature_group = FakeFeatureGroup()
    client = _client_with_fake_feature_store(tmp_path, FakeFeatureStore(feature_group))

    client.insert_feature_group("karachi_air_quality_raw", _frame(), primary_key=["city", "event_time"])

    assert feature_group.insert_calls == [{"write_options": {"wait_for_job": False}}]


def test_hopsworks_insert_uses_legacy_wait_false_fallback(tmp_path):
    feature_group = LegacyFeatureGroup()
    client = _client_with_fake_feature_store(tmp_path, FakeFeatureStore(feature_group))

    client.insert_feature_group("karachi_air_quality_raw", _frame(), primary_key=["city", "event_time"])

    assert feature_group.insert_calls == [
        {"write_options": {"wait_for_job": False}},
        {"wait": False},
    ]


def test_hopsworks_insert_retries_transient_failures_with_small_cap(tmp_path):
    feature_group = FlakyFeatureGroup()
    client = _client_with_fake_feature_store(tmp_path, FakeFeatureStore(feature_group))

    client.insert_feature_group("karachi_air_quality_raw", _frame(), primary_key=["city", "event_time"])

    assert feature_group.insert_calls == 2


def test_hopsworks_operation_timeout_raises_clear_exception():
    with pytest.raises(HopsworksOperationTimeout):
        _run_with_timeout("slow_hopsworks_call", 1, lambda: time.sleep(2))
