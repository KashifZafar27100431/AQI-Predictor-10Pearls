from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.features import FEATURE_COLUMNS
from aqi_predictor.services.modeling import ModelLoadError, load_model_bundle
from aqi_predictor.services.storage import ModelRegistryClient


def _write_model_artifact(model_dir: Path, feature_columns=None) -> None:
    selected = feature_columns or FEATURE_COLUMNS
    model_dir.mkdir(parents=True, exist_ok=True)
    model = Ridge().fit(np.zeros((3, len(selected))), np.array([1.0, 2.0, 3.0]))
    joblib.dump(model, model_dir / "model.joblib")
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": "sklearn",
                "model_name": "ridge",
                "feature_columns": selected,
                "metrics": {"rmse": 1.0, "mae": 1.0, "r2": 0.0},
                "trained_at": "2026-05-30T00:00:00Z",
            },
            handle,
        )


def test_model_loader_uses_explicit_local_fallback_when_registry_fails(tmp_path, monkeypatch):
    _write_model_artifact(tmp_path)
    settings = Settings(
        model_dir=tmp_path,
        hopsworks_api_key="configured",
        hopsworks_project="project",
        allow_local_model_fallback=True,
    )

    def fail_registry(self):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(ModelRegistryClient, "download_latest_hopsworks_model", fail_registry)

    model, metadata = load_model_bundle(settings)

    assert model is not None
    assert metadata["serving_source"] == "local_model_dir"
    assert metadata["feature_count"] == len(FEATURE_COLUMNS)


def test_model_loader_rejects_unknown_feature_schema(tmp_path):
    _write_model_artifact(tmp_path, feature_columns=["unknown_feature"])
    settings = Settings(model_dir=tmp_path, allow_local_model_fallback=True)

    with pytest.raises(ModelLoadError):
        load_model_bundle(settings)


def test_model_loader_fails_when_registry_unavailable_and_local_fallback_disabled(tmp_path, monkeypatch):
    settings = Settings(
        model_dir=tmp_path,
        hopsworks_api_key="configured",
        hopsworks_project="project",
        allow_local_model_fallback=False,
    )

    def fail_registry(self):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(ModelRegistryClient, "download_latest_hopsworks_model", fail_registry)

    with pytest.raises(ModelLoadError):
        load_model_bundle(settings)


def test_registry_download_is_copied_to_controlled_cache(tmp_path, monkeypatch):
    external_dir = tmp_path / "external-download"
    _write_model_artifact(external_dir)

    class FakeModel:
        version = 7

        def download(self):
            return str(external_dir)

    class FakeRegistry:
        def get_models(self, name=None):
            return [FakeModel()]

    class FakeProject:
        def get_model_registry(self):
            return FakeRegistry()

    monkeypatch.setattr(ModelRegistryClient, "_hopsworks_project", lambda self: FakeProject())
    settings = Settings(
        model_dir=tmp_path / "serving",
        hopsworks_api_key="configured",
        hopsworks_project="project",
    )

    downloaded = ModelRegistryClient(settings).download_latest_hopsworks_model()

    assert downloaded["version"] == 7
    assert downloaded["model_path"].is_relative_to((settings.model_dir / "registry_cache").resolve())
    assert downloaded["metadata_path"].is_relative_to((settings.model_dir / "registry_cache").resolve())


def test_registry_download_reuses_existing_complete_cache(tmp_path, monkeypatch):
    settings = Settings(
        model_dir=tmp_path / "serving",
        hopsworks_api_key="configured",
        hopsworks_project="project",
    )
    cached_dir = settings.model_dir / "registry_cache" / "version_9"
    _write_model_artifact(cached_dir)

    class FakeModel:
        version = 9

        def download(self, *args):
            raise AssertionError("Complete model cache should be reused without downloading.")

    class FakeRegistry:
        def get_models(self, name=None):
            return [FakeModel()]

    class FakeProject:
        def get_model_registry(self):
            return FakeRegistry()

    monkeypatch.setattr(ModelRegistryClient, "_hopsworks_project", lambda self: FakeProject())

    downloaded = ModelRegistryClient(settings).download_latest_hopsworks_model()

    assert downloaded["version"] == 9
    assert downloaded["artifact_dir"] == cached_dir.resolve()
    assert downloaded["metadata_path"] == (cached_dir / "metadata.json").resolve()


def test_registry_download_refreshes_incomplete_existing_cache(tmp_path, monkeypatch):
    settings = Settings(
        model_dir=tmp_path / "serving",
        hopsworks_api_key="configured",
        hopsworks_project="project",
    )
    stale_dir = settings.model_dir / "registry_cache" / "version_11"
    (stale_dir / "tensorflow_experiment").mkdir(parents=True)

    class FakeModel:
        version = 11

        def download(self, local_path):
            target = Path(local_path)
            assert not (target / "tensorflow_experiment").exists()
            _write_model_artifact(target)
            return str(target)

    class FakeRegistry:
        def get_models(self, name=None):
            return [FakeModel()]

    class FakeProject:
        def get_model_registry(self):
            return FakeRegistry()

    monkeypatch.setattr(ModelRegistryClient, "_hopsworks_project", lambda self: FakeProject())

    downloaded = ModelRegistryClient(settings).download_latest_hopsworks_model()

    assert downloaded["version"] == 11
    assert downloaded["metadata_path"] == (stale_dir / "metadata.json").resolve()
    assert downloaded["model_path"] == (stale_dir / "model.joblib").resolve()


def test_registry_prefers_approved_model_over_newer_unapproved(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self, version, status):
            self.version = version
            self.status = status

    class FakeRegistry:
        def get_models(self, name=None):
            return [FakeModel(10, ""), FakeModel(4, "APPROVED")]

    settings = Settings(model_dir=tmp_path)
    selected = ModelRegistryClient(settings)._select_hopsworks_model(FakeRegistry())

    assert selected.version == 4


def test_tensorflow_registry_artifacts_are_copied_to_controlled_cache(tmp_path, monkeypatch):
    external_dir = tmp_path / "external-tf"
    tf_dir = external_dir / "tensorflow_experiment"
    tf_dir.mkdir(parents=True)
    (tf_dir / "model.keras").write_text("keras-placeholder", encoding="utf-8")
    (tf_dir / "scaler.joblib").write_bytes(b"joblib-placeholder")
    with (external_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": "tensorflow_keras",
                "model_name": "tensorflow_dense",
                "feature_columns": FEATURE_COLUMNS,
                "artifacts": {
                    "keras_model": "tensorflow_experiment/model.keras",
                    "scaler": "tensorflow_experiment/scaler.joblib",
                },
                "metrics": {"rmse": 0.1, "mae": 0.1, "r2": 0.9},
                "trained_at": "2026-05-30T00:00:00Z",
            },
            handle,
        )

    class FakeModel:
        version = 8

        def download(self):
            return str(external_dir)

    class FakeRegistry:
        def get_models(self, name=None):
            return [FakeModel()]

    class FakeProject:
        def get_model_registry(self):
            return FakeRegistry()

    monkeypatch.setattr(ModelRegistryClient, "_hopsworks_project", lambda self: FakeProject())
    settings = Settings(
        model_dir=tmp_path / "serving-tf",
        hopsworks_api_key="configured",
        hopsworks_project="project",
    )

    downloaded = ModelRegistryClient(settings).download_latest_hopsworks_model()

    assert downloaded["model_path"].name == "model.keras"
    assert (downloaded["artifact_dir"] / "tensorflow_experiment/model.keras").exists()
    assert (downloaded["artifact_dir"] / "tensorflow_experiment/scaler.joblib").exists()
