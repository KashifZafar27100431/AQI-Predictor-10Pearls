from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from pandas.errors import EmptyDataError

from aqi_predictor.config.settings import Settings


RAW_AIR_FEATURE_GROUP = "karachi_air_quality_raw"
RAW_WEATHER_FEATURE_GROUP = "karachi_weather_raw"
FEATURES_FEATURE_GROUP = "karachi_aqi_features"
PREDICTIONS_FEATURE_GROUP = "karachi_aqi_predictions"
MODEL_NAME = "karachi_aqi_predictor"
MODEL_ARTIFACT_FILE = "model.joblib"
MODEL_METADATA_FILE = "metadata.json"

logger = logging.getLogger(__name__)


class FeatureStoreClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.local_dir = settings.local_data_dir
        self._fs = None

    def _path(self, name: str) -> Path:
        return self.local_dir / f"{name}.csv"

    def _read_local(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        if not path.exists():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
        if "event_time" in frame.columns:
            frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        return frame

    def _insert_local(self, name: str, frame: pd.DataFrame, primary_key: Iterable[str]) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_local(name)
        combined = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame.copy()
        key_columns = [column for column in primary_key if column in combined.columns]
        if key_columns:
            combined = combined.drop_duplicates(subset=key_columns, keep="last")
        if "event_time" in combined.columns:
            combined = combined.sort_values("event_time")
        path = self._path(name)
        tmp_path = path.with_suffix(".tmp")
        combined.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

    def _hopsworks_feature_store(self):
        if self._fs is not None:
            return self._fs
        if not (self.settings.hopsworks_api_key and self.settings.hopsworks_project):
            return None
        import hopsworks

        project = hopsworks.login(
            project=self.settings.hopsworks_project,
            api_key_value=self.settings.hopsworks_api_key,
        )
        self._fs = project.get_feature_store()
        return self._fs

    def _insert_hopsworks(
        self,
        name: str,
        frame: pd.DataFrame,
        primary_key: List[str],
        event_time: str = "event_time",
    ) -> None:
        fs = self._hopsworks_feature_store()
        if fs is None:
            return
        feature_group = fs.get_or_create_feature_group(
            name=name,
            version=1,
            description=f"Pearls AQI Predictor feature group: {name}",
            primary_key=primary_key,
            event_time=event_time,
            online_enabled=True,
        )
        try:
            feature_group.insert(frame, wait=True)
        except TypeError:
            feature_group.insert(frame, write_options={"wait_for_job": True})

    def insert_feature_group(
        self,
        name: str,
        frame: pd.DataFrame,
        primary_key: Optional[List[str]] = None,
        event_time: str = "event_time",
    ) -> None:
        if frame is None or frame.empty:
            return
        keys = primary_key or ["city", "event_time"]
        clean = frame.copy()
        if event_time in clean.columns:
            clean[event_time] = pd.to_datetime(clean[event_time], utc=True)
        self._insert_local(name, clean, keys)
        self._insert_hopsworks(name, clean, keys, event_time=event_time)

    def read_feature_group(self, name: str, limit: Optional[int] = None) -> pd.DataFrame:
        frame = pd.DataFrame()
        hopsworks_configured = bool(self.settings.hopsworks_api_key and self.settings.hopsworks_project)
        try:
            fs = self._hopsworks_feature_store()
        except Exception:
            if hopsworks_configured:
                logger.warning("Hopsworks feature-store connection failed for %s; falling back to local cache.", name)
            fs = None
        if fs is not None:
            try:
                frame = fs.get_feature_group(name=name, version=1).read()
                if "event_time" in frame.columns:
                    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
            except Exception:
                if hopsworks_configured:
                    logger.warning("Hopsworks feature-group read failed for %s; falling back to local cache.", name)
                frame = pd.DataFrame()
        if frame.empty:
            frame = self._read_local(name)
            if hopsworks_configured and not frame.empty:
                logger.warning("Using local cached feature group %s because Hopsworks data was unavailable.", name)
        if not frame.empty and "event_time" in frame.columns:
            frame = frame.sort_values("event_time")
        if limit is not None and not frame.empty:
            frame = frame.tail(limit)
        return frame


class MongoStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self._db = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.mongodb_uri)

    def _database(self):
        if not self.settings.mongodb_uri:
            return None
        if self._db is not None:
            return self._db
        from pymongo import MongoClient

        self._client = MongoClient(self.settings.mongodb_uri, serverSelectionTimeoutMS=5000)
        self._db = self._client[self.settings.mongodb_database]
        return self._db

    def upsert_latest(self, record: Dict[str, Any]) -> None:
        db = self._database()
        if db is None:
            return
        payload = dict(record)
        payload["updated_at"] = datetime.now(timezone.utc)
        db.latest.update_one({"city": payload.get("city")}, {"$set": payload}, upsert=True)

    def insert_predictions(self, records: List[Dict[str, Any]]) -> None:
        db = self._database()
        if db is None or not records:
            return
        now = datetime.now(timezone.utc)
        payload = [dict(record, created_at=now) for record in records]
        db.predictions.insert_many(payload)
        alerts = [record for record in payload if record.get("alert_level") in {"unhealthy", "hazardous"}]
        if alerts:
            db.alerts.insert_many(alerts)

    def list_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        db = self._database()
        if db is None:
            return []
        rows = list(db.alerts.find({}, {"_id": 0}).sort("event_time", -1).limit(limit))
        return rows


class ModelRegistryClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir = settings.model_dir

    def metadata_path(self) -> Path:
        return self.model_dir / MODEL_METADATA_FILE

    @property
    def registry_cache_dir(self) -> Path:
        return self.model_dir / "registry_cache"

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with self.metadata_path().open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def load_metadata(self) -> Dict[str, Any]:
        path = self.metadata_path()
        if not path.exists():
            raise FileNotFoundError(f"Model metadata not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _hopsworks_project(self):
        if not (self.settings.hopsworks_api_key and self.settings.hopsworks_project):
            if self.settings.require_hopsworks_model_registry:
                raise RuntimeError("Hopsworks credentials are required for model registry operations.")
            return None
        import hopsworks

        return hopsworks.login(
            project=self.settings.hopsworks_project,
            api_key_value=self.settings.hopsworks_api_key,
        )

    def register_hopsworks(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        project = self._hopsworks_project()
        if project is None:
            return {"registered": False, "reason": "hopsworks_not_configured"}
        registry = project.get_model_registry()
        model = registry.python.create_model(MODEL_NAME, metrics=metrics)
        model.save(str(self.model_dir))
        return {
            "registered": True,
            "model_name": MODEL_NAME,
            "version": _safe_model_version(model),
            "source": "hopsworks_model_registry",
        }

    def download_latest_hopsworks_model(self) -> Dict[str, Any]:
        project = self._hopsworks_project()
        if project is None:
            raise RuntimeError("Hopsworks credentials are not configured.")
        registry = project.get_model_registry()
        model = self._select_hopsworks_model(registry)
        version = _safe_model_version(model)
        target_dir = self.registry_cache_dir / f"version_{version or 'latest'}"
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = model.download(str(target_dir))
        except TypeError:
            downloaded = model.download()
        downloaded_dir = Path(downloaded) if downloaded else target_dir
        try:
            artifact_dir = _safe_resolve_under(self.registry_cache_dir, downloaded_dir)
        except ValueError:
            _copy_expected_model_artifacts(downloaded_dir, target_dir)
            artifact_dir = target_dir.resolve()
        metadata_path = _find_required_file(artifact_dir, MODEL_METADATA_FILE)
        metadata = _read_metadata(metadata_path)
        model_path = _expected_model_path(artifact_dir, metadata)
        return {
            "source": "hopsworks_model_registry",
            "model_name": MODEL_NAME,
            "version": version,
            "artifact_dir": artifact_dir,
            "model_path": model_path,
            "metadata_path": metadata_path,
        }

    def _select_hopsworks_model(self, registry: Any) -> Any:
        if self.settings.hopsworks_model_version is not None:
            return registry.get_model(MODEL_NAME, version=self.settings.hopsworks_model_version)

        candidates: List[Any] = []
        try:
            candidates = list(registry.get_models(name=MODEL_NAME))
        except TypeError:
            candidates = list(registry.get_models(MODEL_NAME))
        except Exception:
            try:
                candidates = [registry.get_model(MODEL_NAME)]
            except TypeError:
                candidates = [registry.get_model(name=MODEL_NAME)]

        if not candidates:
            raise RuntimeError(f"No Hopsworks model versions found for {MODEL_NAME}.")

        approved = [model for model in candidates if _is_approved_model(model)]
        selectable = approved or candidates
        return sorted(selectable, key=_safe_model_version)[-1]


def _safe_model_version(model: Any) -> int:
    raw = getattr(model, "version", None) or getattr(model, "_version", None) or 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _is_approved_model(model: Any) -> bool:
    status = str(getattr(model, "status", "") or getattr(model, "model_status", "")).lower()
    if status in {"approved", "production", "prod"}:
        return True
    tags = getattr(model, "tags", {}) or {}
    if isinstance(tags, dict):
        return str(tags.get("stage", "")).lower() in {"approved", "production", "prod"}
    return False


def _safe_resolve_under(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    candidate_resolved.relative_to(base_resolved)
    return candidate_resolved


def _find_required_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists() and direct.is_file():
        return direct
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Downloaded model artifact is missing {filename}.")
    return matches[0]


def _read_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_model_path(root: Path, metadata: Dict[str, Any]) -> Optional[Path]:
    model_type = metadata.get("model_type")
    artifacts = metadata.get("artifacts", {})
    if model_type == "tensorflow_keras":
        relative = "tensorflow_experiment/model.keras"
        if isinstance(artifacts, dict):
            relative = artifacts.get("keras_model", relative)
        path = root / relative
        if not path.exists():
            path = _find_required_file(root, "model.keras")
        scaler_relative = "tensorflow_experiment/scaler.joblib"
        if isinstance(artifacts, dict):
            scaler_relative = artifacts.get("scaler", scaler_relative)
        scaler_path = root / scaler_relative
        if not scaler_path.exists():
            _find_required_file(root, "scaler.joblib")
        return path
    relative = MODEL_ARTIFACT_FILE
    if isinstance(artifacts, dict):
        relative = artifacts.get("sklearn_model", relative)
    path = root / relative
    if path.exists():
        return path
    return _find_required_file(root, MODEL_ARTIFACT_FILE)


def _copy_expected_model_artifacts(source_dir: Path, target_dir: Path) -> None:
    metadata_path = _find_required_file(source_dir, MODEL_METADATA_FILE)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, target_dir / MODEL_METADATA_FILE)
    metadata = _read_metadata(metadata_path)
    model_type = metadata.get("model_type")
    artifacts = metadata.get("artifacts", {})
    if model_type == "tensorflow_keras":
        keras_relative = "tensorflow_experiment/model.keras"
        scaler_relative = "tensorflow_experiment/scaler.joblib"
        if isinstance(artifacts, dict):
            keras_relative = artifacts.get("keras_model", keras_relative)
            scaler_relative = artifacts.get("scaler", scaler_relative)
        keras_source = source_dir / keras_relative
        if not keras_source.exists():
            keras_source = _find_required_file(source_dir, "model.keras")
        scaler_source = source_dir / scaler_relative
        if not scaler_source.exists():
            scaler_source = _find_required_file(source_dir, "scaler.joblib")
        keras_target = target_dir / keras_relative
        scaler_target = target_dir / scaler_relative
        keras_target.parent.mkdir(parents=True, exist_ok=True)
        scaler_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(keras_source, keras_target)
        shutil.copy2(scaler_source, scaler_target)
        return
    model_source = source_dir / MODEL_ARTIFACT_FILE
    if not model_source.exists():
        model_source = _find_required_file(source_dir, MODEL_ARTIFACT_FILE)
    shutil.copy2(model_source, target_dir / MODEL_ARTIFACT_FILE)
