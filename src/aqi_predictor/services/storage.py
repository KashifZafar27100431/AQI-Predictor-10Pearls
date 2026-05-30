from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from pandas.errors import EmptyDataError

from aqi_predictor.config.settings import Settings


RAW_AIR_FEATURE_GROUP = "karachi_air_quality_raw"
RAW_WEATHER_FEATURE_GROUP = "karachi_weather_raw"
FEATURES_FEATURE_GROUP = "karachi_aqi_features"
PREDICTIONS_FEATURE_GROUP = "karachi_aqi_predictions"
MODEL_NAME = "karachi_aqi_predictor"


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
        try:
            fs = self._hopsworks_feature_store()
        except Exception:
            fs = None
        if fs is not None:
            try:
                frame = fs.get_feature_group(name=name, version=1).read()
                if "event_time" in frame.columns:
                    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
            except Exception:
                frame = pd.DataFrame()
        if frame.empty:
            frame = self._read_local(name)
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
        return self.model_dir / "metadata.json"

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

    def register_hopsworks(self, metrics: Dict[str, float]) -> None:
        if not (self.settings.hopsworks_api_key and self.settings.hopsworks_project):
            return
        import hopsworks

        project = hopsworks.login(
            project=self.settings.hopsworks_project,
            api_key_value=self.settings.hopsworks_api_key,
        )
        registry = project.get_model_registry()
        model = registry.python.create_model(MODEL_NAME, metrics=metrics)
        model.save(str(self.model_dir))
