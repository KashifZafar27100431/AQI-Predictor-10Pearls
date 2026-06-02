from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.features import (
    attach_prediction_metadata,
    build_feature_frame,
    forecast_weather_payload_to_frame,
    json_records,
    localize_event_time,
    pollution_payload_to_frame,
    recent_history_stats,
)
from aqi_predictor.services.modeling import feature_importance, load_model_bundle, predict_frame
from aqi_predictor.services.openweather import OpenWeatherClient, OpenWeatherError
from aqi_predictor.services.sample_data import generate_sample_forecast
from aqi_predictor.services.storage import (
    FEATURES_FEATURE_GROUP,
    PREDICTIONS_FEATURE_GROUP,
    FeatureStoreClient,
    MongoStore,
)


logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(
        self,
        settings: Settings,
        feature_store: Optional[FeatureStoreClient] = None,
        mongo_store: Optional[MongoStore] = None,
    ):
        self.settings = settings
        self.feature_store = feature_store or FeatureStoreClient(settings)
        self.mongo_store = mongo_store or MongoStore(settings)
        self.openweather = OpenWeatherClient(settings)
        self._model_bundle: Optional[Tuple[Any, Dict[str, Any]]] = None

    def latest_features(self, limit: int = 24) -> pd.DataFrame:
        try:
            return self.feature_store.read_feature_group(FEATURES_FEATURE_GROUP, limit=limit)
        except Exception:
            logger.warning("feature_store_read_failed feature_group=%s", FEATURES_FEATURE_GROUP, exc_info=True)
            raise

    def _load_model_bundle(self) -> Tuple[Any, Dict[str, Any]]:
        if self._model_bundle is None:
            self._model_bundle = load_model_bundle(self.settings)
        return self._model_bundle

    def model_metadata(self) -> Dict[str, Any]:
        _, metadata = self._load_model_bundle()
        return metadata

    def _forecast_frame(self, horizon: int, sample: bool = False) -> pd.DataFrame:
        if sample or self.settings.use_sample_data:
            return generate_sample_forecast(self.settings, hours=horizon)
        if not self.openweather.configured:
            raise OpenWeatherError(
                "OPENWEATHER_API_KEY is not configured. Use sample=true only for offline demos."
            )

        try:
            pollution_payload = self.openweather.forecast_air_pollution()
            weather_payload = self.openweather.forecast_weather()
        except OpenWeatherError:
            logger.warning("openweather_forecast_failed city=%s", self.settings.city, exc_info=True)
            raise
        pollution = pollution_payload_to_frame(
            pollution_payload, self.settings.city, self.settings.lat, self.settings.lon
        )
        weather = forecast_weather_payload_to_frame(weather_payload)
        frame = build_feature_frame(pollution, weather, timezone_name=self.settings.timezone)
        frame = frame.sort_values("event_time").head(horizon).copy()
        frame["forecast_horizon"] = np.arange(1, len(frame) + 1)
        return frame

    def predict(
        self,
        horizon: Optional[int] = None,
        sample: bool = False,
        store_predictions: bool = True,
    ) -> Dict[str, Any]:
        forecast_hours = self._clamp_horizon(horizon or self.settings.forecast_hours)
        model, metadata = self._load_model_bundle()
        feature_columns = metadata.get("feature_columns")
        data_source = "sample" if sample or self.settings.use_sample_data else "openweather"
        forecast = self._forecast_frame(forecast_hours, sample=sample)
        history = self.latest_features(limit=48)
        stats = recent_history_stats(history)

        recent_values: List[float] = []
        if history is not None and not history.empty and "aqi_score" in history.columns:
            recent_values = (
                pd.to_numeric(history.sort_values("event_time")["aqi_score"], errors="coerce")
                .dropna()
                .tail(24)
                .astype(float)
                .tolist()
            )
        if not recent_values:
            recent_values = [stats["aqi_lag_1h"]]

        predictions: List[float] = []
        rows = []
        for index, (_, row) in enumerate(forecast.iterrows(), start=1):
            working = row.copy()
            previous = recent_values[-1]
            before_previous = recent_values[-2] if len(recent_values) > 1 else previous
            working["forecast_horizon"] = index
            working["aqi_lag_1h"] = previous
            working["aqi_rolling_3h"] = float(np.mean(recent_values[-3:]))
            working["aqi_rolling_24h"] = float(np.mean(recent_values[-24:]))
            working["aqi_change_rate"] = previous - before_previous
            row_frame = pd.DataFrame([working])
            prediction = float(predict_frame(model, row_frame, feature_columns=feature_columns)[0])
            prediction = float(np.clip(prediction, 0.0, 500.0))
            predictions.append(prediction)
            recent_values.append(prediction)
            rows.append(working)

        prediction_frame = pd.DataFrame(rows).reset_index(drop=True)
        prediction_frame = attach_prediction_metadata(prediction_frame, predictions)
        prediction_frame["city"] = self.settings.city
        prediction_frame["created_at"] = datetime.now(timezone.utc)

        records = json_records(prediction_frame, timezone_name=self.settings.timezone)
        prediction_store = "not_persisted"
        if store_predictions:
            prediction_store = (
                "hopsworks"
                if self.settings.hopsworks_api_key and self.settings.hopsworks_project
                else "local"
            )
            try:
                self.feature_store.insert_feature_group(
                    PREDICTIONS_FEATURE_GROUP,
                    prediction_frame,
                    primary_key=["city", "event_time"],
                )
            except Exception:
                prediction_store = "local"
                logger.warning("Prediction feature-store insert failed; continuing with local response only.")
            self.mongo_store.insert_predictions(records)

        generated_at = datetime.now(timezone.utc)
        generated_at_local = (
            pd.Series([generated_at])
            .dt.tz_convert(self.settings.timezone)
            .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            .iloc[0]
        )
        return {
            "city": self.settings.city,
            "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generated_at_local": generated_at_local,
            "timezone": self.settings.timezone,
            "horizon_hours": int(len(records)),
            "data_source": data_source,
            "is_live": data_source == "openweather",
            "prediction_store": prediction_store,
            "model": {
                "name": metadata.get("model_name"),
                "trained_at": metadata.get("trained_at"),
                "metrics": metadata.get("metrics", {}),
                "serving_source": metadata.get("serving_source"),
                "registry_version": metadata.get("registry_version"),
                "feature_count": metadata.get("feature_count"),
            },
            "predictions": records,
        }

    def latest_payload(self) -> Dict[str, Any]:
        frame = self.latest_features(limit=1)
        if frame.empty:
            return {"city": self.settings.city, "latest": None}
        return {"city": self.settings.city, "latest": json_records(frame.tail(1), self.settings.timezone)[0]}

    def alerts_payload(self, limit: int = 20, include_forecast: bool = True) -> Dict[str, Any]:
        limit = max(1, min(int(limit), self.settings.max_api_limit))
        source = "stored_alerts"
        alerts = self.mongo_store.list_alerts(limit=limit)
        if not alerts:
            predictions = self.feature_store.read_feature_group(PREDICTIONS_FEATURE_GROUP, limit=200)
            if not predictions.empty and "alert_level" in predictions.columns:
                filtered = predictions[~predictions["alert_level"].isin(["normal", None])]
                alerts = json_records(filtered.tail(limit), self.settings.timezone)
                source = "prediction_feature_group"
        if not alerts and include_forecast:
            try:
                forecast = self.predict(
                    horizon=self.settings.forecast_hours,
                    sample=False,
                    store_predictions=False,
                )
                forecast_alerts = [
                    record
                    for record in forecast.get("predictions", [])
                    if record.get("alert_level") not in {None, "normal"}
                ]
                alerts = forecast_alerts[:limit]
                source = "current_forecast"
            except Exception:
                logger.warning("forecast_alert_generation_failed city=%s", self.settings.city, exc_info=True)
        return {
            "city": self.settings.city,
            "alerts": alerts,
            "source": source if alerts else "none",
            "reason": None if alerts else "no_forecast_alerts",
        }

    def model_info_payload(self) -> Dict[str, Any]:
        model, metadata = self._load_model_bundle()
        latest = self.latest_features(limit=200)
        feature_columns = metadata.get("feature_columns")
        explainability = metadata.get("explainability", {})
        importance = []
        if isinstance(explainability, dict):
            importance = explainability.get("top_features", []) or []
        if not importance and not latest.empty:
            importance = feature_importance(model, latest, feature_columns=feature_columns)
        freshness = self._feature_freshness(latest)
        latest_event_time = freshness.get("latest_feature_event_time")
        latest_event_time_local = freshness.get("latest_feature_event_time_local")
        if latest_event_time:
            latest_event_time = pd.Timestamp(latest_event_time)
        return {
            "city": self.settings.city,
            "model": metadata,
            "data_freshness": freshness,
            "feature_importance": importance,
            "explainability": explainability if isinstance(explainability, dict) else {},
        }

    def _feature_freshness(self, latest: pd.DataFrame) -> Dict[str, Any]:
        latest_event_time = None
        latest_event_time_local = None
        age_minutes = None
        status = "unknown"
        if not latest.empty and "event_time" in latest.columns:
            latest_event_time = pd.to_datetime(latest["event_time"], utc=True).max()
            now = pd.Timestamp(datetime.now(timezone.utc))
            age_minutes = max(0.0, float((now - latest_event_time).total_seconds() / 60.0))
            status = "fresh" if age_minutes <= 180 else "stale"
            latest_event_time_local = (
                localize_event_time(pd.Series([latest_event_time]), self.settings.timezone)
                .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                .iloc[0]
            )
        return {
            "latest_feature_event_time": latest_event_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            if latest_event_time is not None
            else None,
            "latest_feature_event_time_local": latest_event_time_local,
            "latest_feature_age_minutes": age_minutes,
            "status": status,
            "timezone": self.settings.timezone,
        }

    def _clamp_horizon(self, horizon: int) -> int:
        return max(1, min(int(horizon), self.settings.max_forecast_hours))
