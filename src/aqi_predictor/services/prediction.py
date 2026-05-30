from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.features import (
    attach_prediction_metadata,
    build_feature_frame,
    forecast_weather_payload_to_frame,
    json_records,
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

    def latest_features(self, limit: int = 24) -> pd.DataFrame:
        return self.feature_store.read_feature_group(FEATURES_FEATURE_GROUP, limit=limit)

    def _forecast_frame(self, horizon: int, sample: bool = False) -> pd.DataFrame:
        if sample or self.settings.use_sample_data:
            return generate_sample_forecast(self.settings, hours=horizon)
        if not self.openweather.configured:
            raise OpenWeatherError(
                "OPENWEATHER_API_KEY is not configured. Use sample=true only for offline demos."
            )

        pollution_payload = self.openweather.forecast_air_pollution()
        weather_payload = self.openweather.forecast_weather()
        pollution = pollution_payload_to_frame(
            pollution_payload, self.settings.city, self.settings.lat, self.settings.lon
        )
        weather = forecast_weather_payload_to_frame(weather_payload)
        frame = build_feature_frame(pollution, weather)
        frame = frame.sort_values("event_time").head(horizon).copy()
        frame["forecast_horizon"] = np.arange(1, len(frame) + 1)
        return frame

    def predict(self, horizon: Optional[int] = None, sample: bool = False) -> Dict[str, Any]:
        forecast_hours = horizon or self.settings.forecast_hours
        model, metadata = load_model_bundle(self.settings)
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
            prediction = float(predict_frame(model, row_frame)[0])
            prediction = float(np.clip(prediction, 0.0, 300.0))
            predictions.append(prediction)
            recent_values.append(prediction)
            rows.append(working)

        prediction_frame = pd.DataFrame(rows).reset_index(drop=True)
        prediction_frame = attach_prediction_metadata(prediction_frame, predictions)
        prediction_frame["city"] = self.settings.city
        prediction_frame["created_at"] = datetime.now(timezone.utc)

        records = json_records(prediction_frame)
        prediction_store = "hopsworks"
        try:
            self.feature_store.insert_feature_group(
                PREDICTIONS_FEATURE_GROUP,
                prediction_frame,
                primary_key=["city", "event_time"],
            )
        except Exception:
            prediction_store = "local"
        self.mongo_store.insert_predictions(records)

        return {
            "city": self.settings.city,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon_hours": int(len(records)),
            "data_source": data_source,
            "is_live": data_source == "openweather",
            "prediction_store": prediction_store,
            "model": {
                "name": metadata.get("model_name"),
                "trained_at": metadata.get("trained_at"),
                "metrics": metadata.get("metrics", {}),
            },
            "predictions": records,
        }

    def latest_payload(self) -> Dict[str, Any]:
        frame = self.latest_features(limit=1)
        if frame.empty:
            return {"city": self.settings.city, "latest": None}
        return {"city": self.settings.city, "latest": json_records(frame.tail(1))[0]}

    def alerts_payload(self, limit: int = 20) -> Dict[str, Any]:
        alerts = self.mongo_store.list_alerts(limit=limit)
        if not alerts:
            predictions = self.feature_store.read_feature_group(PREDICTIONS_FEATURE_GROUP, limit=200)
            if not predictions.empty and "alert_level" in predictions.columns:
                filtered = predictions[predictions["alert_level"].isin(["unhealthy", "hazardous"])]
                alerts = json_records(filtered.tail(limit))
        return {"city": self.settings.city, "alerts": alerts}

    def model_info_payload(self) -> Dict[str, Any]:
        model, metadata = load_model_bundle(self.settings)
        latest = self.latest_features(limit=200)
        importance = feature_importance(model, latest) if not latest.empty else []
        return {
            "city": self.settings.city,
            "model": metadata,
            "feature_importance": importance,
        }
