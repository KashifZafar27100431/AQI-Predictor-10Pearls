from __future__ import annotations

import unittest

import pandas as pd

from aqi_predictor.services.aqi_scale import alert_level, label_from_score
from aqi_predictor.services.features import (
    FEATURE_COLUMNS,
    add_time_features,
    attach_prediction_metadata,
    build_feature_frame,
    current_weather_payload_to_frame,
    json_records,
    merge_weather_features,
    pollution_payload_to_frame,
)


class FeatureEngineeringTests(unittest.TestCase):
    def test_openweather_payload_becomes_model_features(self) -> None:
        pollution_payload = {
            "list": [
                {
                    "dt": 1_700_000_000,
                    "main": {"aqi": 4},
                    "components": {
                        "co": 1200,
                        "no": 1.0,
                        "no2": 40.0,
                        "o3": 70.0,
                        "so2": 12.0,
                        "pm2_5": 80.0,
                        "pm10": 140.0,
                        "nh3": 5.0,
                    },
                }
            ]
        }
        weather_payload = {
            "dt": 1_700_000_000,
            "main": {"temp": 31.5, "feels_like": 35.0, "pressure": 1009, "humidity": 63},
            "wind": {"speed": 4.2, "deg": 220},
            "clouds": {"all": 20},
        }

        pollution = pollution_payload_to_frame(pollution_payload, "Karachi", 24.8607, 67.0011)
        weather = current_weather_payload_to_frame(weather_payload)
        features = build_feature_frame(pollution, weather)

        self.assertEqual(len(features), 1)
        self.assertEqual(float(features.loc[0, "aqi_score"]), 168.0)
        self.assertEqual(features.loc[0, "primary_pollutant"], "pm2_5")
        for column in FEATURE_COLUMNS:
            self.assertIn(column, features.columns)

    def test_aqi_category_boundaries_and_hazardous_alerts(self) -> None:
        self.assertEqual(label_from_score(50), "Good")
        self.assertEqual(label_from_score(51), "Moderate")
        self.assertEqual(label_from_score(100), "Moderate")
        self.assertEqual(label_from_score(101), "Unhealthy for Sensitive Groups")
        self.assertEqual(label_from_score(150), "Unhealthy for Sensitive Groups")
        self.assertEqual(label_from_score(151), "Unhealthy")
        self.assertEqual(label_from_score(200), "Unhealthy")
        self.assertEqual(label_from_score(201), "Very Unhealthy")
        self.assertEqual(label_from_score(300), "Very Unhealthy")
        self.assertEqual(label_from_score(301), "Hazardous")
        self.assertEqual(label_from_score(500), "Hazardous")
        self.assertEqual(alert_level(50), "normal")
        self.assertEqual(alert_level(51), "normal")
        self.assertEqual(alert_level(100), "normal")
        self.assertEqual(alert_level(101), "sensitive_groups")
        self.assertEqual(alert_level(150), "sensitive_groups")
        self.assertEqual(alert_level(151), "unhealthy")
        self.assertEqual(alert_level(200), "unhealthy")
        self.assertEqual(alert_level(201), "very_unhealthy")
        self.assertEqual(alert_level(300), "very_unhealthy")
        self.assertEqual(alert_level(301), "hazardous")
        self.assertEqual(alert_level(500), "hazardous")

        frame = pd.DataFrame({"event_time": [pd.Timestamp("2026-05-30T00:00:00Z")]})
        enriched = attach_prediction_metadata(frame, [301.0])
        self.assertEqual(enriched.loc[0, "aqi_category"], "Hazardous")
        self.assertEqual(enriched.loc[0, "alert_level"], "hazardous")

    def test_time_features_and_json_display_use_karachi_timezone(self) -> None:
        frame = pd.DataFrame({"event_time": [pd.Timestamp("2026-05-30T20:00:00Z")]})
        features = add_time_features(frame, timezone_name="Asia/Karachi")
        self.assertEqual(int(features.loc[0, "hour"]), 1)
        self.assertEqual(int(features.loc[0, "day"]), 31)

        records = json_records(features, timezone_name="Asia/Karachi")
        self.assertEqual(records[0]["event_time"], "2026-05-30T20:00:00Z")
        self.assertEqual(records[0]["event_time_local"], "2026-05-31T01:00:00+0500")
        self.assertEqual(records[0]["event_timezone"], "Asia/Karachi")

    def test_weather_merge_normalizes_timestamp_precision(self) -> None:
        pollution = pd.DataFrame(
            {
                "city": ["Karachi"],
                "event_time": pd.to_datetime(["2026-05-30T00:00:00Z"], utc=True).astype("datetime64[ns, UTC]"),
                "aqi_score": [75],
            }
        )
        weather = pd.DataFrame(
            {
                "event_time": pd.to_datetime(["2026-05-30T00:00:00Z"], utc=True).astype("datetime64[us, UTC]"),
                "temp": [31.0],
            }
        )

        merged = merge_weather_features(pollution, weather)

        self.assertEqual(float(merged.loc[0, "temp"]), 31.0)


if __name__ == "__main__":
    unittest.main()
