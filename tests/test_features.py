from __future__ import annotations

import unittest

from aqi_predictor.services.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    current_weather_payload_to_frame,
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


if __name__ == "__main__":
    unittest.main()
