from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.sample_data import generate_sample_history
from aqi_predictor.services.modeling import train_and_register
from aqi_predictor.services.prediction import PredictionService
from aqi_predictor.services.storage import FEATURES_FEATURE_GROUP, FeatureStoreClient


class TrainingPredictionTests(unittest.TestCase):
    def test_training_and_sample_prediction_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(
                local_data_dir=root / "data",
                model_dir=root / "models",
                use_sample_data=True,
            )
            store = FeatureStoreClient(settings)
            history = generate_sample_history(settings, hours=24 * 10)
            store.insert_feature_group(FEATURES_FEATURE_GROUP, history, primary_key=["city", "event_time"])

            metadata = train_and_register(history, settings)
            self.assertIn("metrics", metadata)
            self.assertTrue((settings.model_dir / "model.joblib").exists())

            payload = PredictionService(settings, feature_store=store).predict(horizon=12, sample=True)
            self.assertEqual(payload["horizon_hours"], 12)
            self.assertEqual(len(payload["predictions"]), 12)
            self.assertIn("predicted_aqi_score", payload["predictions"][0])

            service = PredictionService(settings, feature_store=store)
            short_payload = service.predict(horizon=24, sample=True)
            long_payload = service.predict(horizon=96, sample=True)
            self.assertAlmostEqual(
                short_payload["predictions"][0]["predicted_aqi_score"],
                long_payload["predictions"][0]["predicted_aqi_score"],
                places=8,
            )


if __name__ == "__main__":
    unittest.main()
