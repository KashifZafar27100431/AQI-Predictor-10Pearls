from __future__ import annotations

from pathlib import Path

from aqi_predictor.config.settings import Settings
from aqi_predictor.services import modeling
from aqi_predictor.services.sample_data import generate_sample_history


def test_tensorflow_candidate_can_be_selected_when_it_has_best_rmse(tmp_path, monkeypatch):
    settings = Settings(
        local_data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        hopsworks_api_key=None,
        hopsworks_project=None,
    )
    history = generate_sample_history(settings, hours=24 * 8)

    def fake_tensorflow_experiment(train, test, model_dir: Path, feature_columns):
        tf_dir = model_dir / "tensorflow_experiment"
        tf_dir.mkdir(parents=True, exist_ok=True)
        (tf_dir / "model.keras").write_text("placeholder", encoding="utf-8")
        (tf_dir / "scaler.joblib").write_bytes(b"placeholder")
        return {
            "name": "tensorflow_dense",
            "model_type": "tensorflow_keras",
            "metrics": {"rmse": 0.0, "mae": 0.0, "r2": 1.0},
            "path": str(tf_dir),
            "artifacts": {
                "keras_model": "tensorflow_experiment/model.keras",
                "scaler": "tensorflow_experiment/scaler.joblib",
            },
        }

    monkeypatch.setattr(modeling, "_tensorflow_experiment", fake_tensorflow_experiment)

    metadata = modeling.train_and_register(history, settings)

    assert metadata["model_type"] == "tensorflow_keras"
    assert metadata["model_name"] == "tensorflow_dense"
    assert metadata["metrics"]["rmse"] == 0.0
    assert metadata["artifacts"]["keras_model"] == "tensorflow_experiment/model.keras"
