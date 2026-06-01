from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import aqi_predictor.services.modeling as modeling
from aqi_predictor.services.features import FEATURE_COLUMNS, TARGET_COLUMN
from aqi_predictor.services.modeling import write_explainability_artifacts


def _training_frame(rows: int = 12) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            column: np.linspace(index, index + rows - 1, rows)
            for index, column in enumerate(FEATURE_COLUMNS, start=1)
        }
    )
    frame[TARGET_COLUMN] = np.linspace(50, 80, rows)
    frame["event_time"] = pd.date_range("2026-05-31T00:00:00Z", periods=rows, freq="h")
    return frame


def test_training_writes_precomputed_explainability_artifacts(tmp_path, monkeypatch):
    frame = _training_frame()
    model = Ridge().fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN])

    def fake_shap_importance(model, sample, feature_columns, limit):
        return [{"feature": "pm2_5", "importance": 3.0}]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modeling, "_shap_feature_importance", fake_shap_importance)
    summary = write_explainability_artifacts(
        model,
        frame.iloc[:8],
        frame.iloc[8:],
        FEATURE_COLUMNS,
        tmp_path / "models",
        {
            "model_name": "ridge",
            "model_type": "sklearn",
            "trained_at": "2026-05-31T00:00:00Z",
            "metrics": {"rmse": 0.1, "mae": 0.1, "r2": 0.99},
        },
    )

    model_summary_path = tmp_path / "models" / "explainability" / "shap_summary.json"
    report_summary_path = tmp_path / "reports" / "shap_summary.json"
    report_importance_path = tmp_path / "reports" / "feature_importance.json"

    assert summary["method"] == "shap"
    assert summary["status"] == "ok"
    assert summary["top_features"][0]["feature"] == "pm2_5"
    assert model_summary_path.exists()
    assert report_summary_path.exists()
    assert report_importance_path.exists()
    saved = json.loads(report_summary_path.read_text(encoding="utf-8"))
    assert saved["scope"] == "precomputed_training_holdout_sample"
