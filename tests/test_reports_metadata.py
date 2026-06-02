from __future__ import annotations

import json
from pathlib import Path


def _load_report(name: str) -> dict:
    return json.loads((Path("reports") / name).read_text(encoding="utf-8"))


def test_committed_model_and_explainability_reports_are_consistent():
    metrics = _load_report("model_metrics.json")
    shap = _load_report("shap_summary.json")
    importance = _load_report("feature_importance.json")

    assert metrics["registry_version"] == shap["registry_version"] == importance["registry_version"]
    assert metrics["model_name"] == shap["model_name"] == importance["model_name"]
    assert metrics["trained_at"] == shap["trained_at"] == importance["trained_at"]
    assert metrics["metrics"]["rmse"] == shap["metrics"]["rmse"]
    assert shap["method"] in {"shap", "model_native_feature_importance", "not_available"}
    if shap["method"] != "shap":
        assert shap["status"] in {"fallback", "skipped"}


def test_current_submission_reports_reference_deployed_registry_version_9():
    metrics = _load_report("model_metrics.json")

    assert metrics["registry_version"] == 9
    assert metrics["model_name"] == "ridge"
    assert metrics["training_rows"] == 717
    assert metrics["test_rows"] == 144
