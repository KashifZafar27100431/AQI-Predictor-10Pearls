from __future__ import annotations

from pathlib import Path
import json

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.features import prepare_training_frame
from aqi_predictor.services.modeling import _temporal_split, load_model_bundle, write_explainability_artifacts
from aqi_predictor.services.prediction import PredictionService
from aqi_predictor.services.storage import FEATURES_FEATURE_GROUP


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def _fmt_metric(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return "unknown" if value is None else f"{float(value):.3f}"


def _write_project_report(metadata: dict) -> Path:
    metrics = metadata.get("metrics", {})
    schema = metadata.get("feature_schema", {})
    excluded = schema.get("excluded_feature_columns", [])
    missing = schema.get("weather_missing_fraction", {})
    registry = metadata.get("registry", {})
    registry_version = metadata.get("registry_version") or registry.get("version")
    explainability = metadata.get("explainability", {})
    top_features = explainability.get("top_features", [])[:8]
    candidate_metrics = metadata.get("candidate_metrics", {})

    candidate_lines = []
    for name, values in candidate_metrics.items():
        candidate_lines.append(
            f"- {name}: RMSE {_fmt_metric(values, 'rmse')}, "
            f"MAE {_fmt_metric(values, 'mae')}, R² {_fmt_metric(values, 'r2')}."
        )
    if not candidate_lines:
        candidate_lines.append("- Candidate metrics are unavailable in the downloaded registry metadata.")

    feature_lines = [f"- {item['feature']}: {float(item['importance']):.4f}" for item in top_features]
    if not feature_lines:
        feature_lines.append("- No explainability features were available.")

    weather_lines = [f"- {name}: {float(value):.3f}" for name, value in sorted(missing.items())]
    if not weather_lines:
        weather_lines.append("- Weather missingness metadata is unavailable.")

    content = f"""# Pearls AQI Predictor Report

## Objective

Predict Karachi AQI for the next 72 hours using a serverless ML pipeline with automated feature ingestion, model training, model registry storage, Flask API serving, and an interactive Streamlit dashboard.

## Implemented System

- Feature pipeline fetches OpenWeather air pollution and weather data, computes time, pollutant, weather, lag, rolling, and AQI-change features, and writes them to Hopsworks-compatible feature groups.
- Backfill pipeline creates historical training rows from OpenWeather air-pollution history and joins cached weather rows when available.
- Training pipeline evaluates persistence, moving-average, Ridge Regression, Random Forest, HistGradientBoosting, and an optional TensorFlow dense model experiment.
- Prediction serving loads the latest approved or latest available Hopsworks-registered model artifact first; local model fallback is disabled in production.
- Flask exposes health, diagnostics, prediction, latest, alert, and model-info endpoints.
- Streamlit displays current AQI, 72-hour forecast, pollutant trends, unhealthy alerts, model metrics, and precomputed explainability.

## Latest Registered Model

- Model registry: `karachi_aqi_predictor`.
- Latest verified registry version: {registry_version or "unknown"}.
- Selected model: {metadata.get("model_name", "unknown")}.
- Model type: {metadata.get("model_type", "unknown")}.
- Trained at: {metadata.get("trained_at", "unknown")}.
- Training rows: {metadata.get("training_rows", "unknown")}.
- Test rows: {metadata.get("test_rows", "unknown")}.
- RMSE: {_fmt_metric(metrics, "rmse")}.
- MAE: {_fmt_metric(metrics, "mae")}.
- R²: {_fmt_metric(metrics, "r2")}.

## Candidate Metrics

{chr(10).join(candidate_lines)}

## Feature Engineering

Core model inputs include pollutants, Karachi-local time features, lag features, rolling AQI averages, AQI change rate, and prediction horizon. Weather columns are included only when historical coverage is reliable enough for training.

Excluded high-missingness weather columns:

{chr(10).join(f"- {name}" for name in excluded) if excluded else "- None."}

Weather missingness observed during training:

{chr(10).join(weather_lines)}

## Explainability

- Method: {explainability.get("method", "unknown")}.
- Status: {explainability.get("status", "unknown")}.
- Computed at: {explainability.get("computed_at", "unknown")}.
- Scope: {explainability.get("scope", "unknown")}.

Top contributing features:

{chr(10).join(feature_lines)}

## Data Quality Notes

- AQI score is OpenWeather-derived from pollutant concentrations using EPA-style breakpoints. It is not an official government regulatory Karachi AQI station feed.
- Historical OpenWeather weather access is plan-dependent. Cached weather rows are joined when available, but sparse high-missingness weather fields are excluded from training to avoid misleading features.
- Live prediction still uses OpenWeather forecast weather and forecast pollutant inputs where available.
- GitHub Actions scheduled workflows are best-effort and may drift; the hourly workflow is configured hourly but is not guaranteed to execute exactly every hour.

## Deployment

- Streamlit Community Cloud serves `app/streamlit_app.py`.
- Vercel serves the lightweight Flask API through `api/index.py` and `vercel.json`.
- Cloud Run remains the recommended fallback API target if Vercel hits dependency, memory, timeout, or TensorFlow-serving limits.
"""
    path = Path("reports/project_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _read_local_feature_cache(settings) -> object:
    path = settings.local_data_dir / f"{FEATURES_FEATURE_GROUP}.csv"
    if not path.exists():
        return None
    try:
        import pandas as pd

        frame = pd.read_csv(path)
        if "event_time" in frame.columns:
            frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        return frame
    except Exception:
        return None


def run() -> dict:
    _load_dotenv()
    settings = get_settings()
    service = PredictionService(settings)
    model, metadata = load_model_bundle(settings)
    local_features = _read_local_feature_cache(settings)
    features = local_features if local_features is not None and len(local_features) >= 48 else service.latest_features()
    if not features.empty:
        feature_columns = metadata.get("feature_columns", [])
        training = prepare_training_frame(features, timezone_name=settings.timezone)
        train, test = _temporal_split(training)
        metadata["explainability"] = write_explainability_artifacts(
            model,
            train,
            test,
            feature_columns,
            settings.model_dir,
            metadata,
        )
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    model_metrics = report_dir / "model_metrics.json"
    model_metrics.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    project_report = _write_project_report(metadata)
    return {"status": "ok", "model_metrics": str(model_metrics), "project_report": str(project_report)}


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
