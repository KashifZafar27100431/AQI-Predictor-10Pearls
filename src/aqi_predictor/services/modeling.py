from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aqi_predictor.config.settings import Settings
from aqi_predictor.services.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WEATHER_COLUMNS,
    add_lag_features,
    add_time_features,
    ensure_feature_columns,
    normalize_numeric_features,
    prepare_training_frame,
)
from aqi_predictor.services.storage import MODEL_ARTIFACT_FILE, MODEL_METADATA_FILE, ModelRegistryClient


logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when no trusted, schema-compatible model can be loaded."""


class TensorFlowModelBundle:
    def __init__(self, keras_model: Any, scaler: Any):
        self.keras_model = keras_model
        self.scaler = scaler

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(frame, dtype=float)
        scaled = self.scaler.transform(values)
        return np.asarray(self.keras_model.predict(scaled, verbose=0), dtype=float).ravel()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
    }


def _temporal_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("event_time") if "event_time" in frame.columns else frame.copy()
    split_index = max(1, int(len(ordered) * (1 - test_fraction)))
    split_index = min(split_index, len(ordered) - 1)
    return ordered.iloc[:split_index], ordered.iloc[split_index:]


def _candidate_models() -> Dict[str, Any]:
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=0.01,
            random_state=42,
        ),
    }


def _baseline_metrics(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    y_true = test[TARGET_COLUMN].to_numpy(dtype=float)
    persistence = test["aqi_lag_1h"].to_numpy(dtype=float)
    moving_average = test["aqi_rolling_24h"].to_numpy(dtype=float)
    train_mean = np.repeat(float(train[TARGET_COLUMN].mean()), len(test))
    return {
        "baseline_persistence": regression_metrics(y_true, persistence),
        "baseline_moving_average": regression_metrics(y_true, moving_average),
        "baseline_train_mean": regression_metrics(y_true, train_mean),
    }


def _tensorflow_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_dir: Path,
    feature_columns: List[str],
) -> Optional[Dict[str, Any]]:
    if len(train) < 96:
        return None
    try:
        import tensorflow as tf
    except Exception:
        return None

    x_train = ensure_feature_columns(train, feature_columns).to_numpy(dtype=float)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = ensure_feature_columns(test, feature_columns).to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.fit(x_train_scaled, y_train, epochs=25, batch_size=32, verbose=0)
    predictions = model.predict(x_test_scaled, verbose=0).ravel()
    metrics = regression_metrics(y_test, predictions)

    tf_dir = model_dir / "tensorflow_experiment"
    tf_dir.mkdir(parents=True, exist_ok=True)
    model.save(tf_dir / "model.keras")
    joblib.dump(scaler, tf_dir / "scaler.joblib")
    return {
        "name": "tensorflow_dense",
        "model_type": "tensorflow_keras",
        "metrics": metrics,
        "path": str(tf_dir),
        "artifacts": {
            "keras_model": "tensorflow_experiment/model.keras",
            "scaler": "tensorflow_experiment/scaler.joblib",
        },
    }


def _select_training_feature_columns(frame: pd.DataFrame, settings: Settings) -> Tuple[List[str], Dict[str, Any]]:
    inspected = normalize_numeric_features(
        add_lag_features(add_time_features(frame, timezone_name=settings.timezone))
    )
    missingness: Dict[str, float] = {}
    excluded: List[str] = []
    for column in WEATHER_COLUMNS:
        if column not in inspected.columns or inspected.empty:
            fraction = 1.0
        else:
            fraction = float(inspected[column].isna().mean())
        missingness[column] = fraction
        if fraction > settings.max_weather_missing_fraction:
            excluded.append(column)

    feature_columns = [column for column in FEATURE_COLUMNS if column not in excluded]
    return feature_columns, {
        "all_feature_columns": FEATURE_COLUMNS,
        "feature_columns": feature_columns,
        "excluded_feature_columns": excluded,
        "weather_missing_fraction": missingness,
        "weather_missing_threshold": settings.max_weather_missing_fraction,
    }


def train_and_register(frame: pd.DataFrame, settings: Settings) -> Dict[str, Any]:
    feature_columns, schema = _select_training_feature_columns(frame, settings)
    training = prepare_training_frame(frame, timezone_name=settings.timezone)
    if len(training) < 24:
        raise ValueError("At least 24 hourly feature rows are required for training.")

    train, test = _temporal_split(training)
    x_train = ensure_feature_columns(train, feature_columns)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = ensure_feature_columns(test, feature_columns)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    candidate_metrics: Dict[str, Dict[str, float]] = _baseline_metrics(train, test)
    best_name: Optional[str] = None
    best_sklearn_model: Optional[Any] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_model_type = "sklearn"

    for name, estimator in _candidate_models().items():
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        metrics = regression_metrics(y_test, predictions)
        candidate_metrics[name] = metrics
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name = name
            best_sklearn_model = estimator
            best_metrics = metrics
            best_model_type = "sklearn"

    registry = ModelRegistryClient(settings)
    registry.model_dir.mkdir(parents=True, exist_ok=True)
    tensorflow_result = _tensorflow_experiment(train, test, registry.model_dir, feature_columns)
    if tensorflow_result is not None:
        candidate_metrics[tensorflow_result["name"]] = tensorflow_result["metrics"]
        if best_metrics is None or tensorflow_result["metrics"]["rmse"] < best_metrics["rmse"]:
            best_name = tensorflow_result["name"]
            best_metrics = tensorflow_result["metrics"]
            best_model_type = "tensorflow_keras"

    if best_sklearn_model is None or best_metrics is None or best_name is None:
        raise RuntimeError("No model candidate completed successfully.")

    joblib.dump(best_sklearn_model, registry.model_dir / MODEL_ARTIFACT_FILE)
    metadata = {
        "model_type": best_model_type,
        "model_name": best_name,
        "target": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "feature_schema": schema,
        "artifacts": (
            tensorflow_result["artifacts"]
            if best_model_type == "tensorflow_keras" and tensorflow_result is not None
            else {"sklearn_model": MODEL_ARTIFACT_FILE}
        ),
        "metrics": best_metrics,
        "candidate_metrics": candidate_metrics,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_rows": int(len(training)),
        "test_rows": int(len(test)),
        "tensorflow_experiment": tensorflow_result,
    }
    registry.save_metadata(metadata)
    try:
        metadata["registry"] = registry.register_hopsworks(best_metrics)
    except Exception as exc:
        if settings.require_hopsworks_model_registry:
            raise
        logger.warning("Model registry registration failed; local artifact remains available.")
        metadata["registry"] = {"registered": False, "error_type": type(exc).__name__}
    registry.save_metadata(metadata)
    return metadata


def load_model_bundle(settings: Settings) -> Tuple[Any, Dict[str, Any]]:
    registry = ModelRegistryClient(settings)
    registry_error: Optional[Exception] = None
    if settings.hopsworks_api_key and settings.hopsworks_project:
        try:
            downloaded = registry.download_latest_hopsworks_model()
            return _load_trusted_model(
                downloaded["artifact_dir"],
                downloaded["metadata_path"],
                downloaded["artifact_dir"],
                source="hopsworks_model_registry",
                registry_version=downloaded.get("version"),
            )
        except Exception as exc:
            registry_error = exc
            logger.warning("Hopsworks model registry load failed; evaluating local fallback.")

    if not settings.allow_local_model_fallback:
        message = "No valid Hopsworks model is available and local model fallback is disabled."
        raise ModelLoadError(message) from registry_error

    try:
        metadata_path = registry.model_dir / MODEL_METADATA_FILE
        model, metadata = _load_trusted_model(
            registry.model_dir,
            metadata_path,
            registry.model_dir,
            source="local_model_dir",
            registry_version=None,
        )
        logger.info("Loaded AQI model from explicit local development fallback.")
        return model, metadata
    except Exception as exc:
        raise ModelLoadError("No valid AQI forecasting model is available.") from (registry_error or exc)


def _load_trusted_model(
    artifact_dir: Path,
    metadata_path: Path,
    trusted_root: Path,
    source: str,
    registry_version: Optional[int],
) -> Tuple[Any, Dict[str, Any]]:
    artifact_dir = _safe_artifact_path(artifact_dir, trusted_root)
    metadata_path = _safe_artifact_path(metadata_path, trusted_root)
    if metadata_path.name != MODEL_METADATA_FILE:
        raise ModelLoadError("Unexpected model metadata filename.")
    if not artifact_dir.exists() or not artifact_dir.is_dir() or not metadata_path.exists():
        raise ModelLoadError("Model artifact directory or metadata is missing.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model_type, feature_columns = _validate_model_metadata(metadata)
    if model_type == "sklearn":
        model = _load_sklearn_model(artifact_dir, metadata)
    elif model_type == "tensorflow_keras":
        model = _load_tensorflow_model(artifact_dir, metadata)
    else:
        raise ModelLoadError(f"Unsupported model type: {model_type}")
    _validate_estimator_schema(model, feature_columns)
    enriched = dict(metadata)
    enriched["serving_source"] = source
    enriched["registry_version"] = registry_version
    enriched["loaded_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    enriched["feature_count"] = len(feature_columns)
    return model, enriched


def _load_sklearn_model(artifact_dir: Path, metadata: Dict[str, Any]) -> Any:
    artifacts = metadata.get("artifacts", {})
    relative_path = artifacts.get("sklearn_model", MODEL_ARTIFACT_FILE) if isinstance(artifacts, dict) else MODEL_ARTIFACT_FILE
    model_path = _safe_artifact_path(artifact_dir / relative_path, artifact_dir)
    if model_path.name != MODEL_ARTIFACT_FILE or not model_path.exists():
        raise ModelLoadError("Trusted Scikit-learn model artifact is missing.")
    return joblib.load(model_path)


def _load_tensorflow_model(artifact_dir: Path, metadata: Dict[str, Any]) -> TensorFlowModelBundle:
    artifacts = metadata.get("artifacts", {})
    keras_relative = "tensorflow_experiment/model.keras"
    scaler_relative = "tensorflow_experiment/scaler.joblib"
    if isinstance(artifacts, dict):
        keras_relative = artifacts.get("keras_model", keras_relative)
        scaler_relative = artifacts.get("scaler", scaler_relative)
    keras_path = _safe_artifact_path(artifact_dir / keras_relative, artifact_dir)
    scaler_path = _safe_artifact_path(artifact_dir / scaler_relative, artifact_dir)
    if keras_path.name != "model.keras" or not keras_path.exists():
        raise ModelLoadError("Trusted TensorFlow model artifact is missing.")
    if scaler_path.name != "scaler.joblib" or not scaler_path.exists():
        raise ModelLoadError("Trusted TensorFlow scaler artifact is missing.")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ModelLoadError("TensorFlow model selected, but TensorFlow is not installed.") from exc
    return TensorFlowModelBundle(tf.keras.models.load_model(str(keras_path)), joblib.load(scaler_path))


def _safe_artifact_path(path: Path, trusted_root: Path) -> Path:
    trusted = trusted_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(trusted)
    except ValueError as exc:
        raise ModelLoadError("Model artifact path is outside the trusted model directory.") from exc
    return resolved


def _validate_model_metadata(metadata: Dict[str, Any]) -> Tuple[str, List[str]]:
    model_type = metadata.get("model_type")
    if model_type not in {"sklearn", "tensorflow_keras"}:
        raise ModelLoadError(f"Unsupported model type: {model_type}")
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ModelLoadError("Model metadata is missing a non-empty feature column schema.")
    unknown = sorted(set(feature_columns) - set(FEATURE_COLUMNS))
    if unknown:
        raise ModelLoadError(f"Model metadata contains unknown feature columns: {unknown}")
    return model_type, feature_columns


def _validate_estimator_schema(model: Any, feature_columns: List[str]) -> None:
    fitted_features = getattr(model, "n_features_in_", None)
    if fitted_features is not None and int(fitted_features) != len(feature_columns):
        raise ModelLoadError(
            f"Model expects {fitted_features} features but metadata declares {len(feature_columns)}."
        )


def predict_frame(model: Any, frame: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:
    features = ensure_feature_columns(frame, feature_columns)
    return np.asarray(model.predict(features), dtype=float)


def feature_importance(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    limit: int = 12,
) -> List[Dict[str, float]]:
    selected_columns = feature_columns or FEATURE_COLUMNS
    sample = ensure_feature_columns(frame, selected_columns).tail(min(len(frame), 200))
    if sample.empty:
        return []

    try:
        import shap

        explainer = shap.Explainer(model, sample)
        values = explainer(sample)
        mean_abs = np.abs(values.values).mean(axis=0)
        pairs = sorted(zip(selected_columns, mean_abs), key=lambda item: item[1], reverse=True)
        return [{"feature": name, "importance": float(value)} for name, value in pairs[:limit]]
    except Exception:
        pass

    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_)
    else:
        return []
    pairs = sorted(zip(selected_columns, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(value)} for name, value in pairs[:limit]]


def write_training_report(settings: Settings, metadata: Dict[str, Any]) -> Path:
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "model_metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return path
