from __future__ import annotations

from datetime import datetime, timezone
import json
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
from aqi_predictor.services.features import FEATURE_COLUMNS, TARGET_COLUMN, ensure_feature_columns, prepare_training_frame
from aqi_predictor.services.storage import ModelRegistryClient


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
) -> Optional[Dict[str, Any]]:
    if len(train) < 96:
        return None
    try:
        import tensorflow as tf
    except Exception:
        return None

    x_train = ensure_feature_columns(train).to_numpy(dtype=float)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = ensure_feature_columns(test).to_numpy(dtype=float)
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
    return {"name": "tensorflow_dense", "metrics": metrics, "path": str(tf_dir)}


def train_and_register(frame: pd.DataFrame, settings: Settings) -> Dict[str, Any]:
    training = prepare_training_frame(frame)
    if len(training) < 24:
        raise ValueError("At least 24 hourly feature rows are required for training.")

    train, test = _temporal_split(training)
    x_train = ensure_feature_columns(train)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = ensure_feature_columns(test)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    candidate_metrics: Dict[str, Dict[str, float]] = _baseline_metrics(train, test)
    best_name: Optional[str] = None
    best_model: Optional[Any] = None
    best_metrics: Optional[Dict[str, float]] = None

    for name, estimator in _candidate_models().items():
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        metrics = regression_metrics(y_test, predictions)
        candidate_metrics[name] = metrics
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name = name
            best_model = estimator
            best_metrics = metrics

    registry = ModelRegistryClient(settings)
    registry.model_dir.mkdir(parents=True, exist_ok=True)
    tensorflow_result = _tensorflow_experiment(train, test, registry.model_dir)
    if tensorflow_result is not None:
        candidate_metrics[tensorflow_result["name"]] = tensorflow_result["metrics"]

    if best_model is None or best_metrics is None or best_name is None:
        raise RuntimeError("No model candidate completed successfully.")

    joblib.dump(best_model, registry.model_dir / "model.joblib")
    metadata = {
        "model_type": "sklearn",
        "model_name": best_name,
        "target": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": best_metrics,
        "candidate_metrics": candidate_metrics,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_rows": int(len(training)),
        "test_rows": int(len(test)),
        "tensorflow_experiment": tensorflow_result,
    }
    registry.save_metadata(metadata)
    registry.register_hopsworks(best_metrics)
    return metadata


def load_model_bundle(settings: Settings) -> Tuple[Any, Dict[str, Any]]:
    registry = ModelRegistryClient(settings)
    metadata = registry.load_metadata()
    if metadata.get("model_type") != "sklearn":
        raise ValueError(f"Unsupported model type: {metadata.get('model_type')}")
    model_path = registry.model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    return joblib.load(model_path), metadata


def predict_frame(model: Any, frame: pd.DataFrame) -> np.ndarray:
    features = ensure_feature_columns(frame)
    return np.asarray(model.predict(features), dtype=float)


def feature_importance(model: Any, frame: pd.DataFrame, limit: int = 12) -> List[Dict[str, float]]:
    sample = ensure_feature_columns(frame).tail(min(len(frame), 200))
    if sample.empty:
        return []

    try:
        import shap

        explainer = shap.Explainer(model, sample)
        values = explainer(sample)
        mean_abs = np.abs(values.values).mean(axis=0)
        pairs = sorted(zip(FEATURE_COLUMNS, mean_abs), key=lambda item: item[1], reverse=True)
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
    pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(value)} for name, value in pairs[:limit]]


def write_training_report(settings: Settings, metadata: Dict[str, Any]) -> Path:
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "model_metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return path
