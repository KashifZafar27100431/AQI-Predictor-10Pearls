# Pearls AQI Predictor Report

## Objective

Predict Karachi AQI for the next 72 hours using a serverless ML pipeline with automated feature ingestion, model training, model registry storage, Flask API serving, and an interactive Streamlit dashboard.

## Implemented System

- Feature pipeline fetches OpenWeather air pollution and weather data, computes time, pollutant, weather, lag, rolling, and AQI-change features, and writes them to Hopsworks-compatible feature groups.
- Backfill pipeline creates historical training rows from OpenWeather air-pollution history and joins cached weather rows when available.
- Training pipeline evaluates persistence, moving-average, Ridge Regression, Random Forest, HistGradientBoosting, and an optional TensorFlow dense model experiment.
- Prediction serving loads the pinned Hopsworks model version when `HOPSWORKS_MODEL_VERSION` is set, otherwise the latest approved or latest available registered artifact; local model fallback is disabled in production.
- Flask exposes health, readiness, diagnostics, prediction, latest, alert, and model-info endpoints.
- Streamlit displays current AQI, 72-hour forecast, pollutant trends, unhealthy alerts, model metrics, and precomputed explainability.

## Latest Registered Model

- Model registry: `karachi_aqi_predictor`.
- Latest verified registry version: 9.
- Temporary production pin: set `HOPSWORKS_MODEL_VERSION=9` on Vercel and Streamlit Cloud until latest-model selection is proven stable across cold starts.
- Selected model: ridge.
- Model type: sklearn.
- Trained at: 2026-06-02T05:02:32Z.
- Training rows: 717.
- Test rows: 144.
- RMSE: 0.210.
- MAE: 0.179.
- R²: 0.997.

## Candidate Metrics

- baseline_moving_average: RMSE 3.293, MAE 2.395, R² 0.269.
- baseline_persistence: RMSE 0.957, MAE 0.625, R² 0.938.
- baseline_train_mean: RMSE 12.443, MAE 11.832, R² -9.445.
- hist_gradient_boosting: RMSE 1.168, MAE 1.037, R² 0.908.
- random_forest: RMSE 0.617, MAE 0.494, R² 0.974.
- ridge: RMSE 0.210, MAE 0.179, R² 0.997.
- tensorflow_dense: RMSE 17.431, MAE 14.485, R² -19.499.

## Feature Engineering

Core model inputs include pollutants, Karachi-local time features, lag features, rolling AQI averages, AQI change rate, and prediction horizon. Weather columns are included only when historical coverage is reliable enough for training.

Excluded high-missingness weather columns:

- temp
- feels_like
- pressure
- humidity
- wind_speed
- wind_deg
- clouds

Weather missingness observed during training:

- clouds: 0.961
- feels_like: 0.961
- humidity: 0.961
- pressure: 0.961
- temp: 0.961
- wind_deg: 0.961
- wind_speed: 0.961

## Explainability

- Method: shap.
- Status: ok.
- Computed at: 2026-06-02T05:02:32Z.
- Scope: precomputed_training_holdout_sample.

Top contributing features:

- aqi_lag_1h: 2.6596
- aqi_change_rate: 0.5335
- aqi_rolling_3h: 0.4931
- pm2_5: 0.3142
- pm10: 0.2216
- month: 0.1199
- aqi_rolling_24h: 0.0474
- hour: 0.0383

## Data Quality Notes

- AQI score is OpenWeather-derived from pollutant concentrations using EPA-style breakpoints. It is not an official government regulatory Karachi AQI station feed.
- Historical OpenWeather weather access is plan-dependent. Cached weather rows are joined when available, but sparse high-missingness weather fields are excluded from training to avoid misleading features.
- Live prediction still uses OpenWeather forecast weather and forecast pollutant inputs where available.
- GitHub Actions scheduled workflows are best-effort and may drift; the hourly workflow is configured hourly but is not guaranteed to execute exactly every hour.
- `/model-info`, `/ready`, and the dashboard expose latest feature freshness so stale pipeline data is visible to users.

## Deployment

- Streamlit Community Cloud serves `app/streamlit_app.py`.
- Vercel serves the lightweight Flask API through `api/index.py`, `vercel.json`, and `requirements-api.txt`.
- Cloud Run remains the recommended fallback API target if Vercel hits dependency, memory, timeout, or TensorFlow-serving limits.
