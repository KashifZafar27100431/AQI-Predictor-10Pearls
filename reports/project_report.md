# Pearls AQI Predictor Report

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
- Latest verified registry version: 8.
- Selected model: ridge.
- Model type: sklearn.
- Trained at: 2026-06-01T08:41:40Z.
- Training rows: 712.
- Test rows: 143.
- RMSE: 0.210.
- MAE: 0.183.
- R²: 0.997.

## Candidate Metrics

- baseline_moving_average: RMSE 3.998, MAE 2.718, R² -0.031.
- baseline_persistence: RMSE 0.927, MAE 0.608, R² 0.945.
- baseline_train_mean: RMSE 12.462, MAE 11.824, R² -9.020.
- hist_gradient_boosting: RMSE 1.002, MAE 0.887, R² 0.935.
- random_forest: RMSE 0.600, MAE 0.477, R² 0.977.
- ridge: RMSE 0.210, MAE 0.183, R² 0.997.
- tensorflow_dense: RMSE 15.507, MAE 13.066, R² -14.513.

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

- clouds: 0.968
- feels_like: 0.968
- humidity: 0.968
- pressure: 0.968
- temp: 0.968
- wind_deg: 0.968
- wind_speed: 0.968

## Explainability

- Method: shap.
- Status: ok.
- Computed at: 2026-06-01T08:42:59Z.
- Scope: precomputed_training_holdout_sample.

Top contributing features:

- aqi_lag_1h: 3.2058
- aqi_rolling_3h: 0.5844
- aqi_change_rate: 0.4858
- pm2_5: 0.3884
- pm10: 0.2627
- aqi_rolling_24h: 0.0398
- hour: 0.0379
- no: 0.0373

## Data Quality Notes

- AQI score is OpenWeather-derived from pollutant concentrations using EPA-style breakpoints. It is not an official government regulatory Karachi AQI station feed.
- Historical OpenWeather weather access is plan-dependent. Cached weather rows are joined when available, but sparse high-missingness weather fields are excluded from training to avoid misleading features.
- Live prediction still uses OpenWeather forecast weather and forecast pollutant inputs where available.
- GitHub Actions scheduled workflows are best-effort and may drift; the hourly workflow is configured hourly but is not guaranteed to execute exactly every hour.

## Deployment

- Streamlit Community Cloud serves `app/streamlit_app.py`.
- Vercel serves the lightweight Flask API through `api/index.py` and `vercel.json`.
- Cloud Run remains the recommended fallback API target if Vercel hits dependency, memory, timeout, or TensorFlow-serving limits.
