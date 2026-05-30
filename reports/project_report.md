# Pearls AQI Predictor Report

## Objective

Predict Karachi AQI for the next 72 hours using a serverless ML pipeline with automated feature ingestion, model training, model registry storage, and an interactive dashboard.

## Implemented System

- Feature pipeline fetches OpenWeather air pollution and weather data, computes time, pollutant, weather, lag, rolling, and AQI-change features, and writes them to Hopsworks-compatible feature groups.
- Backfill pipeline creates historical training rows from OpenWeather history or deterministic sample data for local development.
- Training pipeline evaluates persistence, moving average, Ridge Regression, Random Forest, HistGradientBoosting, and an optional TensorFlow dense model experiment.
- Prediction pipeline loads the latest approved or latest available Hopsworks-registered model artifact first, with local fallback only for development, and generates 72 hourly AQI forecasts.
- Flask exposes prediction, latest, alert, model-info, and health endpoints.
- Streamlit provides the dashboard for current AQI, 3-day forecast, pollutant trends, hazardous alerts, model metrics, and feature importance.
- `reports/eda_report.md` provides reproducible EDA generated from the Hopsworks feature group.

## Feature Engineering

Core model inputs:

- Pollutants: CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3.
- Weather: temperature, feels-like temperature, pressure, humidity, wind speed, wind direction, cloud cover.
- Time: hour, day, month, weekday, weekend flag.
- Derived AQI: previous AQI, 3-hour rolling AQI, 24-hour rolling AQI, AQI change rate, forecast horizon.
- AQI score: derived from pollutant concentrations with U.S. EPA-style breakpoints; OpenWeather's native 1-5 pollution level is retained separately as `ow_aqi`.

## Model Evaluation

The training pipeline reports RMSE, MAE, and R² for every candidate, stores the selected development artifact under `models/latest`, and registers the selected model directory in Hopsworks Model Registry. The same metadata is written to `reports/model_metrics.json` after training.

Latest registered run:

- Selected model: Ridge Regression.
- Ridge metrics: RMSE 0.385, MAE 0.317, R² 0.991.
- Random Forest metrics: RMSE 0.671, MAE 0.561, R² 0.971.
- HistGradientBoosting metrics: RMSE 1.293, MAE 1.076, R² 0.894.
- TensorFlow dense experiment: RMSE 13.212, MAE 11.899, R² -10.056.
- Hopsworks model registry: `karachi_aqi_predictor`, latest verified version 4.

TensorFlow is retained as an experiment because it is part of the required stack, but the tabular dataset is small and the Ridge model currently performs better.

## Data Quality Notes

- Historical OpenWeather air-pollution backfill does not return matching historical weather fields.
- Cached historical weather rows are joined when available. Weather columns above the configured missingness threshold are excluded from the training feature schema instead of being treated as reliable predictors.
- Live feature ingestion and prediction-time forecasts use OpenWeather current/forecast weather values.
- AQI scores are derived from pollutant concentrations using EPA-style breakpoints; OpenWeather's native 1-5 level remains stored as `ow_aqi`.

## Automation

GitHub Actions workflows:

- `hourly-feature-pipeline.yml` runs the feature script every hour.
- `daily-training-pipeline.yml` runs training every day.
- `ci.yml` runs tests on push and pull request.

## Deployment

Recommended free/serverless deployment:

- Hopsworks Serverless for Feature Store and Model Registry.
- MongoDB Atlas free tier for app cache and alert history.
- GitHub Actions for scheduled feature/training jobs.
- Streamlit Community Cloud for the dashboard.
- Cloud Run for the Flask API using the included Dockerfile and GitHub Actions deployment workflow.
