# Pearls AQI Predictor Report

## Objective

Predict Karachi AQI for the next 72 hours using a serverless ML pipeline with automated feature ingestion, model training, model registry storage, and an interactive dashboard.

## Implemented System

- Feature pipeline fetches OpenWeather air pollution and weather data, computes time, pollutant, weather, lag, rolling, and AQI-change features, and writes them to Hopsworks-compatible feature groups.
- Backfill pipeline creates historical training rows from OpenWeather history or deterministic sample data for local development.
- Training pipeline evaluates persistence, moving average, Ridge Regression, Random Forest, HistGradientBoosting, and an optional TensorFlow dense model experiment.
- Prediction pipeline loads the latest local/Hopsworks-registered model artifact and generates 72 hourly AQI forecasts.
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

The training pipeline reports RMSE, MAE, and R² for every candidate and stores the selected model under `models/latest`. The same metadata is written to `reports/model_metrics.json` after training.

Latest registered run:

- Selected model: Ridge Regression.
- Ridge metrics: RMSE 0.342, MAE 0.293, R² 0.997.
- Random Forest metrics: RMSE 0.795, MAE 0.545, R² 0.986.
- HistGradientBoosting metrics: RMSE 1.148, MAE 0.844, R² 0.971.
- TensorFlow dense experiment: RMSE 9.298, MAE 8.245, R² -0.899.
- Hopsworks model registry: `karachi_aqi_predictor`, latest verified version 3.

TensorFlow is retained as an experiment because it is part of the required stack, but the tabular dataset is small and the Ridge model currently performs better.

## Data Quality Notes

- Historical OpenWeather air-pollution backfill does not return matching historical weather fields.
- Historical weather columns are retained for schema consistency and filled by the training pipeline with training-set medians.
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
- Render/Fly.io/Railway or another lightweight free tier for Flask.
