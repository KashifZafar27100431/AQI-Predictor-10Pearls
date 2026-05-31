# Serverless Deployment

This project is designed for a managed/serverless deployment:

- GitHub Actions schedules the feature and training pipelines.
- Hopsworks stores feature groups and registered model versions.
- MongoDB Atlas stores dashboard cache, prediction history, and alert records.
- Cloud Run serves the Flask API from the included `Dockerfile`.
- Streamlit Community Cloud serves the dashboard.

## GitHub Secrets

Configure these repository secrets:

```text
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT
MONGODB_URI
MONGODB_DATABASE
ALLOWED_ORIGINS
GCP_PROJECT_ID
GCP_REGION
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_SERVICE_ACCOUNT
```

`ALLOWED_ORIGINS` should include the Streamlit dashboard URL and any local development origin, for example:

```text
https://your-dashboard-url.streamlit.app,http://localhost:8501
```

Do not use `*` in production.

## Pipeline Handoff

1. The hourly workflow fetches current OpenWeather inputs and writes raw/engineered rows to Hopsworks.
2. The daily workflow reads `karachi_aqi_features`, trains/evaluates models, writes `metadata.json`, the selected model artifact, and any TensorFlow scaler artifact, then registers the artifact directory as `karachi_aqi_predictor` in Hopsworks Model Registry.
3. The Flask API starts on Cloud Run with `AQI_ALLOW_LOCAL_MODEL_FALLBACK=false` and `AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY=true`.
4. Each prediction/model-info request attempts to download the latest approved, or latest available, registered Hopsworks model and validates the feature schema before serving.

## Cloud Run API

The included `Dockerfile` runs:

```bash
gunicorn 'app.flask_api:create_app()' --bind 0.0.0.0:${PORT}
```

The image installs `requirements-ml.txt` so Cloud Run can serve either a Scikit-learn model or a TensorFlow model if a future daily training run selects TensorFlow as the best registered artifact.

Manual deployment through GitHub Actions:

1. Configure the GitHub secrets listed above.
2. Run `Deploy Flask API to Cloud Run` from the Actions tab.
3. Confirm `/health` returns `{"status":"ok","city":"Karachi"}`.

Production environment variables for Cloud Run:

```text
AQI_CITY=Karachi
AQI_LAT=24.8607
AQI_LON=67.0011
AQI_TIMEZONE=Asia/Karachi
AQI_FORECAST_HOURS=72
AQI_MAX_FORECAST_HOURS=72
AQI_ALLOW_LOCAL_MODEL_FALLBACK=false
LOCAL_MODEL_FALLBACK_ENABLED=false
AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY=true
OPENWEATHER_API_KEY=<secret>
HOPSWORKS_API_KEY=<secret>
HOPSWORKS_PROJECT=<secret>
MONGODB_URI=<secret>
ALLOWED_ORIGINS=https://your-dashboard-url.streamlit.app
```

`LOCAL_MODEL_FALLBACK_ENABLED` is accepted as a Vercel-friendly alias for `AQI_ALLOW_LOCAL_MODEL_FALLBACK`. If both are set, `AQI_ALLOW_LOCAL_MODEL_FALLBACK` takes precedence.

## Vercel Flask API

Vercel can serve the Flask API through `api/index.py`, which imports the WSGI object named `app` from `app/flask_api.py`. The `vercel.json` rewrite sends every route to that entrypoint, so the same Flask implementation serves:

```text
/health
/latest
/predict?horizon=72
/alerts
/model-info
```

Required Vercel environment variables:

```text
AQI_CITY=Karachi
AQI_LAT=24.8607
AQI_LON=67.0011
AQI_TIMEZONE=Asia/Karachi
AQI_FORECAST_HOURS=72
AQI_MAX_FORECAST_HOURS=72
AQI_ALLOW_LOCAL_MODEL_FALLBACK=false
LOCAL_MODEL_FALLBACK_ENABLED=false
AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY=true
OPENWEATHER_API_KEY=<secret>
HOPSWORKS_API_KEY=<secret>
HOPSWORKS_PROJECT=<secret>
MONGODB_URI=<secret>
MONGODB_DATABASE=pearls_aqi
ALLOWED_ORIGINS=https://karachi-aqi-predictor-10pearls.streamlit.app
```

Keep all secret values in the Vercel project environment settings. Do not commit `.env` files or real API keys.

The Vercel runtime uses the root Python dependency metadata. `requirements.txt` and `pyproject.toml` include the Flask serving dependencies plus Hopsworks, SHAP, Streamlit, and other shared project packages. TensorFlow is intentionally kept out of `requirements.txt`; it is only in `requirements-ml.txt` for training and Cloud Run serving. If the latest registered Hopsworks model is a TensorFlow artifact, or if Vercel build/runtime limits are hit by the shared ML dependencies, use Cloud Run for the Flask API because the included Dockerfile installs `requirements-ml.txt` and is sized for ML serving.

## Streamlit Cloud

Deploy `app/streamlit_app.py` as the Streamlit entrypoint.

The repository includes `.streamlit/config.toml` for Streamlit Cloud defaults. Keep secrets in Streamlit's secrets manager or environment settings, not in the config file.
GitHub Actions repository secrets are not available to Streamlit Community Cloud apps. Add the runtime secrets separately in the Streamlit app's settings.
Set the app's Python version to 3.11 in Streamlit's Advanced settings; Community Cloud does not use `runtime.txt` for this project.

Configure Streamlit secrets/environment as root-level TOML keys:

```toml
OPENWEATHER_API_KEY = "..."
HOPSWORKS_API_KEY = "..."
HOPSWORKS_PROJECT = "..."
MONGODB_URI = "..."
MONGODB_DATABASE = "pearls_aqi"
AQI_CITY = "Karachi"
AQI_LAT = "24.8607"
AQI_LON = "67.0011"
AQI_TIMEZONE = "Asia/Karachi"
AQI_ALLOW_LOCAL_MODEL_FALLBACK = "false"
```

The dashboard can call shared service code directly. If deployed separately from the API, keep both deployments pointed at the same Hopsworks project and MongoDB Atlas database.

The dashboard displays the latest observed Karachi AQI from the feature store separately from the next-hour forecast.

## Local Development Fallback

For local-only development, set:

```text
AQI_ALLOW_LOCAL_MODEL_FALLBACK=true
AQI_USE_SAMPLE_DATA=true
```

Then run the sample backfill/training commands from the README. This local fallback is intentionally disabled in the Cloud Run workflow.
