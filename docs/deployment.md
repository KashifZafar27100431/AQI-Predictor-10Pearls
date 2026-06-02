# Serverless Deployment

This project is designed for a managed/serverless deployment:

- GitHub repo: https://github.com/KashifZafar27100431/AQI-Predictor-10Pearls
- Streamlit dashboard: https://karachi-aqi-predictor-10pearls.streamlit.app/
- Vercel Flask API: https://aqi-predictor-10-pearls.vercel.app/
- GitHub Actions schedules the feature and training pipelines.
- Hopsworks stores feature groups and registered model versions.
- MongoDB Atlas stores dashboard cache, prediction history, and alert records.
- Vercel serves the lightweight Flask API from `api/index.py`.
- Cloud Run remains the Docker/serverless fallback for heavier ML serving.
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
3. The Flask API starts on Vercel, or Cloud Run fallback, with `AQI_ALLOW_LOCAL_MODEL_FALLBACK=false` and `AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY=true`.
4. Each prediction/model-info request attempts to download the latest approved, or latest available, registered Hopsworks model and validates the feature schema before serving.
5. If `HOPSWORKS_MODEL_VERSION` is set, the serving layer loads that exact registry version instead of selecting latest. Pin version 9 for the current deployment until latest-model selection is proven stable across cold starts.

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
HOPSWORKS_MODEL_VERSION=9
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
/
/health
/ready
/latest
/predict?horizon=72
/alerts?limit=20&forecast=true
/model-info
/diagnostics
```

Required Vercel environment variables:

```text
OPENWEATHER_API_KEY=<secret>
HOPSWORKS_API_KEY=<secret>
HOPSWORKS_PROJECT=<secret>
MONGODB_URI=<secret>
MONGODB_DATABASE=pearls_aqi
AQI_CITY=Karachi
AQI_LAT=24.8607
AQI_LON=67.0011
AQI_TIMEZONE=Asia/Karachi
AQI_FORECAST_HOURS=72
AQI_MAX_FORECAST_HOURS=72
AQI_ALLOW_LOCAL_MODEL_FALLBACK=false
LOCAL_MODEL_FALLBACK_ENABLED=false
AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY=true
HOPSWORKS_MODEL_VERSION=9
ALLOWED_ORIGINS=https://karachi-aqi-predictor-10pearls.streamlit.app
```

Keep all secret values in the Vercel project environment settings. Add them to **Production** for the live `vercel.app` deployment. Add the same values to **Preview** if branch or pull-request preview deployments must run real predictions. Use local `.env` for local development; Vercel **Development** variables are only needed when running `vercel dev`. Redeploy Vercel after changing environment variables.

`/alerts` reads MongoDB/Hopsworks-stored alert rows first. If none exist and `forecast=true`, it performs a non-persisted current forecast and returns USG/unhealthy/hazardous forecast advisories, so a forecast with worst AQI 101-150 still produces a sensitive-groups alert response.

When `VERCEL=1` is present, the code defaults to `AQI_ALLOW_LOCAL_MODEL_FALLBACK=false` and `AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY=true` even if those env vars are omitted. Keep the explicit env vars anyway so deployment intent is visible in Vercel settings.

The Vercel runtime uses `requirements-api.txt` through `vercel.json` `installCommand`, so API serving dependencies stay limited to Flask, CORS, pandas, numpy, requests, scikit-learn, joblib, Hopsworks, pyarrow, pymongo, and python-dotenv. `pyarrow` is explicitly included because Hopsworks/HSFS imports Arrow Flight components during registry access. Streamlit, Plotly, SHAP, pytest, ruff, and TensorFlow stay out of the Vercel install and are installed through `requirements.txt`, `requirements-dev.txt`, or `requirements-ml.txt` for dashboard, development, training, and Cloud Run use. If the latest registered Hopsworks model is a TensorFlow artifact, or if Vercel still hits Hopsworks cold-start, dependency, memory, or timeout limits, use Cloud Run for the Flask API because the included Dockerfile installs `requirements-ml.txt` and is sized for heavier ML serving.
`requirements-api.txt` pins `scikit-learn==1.8.0` because the current Hopsworks registry version 9 artifact was serialized with scikit-learn 1.8.0.

When `VERCEL=1` is present, runtime model and data caches default to `/tmp/aqi_predictor/...` because Vercel functions should not rely on writing inside the deployed repository. If `AQI_MODEL_DIR` or `AQI_LOCAL_DATA_DIR` is set to a relative path on Vercel, the app relocates it under `/tmp/aqi_predictor/`. Absolute overrides should also point to `/tmp`.

## Streamlit Cloud

Deploy `app/streamlit_app.py` as the Streamlit entrypoint.

The repository includes `.streamlit/config.toml` for Streamlit Cloud defaults. Keep secrets in Streamlit's secrets manager or environment settings, not in the config file.
GitHub Actions repository secrets are not available to Streamlit Community Cloud apps. Add the runtime secrets separately in the Streamlit app's settings.
Set the app's Python version to 3.11 in Streamlit's Advanced settings; Community Cloud does not use `runtime.txt` for this project.

Configure Streamlit secrets/environment as root-level TOML keys. If `API_BASE_URL` is set, Streamlit calls the deployed Flask API for `/predict`, `/model-info`, `/latest`, and `/alerts`; if it is omitted, Streamlit uses the shared service code directly.

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
AQI_REQUIRE_HOPSWORKS_MODEL_REGISTRY = "true"
HOPSWORKS_MODEL_VERSION = "9"
API_BASE_URL = "https://aqi-predictor-10-pearls.vercel.app"
```

The dashboard can call shared service code directly. If deployed separately from the API, keep both deployments pointed at the same Hopsworks project and MongoDB Atlas database.

The dashboard displays the latest observed Karachi AQI from the feature store separately from the next-hour forecast.
Some command-line checks may receive Streamlit Community Cloud auth redirects even when the app is accessible in a normal browser. Do not disable Streamlit/XSRF protection to satisfy `curl`; verify browser access manually and keep secrets in Streamlit secrets.

## Explainability

Training generates precomputed SHAP artifacts for the selected Scikit-learn model:

```text
reports/shap_summary.json
reports/feature_importance.json
```

The dashboard displays those precomputed artifacts when available. Vercel does not compute SHAP live because the API runtime must stay small and fast.

## Data Freshness and Weather Limitations

OpenWeather historical pollutant data is used for backfill. Historical weather availability is limited by API plan and cached `karachi_weather_raw` coverage. Weather columns with excessive missingness are excluded from training instead of being heavily imputed. Live forecasts can still include OpenWeather forecast weather values where available, but the current registry version 9 model excludes weather columns because historical missingness is about 96%.

`/model-info`, `/ready`, and the Streamlit model section expose latest feature timestamp, feature age, and freshness status so users can see whether scheduled ingestion is current.

## Scheduler Accuracy

GitHub Actions scheduled workflows are best-effort. The hourly feature workflow is configured with `17 * * * *`, but GitHub may delay or skip exact timing under load. Do not present the pipeline as guaranteed to run at the exact same minute every hour.

## Post-change Redeploy

After changing Vercel env vars, redeploy the Vercel project and verify:

```text
https://aqi-predictor-10-pearls.vercel.app/health
https://aqi-predictor-10-pearls.vercel.app/ready
https://aqi-predictor-10-pearls.vercel.app/model-info
https://aqi-predictor-10-pearls.vercel.app/predict?horizon=72
```

After changing Streamlit secrets, reboot the Streamlit Cloud app and verify:

```text
https://karachi-aqi-predictor-10pearls.streamlit.app/
```

## Local Development Fallback

For local-only development, set:

```text
AQI_ALLOW_LOCAL_MODEL_FALLBACK=true
AQI_USE_SAMPLE_DATA=true
```

Then run the sample backfill/training commands from the README. This local fallback is intentionally disabled in the Cloud Run workflow.
