# Pearls AQI Predictor

Serverless AQI forecasting system for Karachi. The project collects air pollution and weather data, engineers hourly features, stores reusable features in Hopsworks, trains multiple forecasting models, registers the best model, and serves 72-hour AQI predictions through Flask and Streamlit.

## Architecture

- **Data source:** OpenWeather Air Pollution API for current, forecast, and historical pollutant data.
- **Feature Store:** Hopsworks feature groups for raw air quality, raw weather, engineered features, and predictions.
- **Backend cache:** MongoDB Atlas for latest dashboard records, prediction history, and hazardous alert events.
- **Training:** scikit-learn baselines and regressors, plus an optional TensorFlow dense model experiment.
- **Serving:** Flask JSON API and Streamlit dashboard.
- **Automation:** GitHub Actions hourly feature pipeline and daily training pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

For the daily TensorFlow experiment:

```bash
pip install -r requirements-ml.txt
```

Fill `.env` with:

```bash
OPENWEATHER_API_KEY=...
HOPSWORKS_API_KEY=...
HOPSWORKS_PROJECT=...
MONGODB_URI=...
```

## Local Demo Flow

Run the project without external credentials by using deterministic sample data:

```bash
python -m aqi_predictor.pipelines.backfill_pipeline --days 60 --sample
python -m aqi_predictor.pipelines.training_pipeline
python -m aqi_predictor.pipelines.predict_pipeline --horizon 72 --sample
```

Start the API:

```bash
python app/flask_api.py
```

Start the dashboard:

```bash
streamlit run app/streamlit_app.py
```

In the dashboard, **Sample mode** uses deterministic synthetic Karachi-like AQI and weather inputs. It is for demos and offline testing only. When Sample mode is off, `OPENWEATHER_API_KEY` must be configured or the app/API will fail explicitly instead of silently showing fake live predictions.

## Production Pipeline Commands

Hourly feature ingestion:

```bash
python -m aqi_predictor.pipelines.feature_pipeline
```

Historical backfill:

```bash
python -m aqi_predictor.pipelines.backfill_pipeline --days 90
```

Daily training:

```bash
python -m aqi_predictor.pipelines.training_pipeline --no-sample-if-empty
```

Batch prediction:

```bash
python -m aqi_predictor.pipelines.predict_pipeline --horizon 72
```

Generate the EDA report:

```bash
PYTHONPATH=src python scripts/generate_eda_report.py
```

Repair old feature rows if the AQI scoring logic changes:

```bash
PYTHONPATH=src python scripts/repair_feature_scores.py
```

## API

- `GET /health`
- `GET /latest`
- `GET /predict?horizon=72`
- `GET /alerts`
- `GET /model-info`

## Feature Groups

- `karachi_air_quality_raw`
- `karachi_weather_raw`
- `karachi_aqi_features`
- `karachi_aqi_predictions`

## Tests

```bash
python -m pytest
```

If pytest is not installed in the current environment:

```bash
PYTHONPATH=src python -m unittest
```

## Notes

- OpenWeather forecast air pollution data includes hourly forecast data. The displayed AQI score is derived from pollutant concentrations with U.S. EPA-style breakpoints, while `ow_aqi` is retained as OpenWeather's coarse 1-5 pollution level.
- Historical OpenWeather air-pollution backfill does not include historical weather values. The feature schema keeps weather columns, live/forecast rows use OpenWeather weather APIs, and model training fills historical weather gaps with training-set medians.
- `reports/eda_report.md` is generated from the Hopsworks feature group when credentials are configured and falls back to local cached features for offline development.
- `reports/model_metrics.json` includes baselines, Ridge, Random Forest, HistGradientBoosting, and a TensorFlow dense experiment. Ridge is selected only when it has the lowest validation RMSE.
- MongoDB is intentionally used as supporting backend storage, not as a replacement for Hopsworks.
- Hopsworks registration is optional at runtime and activates when `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` are set.
- Before publishing or submitting, rotate any API key that was ever pasted into `.env.example` or a terminal log. Keep `.env` local only.

References:

- OpenWeather Air Pollution API: https://openweathermap.org/api/air-pollution
- Hopsworks feature group API: https://docs.hopsworks.ai/latest/python-api/hsfs/feature_group/
- Hopsworks Python model registry: https://docs.hopsworks.ai/latest/user_guides/mlops/registry/frameworks/python/
- GitHub Actions scheduled workflows: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule
