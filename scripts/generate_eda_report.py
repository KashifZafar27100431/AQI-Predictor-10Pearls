from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.storage import FEATURES_FEATURE_GROUP, FeatureStoreClient


FEATURE_PATH = Path("data/processed/karachi_aqi_features.csv")
REPORT_PATH = Path("reports/eda_report.md")
TARGET = "aqi_score"
POLLUTANTS = ["pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3"]
WEATHER = ["temp", "feels_like", "pressure", "humidity", "wind_speed", "wind_deg", "clouds"]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def _load_features() -> tuple[pd.DataFrame, str]:
    _load_dotenv()
    settings = get_settings()
    source = f"local file `{FEATURE_PATH}`"
    frame = pd.DataFrame()
    if settings.hopsworks_api_key and settings.hopsworks_project:
        frame = FeatureStoreClient(settings).read_feature_group(FEATURES_FEATURE_GROUP)
        if not frame.empty:
            source = "Hopsworks feature group `karachi_aqi_features`"
    if frame.empty:
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"{FEATURE_PATH} not found. Run the backfill or feature pipeline before EDA."
            )
        frame = pd.read_csv(FEATURE_PATH)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    return frame.sort_values("event_time"), source


def _table(frame: pd.DataFrame, float_format: str = ".2f") -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=True, floatfmt=float_format)


def _series_table(series: pd.Series, name: str) -> str:
    return series.rename(name).to_frame().to_markdown(floatfmt=".2f")


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in frame.columns]


def build_report(frame: pd.DataFrame, source: str) -> str:
    rows = len(frame)
    start = frame["event_time"].min().strftime("%Y-%m-%d %H:%M UTC")
    end = frame["event_time"].max().strftime("%Y-%m-%d %H:%M UTC")
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    pollutant_columns = _available(POLLUTANTS, frame)
    weather_columns = _available(WEATHER, frame)

    target_summary = target.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("aqi")
    category_counts = frame["aqi_category"].value_counts(dropna=False).to_frame("hours")
    primary_counts = frame["primary_pollutant"].value_counts(dropna=False).head(10).to_frame("hours")
    pollutant_summary = frame[pollutant_columns].describe().T[["mean", "std", "min", "50%", "max"]]
    missing_weather = frame[weather_columns].isna().mean().mul(100).sort_values(ascending=False).to_frame("missing_pct")

    by_hour = (
        frame.assign(hour=frame["event_time"].dt.hour)
        .groupby("hour")[TARGET]
        .agg(["mean", "min", "max", "count"])
    )
    by_weekday = (
        frame.assign(weekday=frame["event_time"].dt.day_name())
        .groupby("weekday")[TARGET]
        .agg(["mean", "min", "max", "count"])
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .dropna(how="all")
    )
    top_hours = frame.nlargest(10, TARGET)[
        ["event_time", TARGET, "aqi_category", "primary_pollutant", "pm2_5", "pm10"]
    ].copy()
    top_hours["event_time"] = top_hours["event_time"].dt.strftime("%Y-%m-%d %H:%M UTC")

    missing_rates = frame[pollutant_columns + weather_columns].isna().mean()
    correlation_base_columns = [
        column for column in pollutant_columns + weather_columns if missing_rates.get(column, 1.0) <= 0.5
    ]
    numeric_columns = correlation_base_columns + [
        "hour",
        "weekday",
        "aqi_lag_1h",
        "aqi_rolling_3h",
        "aqi_rolling_24h",
        "aqi_change_rate",
    ]
    numeric_columns = _available(numeric_columns, frame)
    correlations = (
        frame[numeric_columns + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(12)
        .to_frame("corr_with_aqi")
    )

    weather_missing_max = float(missing_weather["missing_pct"].max()) if not missing_weather.empty else 0.0
    weather_note = (
        "Historical OpenWeather air-pollution backfill does not include historical weather. "
        "Those weather columns are retained in the schema and filled during model preparation "
        "with training-set medians, while live feature ingestion and forecast prediction use "
        "OpenWeather current/forecast weather values. This keeps the Feature Store contract "
        "stable, but the historical weather signal should be considered imputed."
    )
    if weather_missing_max == 0.0:
        weather_note = (
            "Weather fields are fully populated in this local feature cache. The backfill code "
            "still supports missing historical weather and the training pipeline fills any gaps "
            "with training-set medians."
        )

    return f"""# Karachi AQI EDA Report

Generated from {source}.

## Dataset Coverage

- Rows: {rows}
- Time range: {start} to {end}
- Target: `{TARGET}` derived from pollutant concentrations with EPA-style breakpoints.

## AQI Distribution

{_table(target_summary)}

## AQI Categories

{_table(category_counts)}

## Primary Pollutants

{_table(primary_counts)}

## Pollutant Summary

{_table(pollutant_summary)}

## Weather Missingness

{_table(missing_weather)}

{weather_note}

## Hourly AQI Pattern

{_table(by_hour)}

## Weekday AQI Pattern

{_table(by_weekday)}

## Strongest Numeric Correlations With AQI

{_table(correlations, float_format=".3f")}

Weather columns with more than 50% missing values are excluded from this correlation table to avoid misleading sparse-column correlations.

## Highest AQI Hours

{_table(top_hours)}

## Findings

- Recent Karachi AQI in this dataset is mostly in the Good to Moderate range, with PM2.5 and PM10 driving the AQI score.
- Lag and rolling AQI features are strongly related to the target, so persistence remains an important baseline.
- Historical weather features are weaker than pollutant features because the pollution history endpoint does not return matching weather history.
- The dashboard should present predictions as model estimates from OpenWeather-derived pollutant data, not as official regulatory station readings.
"""


def main() -> None:
    frame, source = _load_features()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(frame, source), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
