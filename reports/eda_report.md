# Karachi AQI EDA Report

Generated from Hopsworks feature group `karachi_aqi_features`.

## Dataset Coverage

- Rows: 676
- Time range: 2026-04-29 17:00 UTC to 2026-05-29 18:32 UTC
- Target: `aqi_score` derived from pollutant concentrations with EPA-style breakpoints.

## AQI Distribution

|       |    aqi |
|:------|-------:|
| count | 676.00 |
| mean  |  69.52 |
| std   |  20.49 |
| min   |  38.00 |
| 10%   |  46.00 |
| 25%   |  56.00 |
| 50%   |  63.00 |
| 75%   |  78.00 |
| 90%   | 104.00 |
| max   | 120.00 |

## AQI Categories

| aqi_category                   |   hours |
|:-------------------------------|--------:|
| Moderate                       |  491.00 |
| Good                           |   98.00 |
| Unhealthy for Sensitive Groups |   87.00 |

## Primary Pollutants

| primary_pollutant   |   hours |
|:--------------------|--------:|
| pm2_5               |  577.00 |
| pm10                |   99.00 |

## Pollutant Summary

|       |   mean |   std |   min |   50% |    max |
|:------|-------:|------:|------:|------:|-------:|
| pm2_5 |  18.72 | 10.07 |  6.80 | 15.46 |  43.09 |
| pm10  |  72.75 | 23.89 | 35.27 | 65.89 | 136.37 |
| co    |  93.22 | 15.01 | 70.29 | 87.60 | 129.49 |
| no2   |   0.08 |  0.04 |  0.03 |  0.07 |   0.36 |
| o3    |  56.80 | 21.13 | 31.14 | 50.45 | 111.58 |
| so2   |   0.41 |  0.25 |  0.15 |  0.34 |   1.60 |
| nh3   |   0.01 |  0.05 |  0.00 |  0.00 |   0.44 |

## Weather Missingness

|            |   missing_pct |
|:-----------|--------------:|
| temp       |         99.70 |
| feels_like |         99.70 |
| pressure   |         99.70 |
| humidity   |         99.70 |
| wind_speed |         99.70 |
| wind_deg   |         99.70 |
| clouds     |         99.70 |

Historical OpenWeather air-pollution backfill does not include historical weather. Those weather columns are retained in the schema and filled during model preparation with training-set medians, while live feature ingestion and forecast prediction use OpenWeather current/forecast weather values. This keeps the Feature Store contract stable, but the historical weather signal should be considered imputed.

## Hourly AQI Pattern

|   hour |   mean |   min |    max |   count |
|-------:|-------:|------:|-------:|--------:|
|      0 |  70.68 | 40.00 | 117.00 |   28.00 |
|      1 |  71.21 | 41.00 | 115.00 |   28.00 |
|      2 |  71.32 | 42.00 | 115.00 |   28.00 |
|      3 |  69.75 | 43.00 | 115.00 |   28.00 |
|      4 |  70.71 | 43.00 | 114.00 |   28.00 |
|      5 |  70.36 | 43.00 | 112.00 |   28.00 |
|      6 |  70.11 | 43.00 | 109.00 |   28.00 |
|      7 |  69.68 | 43.00 | 111.00 |   28.00 |
|      8 |  69.29 | 42.00 | 113.00 |   28.00 |
|      9 |  68.89 | 41.00 | 116.00 |   28.00 |
|     10 |  68.75 | 41.00 | 119.00 |   28.00 |
|     11 |  68.68 | 40.00 | 120.00 |   28.00 |
|     12 |  68.43 | 40.00 | 119.00 |   28.00 |
|     13 |  68.14 | 40.00 | 118.00 |   28.00 |
|     14 |  68.11 | 38.00 | 117.00 |   28.00 |
|     15 |  68.00 | 38.00 | 116.00 |   28.00 |
|     16 |  67.86 | 38.00 | 116.00 |   28.00 |
|     17 |  68.53 | 40.00 | 116.00 |   30.00 |
|     18 |  68.67 | 41.00 | 117.00 |   30.00 |
|     19 |  69.93 | 40.00 | 117.00 |   28.00 |
|     20 |  69.68 | 39.00 | 115.00 |   28.00 |
|     21 |  70.43 | 39.00 | 115.00 |   28.00 |
|     22 |  70.71 | 39.00 | 118.00 |   28.00 |
|     23 |  70.61 | 39.00 | 119.00 |   28.00 |

## Weekday AQI Pattern

| weekday   |   mean |   min |    max |   count |
|:----------|-------:|------:|-------:|--------:|
| Monday    |  69.26 | 43.00 | 103.00 |   73.00 |
| Tuesday   |  73.81 | 53.00 | 119.00 |   72.00 |
| Wednesday |  70.29 | 53.00 | 117.00 |  102.00 |
| Thursday  |  66.22 | 39.00 | 106.00 |  120.00 |
| Friday    |  71.55 | 40.00 | 120.00 |  117.00 |
| Saturday  |  71.84 | 43.00 | 115.00 |   96.00 |
| Sunday    |  64.98 | 38.00 | 109.00 |   96.00 |

## Strongest Numeric Correlations With AQI

|                 |   corr_with_aqi |
|:----------------|----------------:|
| aqi_lag_1h      |           0.992 |
| aqi_rolling_3h  |           0.985 |
| pm2_5           |           0.982 |
| aqi_rolling_24h |           0.887 |
| pm10            |           0.860 |
| o3              |           0.824 |
| co              |           0.787 |
| so2             |           0.708 |
| no2             |           0.541 |
| aqi_change_rate |           0.060 |
| weekday         |          -0.049 |
| nh3             |           0.021 |

Weather columns with more than 50% missing values are excluded from this correlation table to avoid misleading sparse-column correlations.

## Highest AQI Hours

|     | event_time           |   aqi_score | aqi_category                   | primary_pollutant   |   pm2_5 |   pm10 |
|----:|:---------------------|------------:|:-------------------------------|:--------------------|--------:|-------:|
|  42 | 2026-05-01 11:00 UTC |         120 | Unhealthy for Sensitive Groups | pm2_5               |   43.09 | 113.40 |
|  41 | 2026-05-01 10:00 UTC |         119 | Unhealthy for Sensitive Groups | pm2_5               |   42.63 | 114.55 |
|  43 | 2026-05-01 12:00 UTC |         119 | Unhealthy for Sensitive Groups | pm2_5               |   42.92 | 111.21 |
| 294 | 2026-05-12 23:00 UTC |         119 | Unhealthy for Sensitive Groups | pm2_5               |   42.81 | 124.42 |
|  44 | 2026-05-01 13:00 UTC |         118 | Unhealthy for Sensitive Groups | pm2_5               |   42.53 | 108.85 |
| 293 | 2026-05-12 22:00 UTC |         118 | Unhealthy for Sensitive Groups | pm2_5               |   42.48 | 128.59 |
|  45 | 2026-05-01 14:00 UTC |         117 | Unhealthy for Sensitive Groups | pm2_5               |   42.01 | 106.33 |
|  49 | 2026-05-01 18:00 UTC |         117 | Unhealthy for Sensitive Groups | pm2_5               |   41.85 | 105.33 |
|  50 | 2026-05-01 19:00 UTC |         117 | Unhealthy for Sensitive Groups | pm2_5               |   41.81 | 104.12 |
| 295 | 2026-05-13 00:00 UTC |         117 | Unhealthy for Sensitive Groups | pm2_5               |   42.01 | 119.20 |

## Findings

- Recent Karachi AQI in this dataset is mostly in the Good to Moderate range, with PM2.5 and PM10 driving the AQI score.
- Lag and rolling AQI features are strongly related to the target, so persistence remains an important baseline.
- Historical weather features are weaker than pollutant features because the pollution history endpoint does not return matching weather history.
- The dashboard should present predictions as model estimates from OpenWeather-derived pollutant data, not as official regulatory station readings.
