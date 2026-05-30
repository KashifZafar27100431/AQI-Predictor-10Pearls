# Karachi AQI EDA Report

Generated from Hopsworks feature group `karachi_aqi_features`.

## Dataset Coverage

- Rows: 695
- Time range: 2026-04-29 17:00 UTC to 2026-05-30 13:00 UTC
- Local analysis timezone: `Asia/Karachi`
- Target: `aqi_score` derived from pollutant concentrations with EPA-style breakpoints.

## AQI Distribution

|       |    aqi |
|:------|-------:|
| count | 695.00 |
| mean  |  69.28 |
| std   |  20.26 |
| min   |  38.00 |
| 10%   |  46.00 |
| 25%   |  56.00 |
| 50%   |  63.00 |
| 75%   |  77.50 |
| 90%   | 104.00 |
| max   | 120.00 |

## AQI Categories

| aqi_category                   |   hours |
|:-------------------------------|--------:|
| Moderate                       |  510.00 |
| Good                           |   98.00 |
| Unhealthy for Sensitive Groups |   87.00 |

## Primary Pollutants

| primary_pollutant   |   hours |
|:--------------------|--------:|
| pm2_5               |  596.00 |
| pm10                |   99.00 |

## Pollutant Summary

|       |   mean |   std |   min |   50% |    max |
|:------|-------:|------:|------:|------:|-------:|
| pm2_5 |  18.60 |  9.95 |  6.80 | 15.39 |  43.09 |
| pm10  |  72.66 | 23.61 | 35.27 | 67.27 | 136.37 |
| co    |  92.76 | 15.06 | 70.29 | 87.08 | 129.49 |
| no2   |   0.08 |  0.04 |  0.03 |  0.07 |   0.36 |
| o3    |  56.25 | 21.10 | 31.14 | 49.92 | 111.58 |
| so2   |   0.41 |  0.24 |  0.15 |  0.34 |   1.60 |
| nh3   |   0.01 |  0.05 |  0.00 |  0.00 |   0.44 |

## Weather Missingness

|            |   missing_pct |
|:-----------|--------------:|
| temp       |         99.14 |
| feels_like |         99.14 |
| pressure   |         99.14 |
| humidity   |         99.14 |
| wind_speed |         99.14 |
| wind_deg   |         99.14 |
| clouds     |         99.14 |

OpenWeather historical air-pollution backfill does not provide matching historical weather on every plan. The current backfill uses cached `karachi_weather_raw` rows when available. During training, weather columns above the configured missingness threshold are excluded from the model feature schema instead of being treated as reliable historical predictors. Live feature ingestion and forecast prediction still use OpenWeather current/forecast weather values when available.

## Hourly AQI Pattern

|   hour |   mean |   min |    max |   count |
|-------:|-------:|------:|-------:|--------:|
|      0 |  69.55 | 40.00 | 117.00 |   29.00 |
|      1 |  69.31 | 39.00 | 115.00 |   29.00 |
|      2 |  70.03 | 39.00 | 115.00 |   29.00 |
|      3 |  70.34 | 39.00 | 118.00 |   29.00 |
|      4 |  70.24 | 39.00 | 119.00 |   29.00 |
|      5 |  70.38 | 40.00 | 117.00 |   29.00 |
|      6 |  70.90 | 41.00 | 115.00 |   29.00 |
|      7 |  71.03 | 42.00 | 115.00 |   29.00 |
|      8 |  69.48 | 43.00 | 115.00 |   29.00 |
|      9 |  70.38 | 43.00 | 114.00 |   29.00 |
|     10 |  70.03 | 43.00 | 112.00 |   29.00 |
|     11 |  69.79 | 43.00 | 109.00 |   29.00 |
|     12 |  69.41 | 43.00 | 111.00 |   29.00 |
|     13 |  69.00 | 42.00 | 113.00 |   29.00 |
|     14 |  68.62 | 41.00 | 116.00 |   29.00 |
|     15 |  68.55 | 41.00 | 119.00 |   29.00 |
|     16 |  68.45 | 40.00 | 120.00 |   29.00 |
|     17 |  68.21 | 40.00 | 119.00 |   29.00 |
|     18 |  67.93 | 40.00 | 118.00 |   29.00 |
|     19 |  68.07 | 38.00 | 117.00 |   28.00 |
|     20 |  67.93 | 38.00 | 116.00 |   28.00 |
|     21 |  67.82 | 38.00 | 116.00 |   28.00 |
|     22 |  68.50 | 40.00 | 116.00 |   30.00 |
|     23 |  68.63 | 41.00 | 117.00 |   30.00 |

## Weekday AQI Pattern

| weekday   |   mean |   min |    max |   count |
|:----------|-------:|------:|-------:|--------:|
| Monday    |  68.96 | 39.00 | 109.00 |   78.00 |
| Tuesday   |  70.82 | 46.00 |  99.00 |   72.00 |
| Wednesday |  72.42 | 53.00 | 119.00 |   92.00 |
| Thursday  |  66.43 | 40.00 | 106.00 |  120.00 |
| Friday    |  70.35 | 39.00 | 120.00 |  122.00 |
| Saturday  |  71.28 | 43.00 | 117.00 |  115.00 |
| Sunday    |  65.17 | 38.00 | 113.00 |   96.00 |

## Strongest Numeric Correlations With AQI

|                 |   corr_with_aqi |
|:----------------|----------------:|
| aqi_lag_1h      |           0.992 |
| aqi_rolling_3h  |           0.985 |
| pm2_5           |           0.982 |
| aqi_rolling_24h |           0.888 |
| pm10            |           0.859 |
| o3              |           0.822 |
| co              |           0.784 |
| so2             |           0.704 |
| no2             |           0.533 |
| aqi_change_rate |           0.058 |
| weekday         |          -0.045 |
| hour            |          -0.030 |

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
