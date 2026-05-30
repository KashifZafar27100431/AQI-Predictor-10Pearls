from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple


OPENWEATHER_AQI_LABELS: Dict[int, str] = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}

AQI_SCORE_BY_OPENWEATHER_LEVEL: Dict[int, int] = {
    1: 25,
    2: 75,
    3: 125,
    4: 175,
    5: 250,
}

US_AQI_LABELS: Dict[int, str] = {
    1: "Good",
    2: "Moderate",
    3: "Unhealthy for Sensitive Groups",
    4: "Unhealthy",
    5: "Very Unhealthy",
    6: "Hazardous",
}

Breakpoint = Tuple[float, float, int, int]

PM25_BREAKPOINTS: Tuple[Breakpoint, ...] = (
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
)

PM10_BREAKPOINTS: Tuple[Breakpoint, ...] = (
    (0.0, 54.0, 0, 50),
    (55.0, 154.0, 51, 100),
    (155.0, 254.0, 101, 150),
    (255.0, 354.0, 151, 200),
    (355.0, 424.0, 201, 300),
    (425.0, 604.0, 301, 500),
)

O3_8H_PPM_BREAKPOINTS: Tuple[Breakpoint, ...] = (
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
)

CO_8H_PPM_BREAKPOINTS: Tuple[Breakpoint, ...] = (
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
)

NO2_1H_PPB_BREAKPOINTS: Tuple[Breakpoint, ...] = (
    (0.0, 53.0, 0, 50),
    (54.0, 100.0, 51, 100),
    (101.0, 360.0, 101, 150),
    (361.0, 649.0, 151, 200),
    (650.0, 1249.0, 201, 300),
    (1250.0, 1649.0, 301, 400),
    (1650.0, 2049.0, 401, 500),
)

SO2_1H_PPB_BREAKPOINTS: Tuple[Breakpoint, ...] = (
    (0.0, 35.0, 0, 50),
    (36.0, 75.0, 51, 100),
    (76.0, 185.0, 101, 150),
    (186.0, 304.0, 151, 200),
    (305.0, 604.0, 201, 300),
    (605.0, 804.0, 301, 400),
    (805.0, 1004.0, 401, 500),
)


def aqi_score_from_openweather_level(level: int) -> int:
    return AQI_SCORE_BY_OPENWEATHER_LEVEL.get(int(level), 0)


def openweather_level_from_score(score: float) -> int:
    if score <= 50:
        return 1
    if score <= 100:
        return 2
    if score <= 150:
        return 3
    if score <= 200:
        return 4
    if score <= 300:
        return 5
    return 6


def label_from_score(score: float) -> str:
    return US_AQI_LABELS[openweather_level_from_score(score)]


def color_from_score(score: float) -> str:
    level = openweather_level_from_score(score)
    return {
        1: "#2e7d32",
        2: "#f9a825",
        3: "#ef6c00",
        4: "#c62828",
        5: "#6a1b9a",
        6: "#7f1d1d",
    }[level]


def alert_level(score: float) -> str:
    if score >= 301:
        return "hazardous"
    if score >= 151:
        return "unhealthy"
    return "normal"


def _interpolate_aqi(concentration: float, breakpoints: Iterable[Breakpoint]) -> Optional[float]:
    for low_c, high_c, low_aqi, high_aqi in breakpoints:
        if low_c <= concentration <= high_c:
            return ((high_aqi - low_aqi) / (high_c - low_c)) * (concentration - low_c) + low_aqi
    return None


def _ugm3_to_ppb(value: float, molecular_weight: float) -> float:
    return value * 24.45 / molecular_weight


def _ugm3_to_ppm(value: float, molecular_weight: float) -> float:
    return _ugm3_to_ppb(value, molecular_weight) / 1000.0


def component_aqi_scores(components: Dict[str, float]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    pollutant_specs = {
        "pm2_5": ("pm2_5", PM25_BREAKPOINTS, lambda value: value),
        "pm10": ("pm10", PM10_BREAKPOINTS, lambda value: value),
        "o3": ("o3", O3_8H_PPM_BREAKPOINTS, lambda value: _ugm3_to_ppm(value, 48.00)),
        "co": ("co", CO_8H_PPM_BREAKPOINTS, lambda value: _ugm3_to_ppm(value, 28.01)),
        "no2": ("no2", NO2_1H_PPB_BREAKPOINTS, lambda value: _ugm3_to_ppb(value, 46.0055)),
        "so2": ("so2", SO2_1H_PPB_BREAKPOINTS, lambda value: _ugm3_to_ppb(value, 64.066)),
    }
    for name, (_, breakpoints, converter) in pollutant_specs.items():
        raw = components.get(name)
        if raw is None:
            continue
        try:
            converted = converter(float(raw))
        except (TypeError, ValueError):
            continue
        score = _interpolate_aqi(converted, breakpoints)
        if score is not None:
            scores[name] = float(max(0.0, min(score, 500.0)))
    return scores


def aqi_score_from_components(components: Dict[str, float]) -> Optional[int]:
    scores = component_aqi_scores(components)
    if not scores:
        return None
    return int(round(max(scores.values())))


def primary_pollutant_from_components(components: Dict[str, float]) -> str:
    scores = component_aqi_scores(components)
    if not scores:
        return "unknown"
    return max(scores, key=scores.get)
