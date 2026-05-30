from __future__ import annotations

import json
from typing import Any

import pandas as pd

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.aqi_scale import (
    aqi_score_from_components,
    label_from_score,
    primary_pollutant_from_components,
)
from aqi_predictor.services.features import (
    LAG_COLUMNS,
    POLLUTANT_COLUMNS,
    add_lag_features,
    add_time_features,
    normalize_numeric_features,
)
from aqi_predictor.services.storage import FEATURES_FEATURE_GROUP, FeatureStoreClient


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def _component_dict(row: pd.Series) -> dict[str, Any]:
    return {
        column: float(row[column]) if column in row and pd.notna(row[column]) else None
        for column in POLLUTANT_COLUMNS
    }


def repair() -> dict[str, int]:
    _load_dotenv()
    settings = get_settings()
    store = FeatureStoreClient(settings)
    frame = store.read_feature_group(FEATURES_FEATURE_GROUP)
    if frame.empty:
        return {
            "rows": 0,
            "rows_written": 0,
            "corrected_score_rows": 0,
            "corrected_primary_rows": 0,
        }

    repaired = frame.sort_values(["city", "event_time"]).copy()
    old_scores = pd.to_numeric(repaired["aqi_score"], errors="coerce").round()
    old_primary = repaired["primary_pollutant"].astype(str)
    old_categories = repaired["aqi_category"].astype(str)
    old_lag_features = repaired[LAG_COLUMNS].copy()

    new_scores = []
    new_primary = []
    for _, row in repaired.iterrows():
        components = _component_dict(row)
        score = aqi_score_from_components(components)
        if score is None:
            score = int(row.get("aqi_score", 0) or 0)
        new_scores.append(score)
        new_primary.append(primary_pollutant_from_components(components))

    repaired["aqi_score"] = new_scores
    repaired["aqi_category"] = [label_from_score(score) for score in new_scores]
    repaired["primary_pollutant"] = new_primary
    repaired = normalize_numeric_features(add_lag_features(add_time_features(repaired)))

    changed_mask = (
        (old_scores != pd.Series(new_scores, index=repaired.index))
        | (old_primary != pd.Series(new_primary, index=repaired.index))
        | (old_categories != repaired["aqi_category"].astype(str))
    )
    for column in LAG_COLUMNS:
        changed_mask = changed_mask | (
            pd.to_numeric(old_lag_features[column], errors="coerce").round(8)
            != pd.to_numeric(repaired[column], errors="coerce").round(8)
        )

    changed = repaired.loc[changed_mask].copy()
    if changed.empty:
        return {
            "rows": int(len(repaired)),
            "rows_written": 0,
            "corrected_score_rows": 0,
            "corrected_primary_rows": 0,
        }

    store.insert_feature_group(
        FEATURES_FEATURE_GROUP,
        changed,
        primary_key=["city", "event_time"],
    )

    return {
        "rows": int(len(repaired)),
        "rows_written": int(len(changed)),
        "corrected_score_rows": int((old_scores != pd.Series(new_scores, index=repaired.index)).sum()),
        "corrected_primary_rows": int((old_primary != pd.Series(new_primary, index=repaired.index)).sum()),
    }


def main() -> None:
    print(json.dumps(repair(), indent=2))


if __name__ == "__main__":
    main()
