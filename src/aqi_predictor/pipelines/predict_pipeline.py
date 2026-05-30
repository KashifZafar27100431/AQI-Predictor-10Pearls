from __future__ import annotations

import argparse
import json

from aqi_predictor.config.settings import get_settings
from aqi_predictor.services.prediction import PredictionService


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def run(horizon: int = 72, sample: bool = False) -> dict:
    _load_dotenv()
    settings = get_settings()
    service = PredictionService(settings)
    return service.predict(horizon=horizon, sample=sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference for the next AQI horizon.")
    parser.add_argument("--horizon", type=int, default=72, help="Forecast horizon in hours.")
    parser.add_argument("--sample", action="store_true", help="Use deterministic sample forecast inputs.")
    args = parser.parse_args()
    print(json.dumps(run(horizon=args.horizon, sample=args.sample), indent=2))


if __name__ == "__main__":
    main()

