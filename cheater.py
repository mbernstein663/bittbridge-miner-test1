from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from bittbridge.utils.timestamp import get_now, round_to_interval, to_datetime, to_str


ISO_NE_BASE_URL = "https://webservices.iso-ne.com/api/v1.1"


class CheaterForecastError(RuntimeError):
    """Raised when the ISO-NE hourly forecast cannot produce a target prediction."""


def _get_credentials() -> tuple[str, str]:
    load_dotenv()
    username = os.getenv("ISO_NE_USERNAME")
    password = os.getenv("ISO_NE_PASSWORD")
    if not username or not password:
        raise CheaterForecastError(
            "Missing ISO_NE_USERNAME or ISO_NE_PASSWORD in environment/.env."
        )
    return username, password


def _extract_forecast_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    container = payload.get("HourlyLoadForecasts", payload)
    forecasts = container.get("HourlyLoadForecast") if isinstance(container, dict) else None
    if forecasts is None:
        raise CheaterForecastError("ISO-NE response did not include HourlyLoadForecast rows.")
    if isinstance(forecasts, dict):
        return [forecasts]
    if isinstance(forecasts, list):
        return forecasts
    raise CheaterForecastError("ISO-NE HourlyLoadForecast payload had an unexpected shape.")


def _fetch_hourly_load_forecast(day_yyyymmdd: str) -> list[dict[str, Any]]:
    username, password = _get_credentials()
    url = f"{ISO_NE_BASE_URL}/hourlyloadforecast/day/{day_yyyymmdd}.json"
    response = requests.get(
        url,
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return _extract_forecast_rows(response.json())


def _fetch_forecast_rows_around(target_timestamp: str) -> list[dict[str, Any]]:
    target_dt = to_datetime(target_timestamp)
    target_day = target_dt.strftime("%Y%m%d")
    rows = _fetch_hourly_load_forecast(target_day)

    # Include the next day's first hourly point so late-evening 5-minute slots can
    # interpolate between 23:00 and 00:00. If unavailable, the target-day data may
    # still be sufficient for exact hourly timestamps.
    next_day = (target_dt + timedelta(days=1)).strftime("%Y%m%d")
    if next_day != target_day:
        try:
            rows.extend(_fetch_hourly_load_forecast(next_day))
        except Exception:
            pass

    return rows


def _interpolate_five_minute_prediction(
    forecast_rows: list[dict[str, Any]],
    target_timestamp: str,
) -> float:
    frame = pd.DataFrame(forecast_rows)
    required = {"BeginDate", "LoadMw"}
    missing = required.difference(frame.columns)
    if missing:
        raise CheaterForecastError(f"ISO-NE forecast rows missing columns: {sorted(missing)}")

    frame["BeginDate"] = pd.to_datetime(frame["BeginDate"], utc=True, errors="coerce")
    frame["LoadMw"] = pd.to_numeric(frame["LoadMw"], errors="coerce")
    frame = frame.dropna(subset=["BeginDate", "LoadMw"])
    if frame.empty:
        raise CheaterForecastError("ISO-NE forecast rows contained no usable BeginDate/LoadMw data.")

    hourly = (
        frame.groupby("BeginDate", sort=True)["LoadMw"]
        .last()
        .sort_index()
    )
    five_minute = hourly.resample("5min").interpolate(method="linear")

    target_dt = to_datetime(target_timestamp).replace(second=0, microsecond=0)
    target_index = pd.Timestamp(target_dt).tz_convert("UTC")
    if target_index not in five_minute.index:
        raise CheaterForecastError(
            "Validator timestamp was outside the ISO-NE hourly forecast interpolation range."
        )

    prediction = five_minute.loc[target_index]
    if pd.isna(prediction):
        raise CheaterForecastError("ISO-NE interpolation produced NaN for the validator timestamp.")
    return float(prediction)


def predict_load_mw_for_timestamp(target_timestamp: str) -> float:
    """
    Return the official ISO-NE hourly forecast interpolated to the validator timestamp.

    The validator sends timestamps six hours ahead on 5-minute boundaries. This
    function uses that timestamp directly instead of recalculating the horizon.
    """
    rows = _fetch_forecast_rows_around(target_timestamp)
    return _interpolate_five_minute_prediction(rows, target_timestamp)


def validator_target_timestamp() -> str:
    target_time = round_to_interval(get_now(), interval_minutes=5) + timedelta(hours=6)
    return to_str(target_time)


class CheaterHourlyForecastPredictor:
    def predict(self, timestamp: str) -> float | None:
        try:
            return predict_load_mw_for_timestamp(timestamp)
        except Exception:
            return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch ISO-NE hourly load forecast and interpolate to a validator timestamp."
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Validator timestamp to predict. Defaults to the repo validator's now+6h timestamp.",
    )
    args = parser.parse_args(argv)

    timestamp = args.timestamp or validator_target_timestamp()
    try:
        prediction = predict_load_mw_for_timestamp(timestamp)
    except Exception as exc:
        print(f"Failed to retrieve ISO-NE hourly forecast: {exc}", file=sys.stderr)
        return 1

    print(f"{prediction:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
