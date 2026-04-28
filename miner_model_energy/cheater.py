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

from bittbridge.utils.timestamp import to_datetime


ISO_NE_BASE_URL = "https://webservices.iso-ne.com/api/v1.1"


class CheaterForecastError(RuntimeError):
    """Raised when ISO-NE hourly forecast data cannot produce a target prediction."""


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

    # Late-evening 5-minute targets need the following day's 00:00 hourly point
    # to interpolate after 23:00 without extrapolating.
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
    if "CreationDate" in frame.columns:
        frame["CreationDate"] = pd.to_datetime(frame["CreationDate"], utc=True, errors="coerce")
        frame = frame.sort_values(
            ["BeginDate", "CreationDate"],
            kind="stable",
            na_position="first",
        )

    frame = frame.dropna(subset=["BeginDate", "LoadMw"])
    if frame.empty:
        raise CheaterForecastError("ISO-NE forecast rows contained no usable BeginDate/LoadMw data.")

    hourly = frame.groupby("BeginDate", sort=True)["LoadMw"].last().sort_index()
    five_minute = hourly.resample("5min").interpolate(method="linear")

    target_dt = to_datetime(target_timestamp)
    target_index = pd.Timestamp(target_dt).tz_convert("UTC")
    if target_index not in five_minute.index:
        raise CheaterForecastError(
            "Validator timestamp is not on a 5-minute ISO-NE forecast interpolation slot."
        )

    prediction = five_minute.loc[target_index]
    if pd.isna(prediction):
        raise CheaterForecastError("ISO-NE interpolation produced NaN for the validator timestamp.")
    return float(prediction)


def predict_load_mw_for_timestamp(target_timestamp: str) -> float:
    """
    Return ISO-NE hourly LoadMw interpolated exactly at the validator timestamp.

    This function does not calculate or replace timestamps. The validator request
    timestamp is the single source of truth for the returned prediction.
    """
    if not target_timestamp:
        raise CheaterForecastError("Validator timestamp is required.")
    rows = _fetch_forecast_rows_around(target_timestamp)
    return _interpolate_five_minute_prediction(rows, target_timestamp)


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
        required=True,
        help="Validator timestamp to predict. The script never generates its own timestamp.",
    )
    args = parser.parse_args(argv)

    try:
        prediction = predict_load_mw_for_timestamp(args.timestamp)
    except Exception as exc:
        print(f"Failed to retrieve ISO-NE hourly forecast: {exc}", file=sys.stderr)
        return 1

    print(f"{prediction:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
