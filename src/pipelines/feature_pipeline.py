"""
Hourly AQI Feature Pipeline.

Purpose
-------
Build fresh inference-time AQI/weather features and upload
the latest usable rows to the Hopsworks Feature Store.

Flow
----
OpenAQ PM2.5
    +
Open-Meteo weather
    ↓
AQI calculation
    ↓
Time features
    ↓
Timestamp-aware AQI historical features
    ↓
Latest usable rows
    ↓
Hopsworks Feature Store

Important
---------
This is an inference-time feature pipeline.

It does NOT calculate future AQI targets because future AQI
values are not available yet.

However, the target columns are preserved as NULL/NaN so that
the dataframe remains compatible with the existing Hopsworks
Feature Group schema.

The historical training pipeline remains separate.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Optional

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from src.features.aqi_calculator import add_aqi_from_pm25
from src.features.feature_engineering import (
    add_aqi_category,
    add_aqi_features,
    add_time_features,
)
from src.features.openaq_client import OpenAQClient
from src.features.openmeteo_client import OpenMeteoClient
from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
]

LOOKBACK_DAYS = 30
UPLOAD_LOOKBACK_HOURS = 6

FEATURE_GROUP_NAME = "aqi_features"

FEATURE_GROUP_VERSION = int(
    os.getenv(
        "HOPSWORKS_FEATURE_GROUP_VERSION",
        "6",
    )
)

MAX_HOPS_RETRIES = 3
HOPS_RETRY_DELAY_SECONDS = 10
WEATHER_MATCH_TOLERANCE_MINUTES = 30


# ============================================================
# Feature Group Schema
# ============================================================

LIVE_FEATURE_COLUMNS = [
    "timestamp",
    "city",
    "pm25",
    "aqi",
    "aqi_category",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "rain_1h",
    "rain_3h",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_roll_mean_3h",
    "aqi_roll_mean_6h",
    "aqi_roll_mean_24h",
    "aqi_roll_std_24h",
    "aqi_change_1h",
    "aqi_change_rate",
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


# ============================================================
# Dtype schema
# ============================================================

FLOAT_COLUMNS = [
    "pm25",
    "aqi",
    "temperature",
    "pressure",
    "wind_speed",
    "rain_1h",
    "rain_3h",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_roll_mean_3h",
    "aqi_roll_mean_6h",
    "aqi_roll_mean_24h",
    "aqi_roll_std_24h",
    "aqi_change_1h",
    "aqi_change_rate",
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
    # Cyclical features
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]

# int32 columns
INT32_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
]

# int64 columns — FG v6 schema stored these as bigint
# because the original training_data.parquet had them
# as integers (humidity = whole %, wind_direction = whole °)
INT64_COLUMNS = [
    "humidity",
    "wind_direction",
]


# ============================================================
# Utility: dtype normalization
# ============================================================

def normalize_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize dataframe dtypes to match Hopsworks
    Feature Group v6 schema exactly.

    Key fix: humidity and wind_direction are stored as
    bigint (int64) in FG v6 because the original training
    data uploaded them as integers. Sending them as float64
    causes a schema compatibility error.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="raise",
        )

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    if "city" in df.columns:
        df["city"] = df["city"].astype(str)

    # --------------------------------------------------------
    # Float columns
    # --------------------------------------------------------

    for column in FLOAT_COLUMNS:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .astype("float64")
            )

    # --------------------------------------------------------
    # int32 columns
    # --------------------------------------------------------

    for column in INT32_COLUMNS:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .fillna(0)
                .astype("int32")
            )

    # --------------------------------------------------------
    # int64 columns (bigint in FG v6)
    # --------------------------------------------------------

    for column in INT64_COLUMNS:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .fillna(0)
                .round()
                .astype("int64")
            )

    # --------------------------------------------------------
    # AQI category
    # --------------------------------------------------------

    if "aqi_category" in df.columns:
        df["aqi_category"] = (
            df["aqi_category"].astype("string")
        )

    return df


# ============================================================
# Utility: schema validation
# ============================================================

def validate_feature_schema(
    df: pd.DataFrame,
) -> None:
    """Validate all Feature Group columns are present."""

    missing = [
        column
        for column in LIVE_FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Live feature dataframe is missing required "
            f"columns: {missing}"
        )


# ============================================================
# Utility: log dtypes
# ============================================================

def log_feature_dtypes(
    df: pd.DataFrame,
) -> None:
    """Log final feature dtypes for debugging."""

    logger.info(
        "Final upload dtypes:\n%s",
        df.dtypes.to_string(),
    )


# ============================================================
# Step 1 — Fetch AQI
# ============================================================

def fetch_aqi(
    cities: list[str],
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """
    Fetch recent PM2.5 observations from OpenAQ.
    Failed cities are skipped, not errored.
    """

    client = OpenAQClient()
    frames: list[pd.DataFrame] = []

    for city in cities:

        try:

            logger.info(
                "OpenAQ: fetching %s -> %s for %s",
                date_from,
                date_to,
                city,
            )

            df = client.fetch_city_historical(
                city=city,
                date_from=date_from,
                date_to=date_to,
            )

            if df.empty:
                logger.warning(
                    "OpenAQ returned no data for %s.", city
                )
                continue

            df["timestamp"] = pd.to_datetime(
                df["timestamp"], utc=True, errors="coerce"
            )
            df["pm25"] = pd.to_numeric(
                df["pm25"], errors="coerce"
            )

            df = df.dropna(subset=["timestamp"])
            df.loc[df["pm25"] <= 0, "pm25"] = pd.NA
            df = df.dropna(subset=["pm25"])

            if df.empty:
                logger.warning(
                    "No valid PM2.5 rows for %s.", city
                )
                continue

            df = (
                df
                .drop_duplicates(
                    subset=["city", "timestamp"],
                    keep="last",
                )
                .sort_values(["city", "timestamp"])
                .reset_index(drop=True)
            )

            frames.append(df)

            logger.info(
                "OpenAQ: %d usable rows for %s.",
                len(df),
                city,
            )

        except Exception as exc:
            logger.error(
                "OpenAQ failed for %s: %s", city, exc
            )

    if not frames:
        raise RuntimeError(
            "No usable OpenAQ data was fetched for any city."
        )

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(
            subset=["city", "timestamp"], keep="last"
        )
        .sort_values(["city", "timestamp"])
        .reset_index(drop=True)
    )

    logger.info(
        "OpenAQ complete: %d total usable rows.", len(result)
    )
    return result


# ============================================================
# Step 2 — Fetch weather
# ============================================================

def fetch_weather(
    cities: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo.
    Failed cities are skipped, not errored.
    """

    client = OpenMeteoClient()
    frames: list[pd.DataFrame] = []

    for city in cities:

        try:

            logger.info(
                "Open-Meteo: fetching %s -> %s for %s",
                start_date,
                end_date,
                city,
            )

            df = client.fetch_historical(
                city=city,
                start_date=start_date,
                end_date=end_date,
            )

            if df.empty:
                logger.warning(
                    "Open-Meteo returned no data for %s.",
                    city,
                )
                continue

            df["timestamp"] = pd.to_datetime(
                df["timestamp"], utc=True, errors="coerce"
            )
            df = df.dropna(subset=["timestamp"])

            df = (
                df
                .drop_duplicates(
                    subset=["city", "timestamp"],
                    keep="last",
                )
                .sort_values(["city", "timestamp"])
                .reset_index(drop=True)
            )

            frames.append(df)

            logger.info(
                "Open-Meteo: %d rows for %s.",
                len(df),
                city,
            )

        except Exception as exc:
            logger.error(
                "Open-Meteo failed for %s: %s", city, exc
            )

    if not frames:
        raise RuntimeError(
            "No usable weather data was fetched."
        )

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(
            subset=["city", "timestamp"], keep="last"
        )
        .sort_values(["city", "timestamp"])
        .reset_index(drop=True)
    )

    logger.info(
        "Open-Meteo complete: %d total rows.", len(result)
    )
    return result


# ============================================================
# Step 3 — Merge AQI + weather
# ============================================================

def merge_aqi_weather(
    aqi_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nearest-timestamp merge per city with 30-min tolerance.
    """

    frames: list[pd.DataFrame] = []

    for city in sorted(
        aqi_df["city"].dropna().astype(str).unique()
    ):

        city_aqi = (
            aqi_df[aqi_df["city"] == city]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        city_weather = (
            weather_df[weather_df["city"] == city]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if city_aqi.empty:
            continue

        if city_weather.empty:
            logger.warning(
                "No weather data for %s — skipping.", city
            )
            continue

        city_weather = city_weather.drop(
            columns=["city"], errors="ignore"
        )

        merged = pd.merge_asof(
            city_aqi,
            city_weather,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(
                minutes=WEATHER_MATCH_TOLERANCE_MINUTES
            ),
        )

        required_weather = [
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "rain_1h",
            "rain_3h",
        ]

        missing_weather = [
            c for c in required_weather
            if c not in merged.columns
        ]

        if missing_weather:
            logger.warning(
                "%s missing weather columns: %s",
                city,
                missing_weather,
            )
            continue

        merged = merged.dropna(subset=required_weather)

        if merged.empty:
            logger.warning(
                "No valid AQI/weather matches for %s.", city
            )
            continue

        merged["city"] = city
        frames.append(merged)

        logger.info(
            "Merged %d rows for %s.", len(merged), city
        )

    if not frames:
        raise RuntimeError(
            "No AQI/weather city merges succeeded."
        )

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["city", "timestamp"])
        .reset_index(drop=True)
    )


# ============================================================
# Step 4 — Build features
# ============================================================

def build_features(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build inference-time features.
    Target columns are set to NaN (future is unknown).
    """

    df = merged_df.copy()

    # --------------------------------------------------------
    # PM2.5
    # --------------------------------------------------------

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df.loc[df["pm25"] <= 0, "pm25"] = pd.NA
    df = df.dropna(subset=["pm25"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No valid PM2.5 observations remain."
        )

    # --------------------------------------------------------
    # AQI
    # --------------------------------------------------------

    df = add_aqi_from_pm25(df)
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No valid AQI observations remain."
        )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = add_time_features(df)
    df = add_aqi_features(df)
    df = add_aqi_category(df)

    # --------------------------------------------------------
    # Target columns — NaN at inference time
    # --------------------------------------------------------

    for col in ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).astype("float64")

    # --------------------------------------------------------
    # Validate + select schema columns
    # --------------------------------------------------------

    validate_feature_schema(df)
    df = df[LIVE_FEATURE_COLUMNS].copy()

    # --------------------------------------------------------
    # Normalize dtypes
    # --------------------------------------------------------

    df = normalize_dtypes(df)

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["city", "timestamp"], keep="last"
        )
        .sort_values(["city", "timestamp"])
        .reset_index(drop=True)
    )

    logger.info(
        "Feature engineering complete: %d rows x %d columns.",
        len(df),
        len(df.columns),
    )
    return df


# ============================================================
# Step 5 — Select fresh rows
# ============================================================

def select_fresh_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select only the newest rows for upload.
    Falls back to latest row per city if nothing is fresh.
    """

    if df.empty:
        raise RuntimeError("Feature dataframe is empty.")

    cutoff = (
        pd.Timestamp.now(tz="UTC")
        - pd.Timedelta(hours=UPLOAD_LOOKBACK_HOURS)
    )

    fresh = df[df["timestamp"] >= cutoff].copy()

    if fresh.empty:
        logger.warning(
            "No rows newer than %s — using latest per city.",
            cutoff,
        )
        fresh = (
            df
            .sort_values(["city", "timestamp"])
            .groupby("city", as_index=False, group_keys=False)
            .tail(1)
        )

    fresh = (
        fresh
        .drop_duplicates(
            subset=["city", "timestamp"], keep="last"
        )
        .sort_values(["city", "timestamp"])
        .reset_index(drop=True)
    )

    logger.info(
        "Selected %d fresh rows for upload.", len(fresh)
    )
    return fresh


# ============================================================
# Step 6 — Upload to Hopsworks
# ============================================================

def upload_to_feature_store(
    df: pd.DataFrame,
    project,
) -> None:
    """
    Upload fresh live features to Hopsworks Feature Store.
    Retries transient failures up to MAX_HOPS_RETRIES times.
    """

    if df.empty:
        raise RuntimeError("Cannot upload an empty dataframe.")

    fs = project.get_feature_store()

    logger.info(
        "Opening Feature Group '%s' version %s.",
        FEATURE_GROUP_NAME,
        FEATURE_GROUP_VERSION,
    )

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    # Final normalization + validation
    upload_df = normalize_dtypes(df.copy())
    validate_feature_schema(upload_df)

    upload_df = (
        upload_df
        .drop_duplicates(
            subset=["city", "timestamp"], keep="last"
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Uploading columns: %s",
        upload_df.columns.tolist(),
    )
    log_feature_dtypes(upload_df)

    logger.info(
        "Uploading %d rows to Hopsworks.", len(upload_df)
    )

    last_exception: Optional[Exception] = None

    for attempt in range(1, MAX_HOPS_RETRIES + 1):

        try:
            fg.insert(
                upload_df,
                write_options={"wait_for_job": True},
                validation_options={
                    "run_validation": False,
                    "save_report": False,
                },
            )
            logger.info(
                "Hopsworks upload succeeded on attempt %d.",
                attempt,
            )
            return

        except Exception as exc:
            last_exception = exc
            logger.warning(
                "Hopsworks upload failed on attempt %d/%d: %s",
                attempt,
                MAX_HOPS_RETRIES,
                exc,
            )
            if attempt < MAX_HOPS_RETRIES:
                time.sleep(HOPS_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Hopsworks upload failed after {MAX_HOPS_RETRIES} attempts."
    ) from last_exception


# ============================================================
# Main pipeline
# ============================================================

def run_feature_pipeline() -> None:
    """Run the complete hourly feature pipeline."""

    logger.info("Starting hourly AQI feature pipeline...")

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is not set.")
    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT is not set.")

    # --------------------------------------------------------
    # Date range: last 30 days
    # --------------------------------------------------------

    today = date.today()
    date_from = today - timedelta(days=LOOKBACK_DAYS)

    logger.info(
        "Feature history window: %s -> %s", date_from, today
    )

    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    aqi_df = fetch_aqi(
        cities=CITIES,
        date_from=date_from.isoformat(),
        date_to=today.isoformat(),
    )

    weather_df = fetch_weather(
        cities=CITIES,
        start_date=date_from,
        end_date=today,
    )

    # --------------------------------------------------------
    # Merge + features
    # --------------------------------------------------------

    merged_df = merge_aqi_weather(aqi_df, weather_df)
    features_df = build_features(merged_df)
    upload_df = select_fresh_rows(features_df)

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    logger.info("Connecting to Hopsworks...")

    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name,
    )

    try:
        upload_to_feature_store(upload_df, project)
    finally:
        try:
            project.close()
        except Exception:
            pass

    logger.info(
        "Hourly AQI feature pipeline completed successfully."
    )


if __name__ == "__main__":
    run_feature_pipeline()