"""
Feature engineering for the Pearls AQI Predictor project.

This module:

1. Normalizes city names.
2. Merges AQICN and OpenWeather observations.
3. Creates time-based features.
4. Creates AQI historical features.
5. Creates pollutant-derived features.
6. Creates future AQI targets.

Historical lag/rolling features require historical data.
"""

import numpy as np
import pandas as pd

from src.utils.logger import logger


# ============================================================
# City Normalization
# ============================================================

CITY_NAME_MAP = {
    "lahore us embassy, pakistan": "Lahore",
    "karachi, pakistan": "Karachi",
    "islamabad, pakistan": "Islamabad",
    "sukkur, pakistan": "Sukkur",
}


def normalise_city(name: str) -> str:
    """
    Convert API-specific city/station names into standard names.
    """

    if not isinstance(name, str):
        return str(name)

    key = name.strip().lower()

    return CITY_NAME_MAP.get(
        key,
        name.strip().title(),
    )


# ============================================================
# Merge AQICN + OpenWeather
# ============================================================

def merge_dataframes(
    aqicn_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge AQICN and OpenWeather observations.

    AQICN and OpenWeather timestamps are kept separately
    because they may not represent exactly the same observation time.
    """

    logger.info("Starting AQICN + OpenWeather merge.")

    aqicn = aqicn_df.copy()
    weather = weather_df.copy()

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_aqicn = {
        "timestamp",
        "city",
        "aqi",
    }

    required_weather = {
        "timestamp",
        "city",
    }

    missing_aqicn = required_aqicn - set(aqicn.columns)

    missing_weather = required_weather - set(weather.columns)

    if missing_aqicn:
        raise ValueError(
            f"AQICN data missing columns: {missing_aqicn}"
        )

    if missing_weather:
        raise ValueError(
            f"Weather data missing columns: {missing_weather}"
        )

    # --------------------------------------------------------
    # Normalize city names
    # --------------------------------------------------------

    aqicn["city"] = aqicn["city"].apply(
        normalise_city
    )

    weather["city"] = weather["city"].apply(
        normalise_city
    )

    # --------------------------------------------------------
    # Standardize timestamps
    # --------------------------------------------------------

    aqicn["aqicn_timestamp"] = pd.to_datetime(
        aqicn["timestamp"],
        errors="coerce",
        utc=True,
    )

    weather["weather_timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="coerce",
        utc=True,
    )

    aqicn = aqicn.drop(
        columns=["timestamp"]
    )

    weather = weather.drop(
        columns=["timestamp"]
    )

    # --------------------------------------------------------
    # Create canonical timestamp
    #
    # For the current pipeline, use AQICN's observation time.
    # Historical pipeline will later align observations hourly.
    # --------------------------------------------------------

    aqicn["timestamp"] = aqicn["aqicn_timestamp"]

    # --------------------------------------------------------
    # Merge observations by city + nearest timestamp
    # --------------------------------------------------------

    aqicn = aqicn.sort_values(
        ["city", "aqicn_timestamp"]
    )

    weather = weather.sort_values(
        ["city", "weather_timestamp"]
    )

    merged = pd.merge_asof(
        aqicn,
        weather,
        left_on="aqicn_timestamp",
        right_on="weather_timestamp",
        by="city",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )

    # Remove rows where no weather observation was close enough
    merged = merged.dropna(
        subset=["weather_timestamp"]
    ).reset_index(drop=True)

    if merged.empty:
        logger.warning(
            "AQICN/OpenWeather merge produced zero rows."
        )
    else:
        logger.info(
            "Successfully merged %d row(s).",
            len(merged),
        )

    return merged


# ============================================================
# Time Features
# ============================================================

def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar and cyclical time features.
    """

    logger.info("Creating time features.")

    out = df.copy()

    if "timestamp" not in out.columns:

        raise ValueError(
            "'timestamp' column is required."
        )

    timestamp = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    out["hour"] = timestamp.dt.hour

    out["day_of_week"] = timestamp.dt.dayofweek

    out["day_of_month"] = timestamp.dt.day

    out["month"] = timestamp.dt.month

    out["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical encoding
    # --------------------------------------------------------

    out["hour_sin"] = np.sin(
        2 * np.pi * out["hour"] / 24
    )

    out["hour_cos"] = np.cos(
        2 * np.pi * out["hour"] / 24
    )

    out["day_sin"] = np.sin(
        2 * np.pi * out["day_of_week"] / 7
    )

    out["day_cos"] = np.cos(
        2 * np.pi * out["day_of_week"] / 7
    )

    out["month_sin"] = np.sin(
        2 * np.pi * out["month"] / 12
    )

    out["month_cos"] = np.cos(
        2 * np.pi * out["month"] / 12
    )

    return out


# ============================================================
# AQI Historical Features
# ============================================================

def add_aqi_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical AQI features.

    These features require a genuine historical time series.
    """

    logger.info("Creating AQI historical features.")

    out = df.copy()

    if "aqi" not in out.columns:

        raise ValueError(
            "'aqi' column is required."
        )

    out = out.sort_values(
        ["city", "timestamp"]
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in [1, 3, 6, 12, 24]:

        out[f"aqi_lag_{lag}h"] = (
            out.groupby("city")["aqi"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    grouped = out.groupby("city")["aqi"]

    out["aqi_roll_mean_3h"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    out["aqi_roll_mean_6h"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(6)
            .mean()
        )
    )

    out["aqi_roll_mean_24h"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(24)
            .mean()
        )
    )

    out["aqi_roll_std_24h"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(24)
            .std()
        )
    )

    # --------------------------------------------------------
    # AQI change
    # --------------------------------------------------------

    previous_aqi = (
        out.groupby("city")["aqi"]
        .shift(1)
    )

    out["aqi_change_1h"] = (
        out["aqi"] - previous_aqi
    )

    out["aqi_change_rate"] = (
        out["aqi_change_1h"]
        / previous_aqi.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # PM2.5 / PM10 ratio
    # --------------------------------------------------------

    pm25 = pd.to_numeric(
        out["pm25"],
        errors="coerce",
    )

    pm10 = pd.to_numeric(
        out["pm10"],
        errors="coerce",
    )

    out["pm25_to_pm10_ratio"] = np.where(
        (pm10 > 0)
        & pm25.notna()
        & pm10.notna(),

        pm25 / pm10,

        np.nan,
    )

    return out


# ============================================================
# Future Target
# ============================================================

def add_future_target(
    df: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Create a future AQI target.

    horizon=1  -> next hourly AQI
    horizon=24 -> AQI 24 hours later
    horizon=48 -> AQI 48 hours later
    horizon=72 -> AQI 72 hours later
    """

    if horizon <= 0:

        raise ValueError(
            "Horizon must be greater than zero."
        )

    logger.info(
        "Creating future AQI target: +%d hours.",
        horizon,
    )

    out = df.copy()

    out = out.sort_values(
        ["city", "timestamp"]
    )

    out[f"target_aqi_{horizon}h"] = (
        out.groupby("city")["aqi"]
        .shift(-horizon)
    )

    return out


# ============================================================
# AQI Category
# ============================================================

def add_aqi_category(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add AQI category for dashboard/analytics purposes.

    This should NOT be used as a model input feature.
    """

    out = df.copy()

    out["aqi_category"] = (
        out["aqi"].apply(
            _aqi_category
        )
    )

    return out


# ============================================================
# Complete Feature Pipeline
# ============================================================

def build_features(
    aqicn_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    target_horizon: int = 1,
) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    """

    logger.info(
        "Starting complete feature engineering pipeline."
    )

    # Step 1
    merged = merge_dataframes(
        aqicn_df,
        weather_df,
    )

    # Step 2
    features = add_time_features(
        merged
    )

    # Step 3
    features = add_aqi_features(
        features
    )

    # Step 4
    features = add_future_target(
        features,
        horizon=target_horizon,
    )

    # Step 5
    features = add_aqi_category(
        features
    )

    logger.info(
        "Feature engineering complete. Shape: %s",
        features.shape,
    )

    return features


# ============================================================
# AQI Category Helper
# ============================================================

def _aqi_category(
    aqi_value,
) -> str:
    """
    Convert numeric AQI into an AQI category.
    """

    try:

        value = float(aqi_value)

    except (TypeError, ValueError):

        return "Unknown"

    if value <= 50:

        return "Good"

    if value <= 100:

        return "Moderate"

    if value <= 150:

        return "Unhealthy for Sensitive Groups"

    if value <= 200:

        return "Unhealthy"

    if value <= 300:

        return "Very Unhealthy"

    return "Hazardous"