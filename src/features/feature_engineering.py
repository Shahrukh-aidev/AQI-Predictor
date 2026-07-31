"""
Feature engineering for the Pearls AQI Predictor project.

This module:

1. Normalizes city names.
2. Merges AQICN and OpenWeather observations.
3. Creates time-based features.
4. Creates timestamp-aware AQI historical features.
5. Creates pollutant-derived features.
6. Creates timestamp-aware future AQI targets.
7. Creates historical training features.

Historical training features are timestamp-aware so that
missing observations and large gaps do not create false
24h/48h/72h relationships.
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

    AQICN and weather timestamps are kept separately
    because they may not represent exactly the same
    observation time.

    Weather observations are matched to the nearest
    observation within 30 minutes.
    """

    logger.info(
        "Starting AQICN + OpenWeather merge."
    )

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

    missing_aqicn = (
        required_aqicn
        - set(aqicn.columns)
    )

    missing_weather = (
        required_weather
        - set(weather.columns)
    )

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
    # Remove invalid timestamps
    # --------------------------------------------------------

    aqicn = aqicn.dropna(
        subset=["aqicn_timestamp"]
    )

    weather = weather.dropna(
        subset=["weather_timestamp"]
    )

    # --------------------------------------------------------
    # Canonical AQI timestamp
    # --------------------------------------------------------

    aqicn["timestamp"] = (
        aqicn["aqicn_timestamp"]
    )

    # --------------------------------------------------------
    # Sort for merge_asof
    # --------------------------------------------------------

    aqicn = aqicn.sort_values(
        ["city", "aqicn_timestamp"]
    ).reset_index(drop=True)

    weather = weather.sort_values(
        ["city", "weather_timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Merge nearest weather observation
    # --------------------------------------------------------

    merged = pd.merge_asof(
        aqicn,
        weather,
        left_on="aqicn_timestamp",
        right_on="weather_timestamp",
        by="city",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )

    # --------------------------------------------------------
    # Remove rows without weather match
    # --------------------------------------------------------

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

    logger.info(
        "Creating time features."
    )

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

    out["day_of_week"] = (
        timestamp.dt.dayofweek
    )

    out["day_of_month"] = (
        timestamp.dt.day
    )

    out["month"] = (
        timestamp.dt.month
    )

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
    Create timestamp-aware historical AQI features.

    IMPORTANT
    ---------
    Lag features are based on exact timestamps.

    For example:

        aqi_lag_24h

    means:

        AQI at exactly timestamp - 24 hours.

    It does NOT mean the 24th previous available row.

    Therefore, if there is a 61-day gap, the 61-day-old
    AQI will NOT be incorrectly used as a 24h lag.

    Rolling statistics are also based on actual time windows.
    """

    logger.info(
        "Creating timestamp-aware AQI historical features."
    )

    out = df.copy()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    required = {
        "city",
        "timestamp",
        "aqi",
    }

    missing = (
        required
        - set(out.columns)
    )

    if missing:

        raise ValueError(
            f"AQI feature data missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Standardize timestamp
    # --------------------------------------------------------

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    out = out.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create exact AQI lookup
    # --------------------------------------------------------

    history = (
        out[
            [
                "city",
                "timestamp",
                "aqi",
            ]
        ]
        .drop_duplicates(
            subset=[
                "city",
                "timestamp",
            ],
            keep="last",
        )
        .set_index(
            [
                "city",
                "timestamp",
            ]
        )["aqi"]
        .sort_index()
    )

    # --------------------------------------------------------
    # Helper for exact timestamp lag
    # --------------------------------------------------------

    def get_lag(hours: int) -> pd.Series:

        requested_timestamps = (
            out["timestamp"]
            - pd.Timedelta(hours=hours)
        )

        lookup = pd.MultiIndex.from_arrays(
            [
                out["city"].to_numpy(),
                requested_timestamps.to_numpy(),
            ],
            names=[
                "city",
                "timestamp",
            ],
        )

        values = (
            history
            .reindex(lookup)
            .to_numpy()
        )

        return pd.Series(
            values,
            index=out.index,
            dtype=float,
        )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in [1, 3, 6, 12, 24]:

        out[f"aqi_lag_{lag}h"] = (
            get_lag(lag)
        )

    # --------------------------------------------------------
    # Timestamp-aware rolling features
    #
    # IMPORTANT:
    # We do NOT use groupby.apply() here.
    #
    # Instead, calculate features city-by-city and assign
    # them back using the original dataframe index.
    #
    # This prevents index/city-column corruption.
    # --------------------------------------------------------

    out["aqi_roll_mean_3h"] = np.nan
    out["aqi_roll_mean_6h"] = np.nan
    out["aqi_roll_mean_24h"] = np.nan
    out["aqi_roll_std_24h"] = np.nan

    for city, city_indices in out.groupby(
        "city",
        sort=False,
    ).groups.items():

        city_indices = list(city_indices)

        city_df = (
            out.loc[city_indices]
            .sort_values("timestamp")
        )

        series = (
            city_df
            .set_index("timestamp")["aqi"]
        )

        # Exclude current observation.
        previous = series.shift(1)

        roll_3 = (
            previous
            .rolling(
                "3h",
                min_periods=3,
            )
            .mean()
        )

        roll_6 = (
            previous
            .rolling(
                "6h",
                min_periods=6,
            )
            .mean()
        )

        roll_24 = (
            previous
            .rolling(
                "24h",
                min_periods=24,
            )
            .mean()
        )

        std_24 = (
            previous
            .rolling(
                "24h",
                min_periods=24,
            )
            .std()
        )

        out.loc[
            city_df.index,
            "aqi_roll_mean_3h",
        ] = roll_3.to_numpy()

        out.loc[
            city_df.index,
            "aqi_roll_mean_6h",
        ] = roll_6.to_numpy()

        out.loc[
            city_df.index,
            "aqi_roll_mean_24h",
        ] = roll_24.to_numpy()

        out.loc[
            city_df.index,
            "aqi_roll_std_24h",
        ] = std_24.to_numpy()

    # --------------------------------------------------------
    # AQI change
    #
    # Since lag_1h is timestamp-aware, this is also
    # timestamp-aware.
    # --------------------------------------------------------

    out["aqi_change_1h"] = (
        out["aqi"]
        - out["aqi_lag_1h"]
    )

    out["aqi_change_rate"] = (
        out["aqi_change_1h"]
        / out["aqi_lag_1h"].replace(
            0,
            np.nan,
        )
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
        (
            (pm10 > 0)
            & pm25.notna()
            & pm10.notna()
        ),
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
    Create a timestamp-aware future AQI target.

    horizon=1
        AQI exactly 1 hour later.

    horizon=24
        AQI exactly 24 hours later.

    horizon=48
        AQI exactly 48 hours later.

    horizon=72
        AQI exactly 72 hours later.

    If the exact future timestamp does not exist,
    the target is NaN.

    This prevents a row 61 days later from being treated
    as a 24h future target.
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

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    required = {
        "city",
        "timestamp",
        "aqi",
    }

    missing = (
        required
        - set(out.columns)
    )

    if missing:

        raise ValueError(
            f"Future target data missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Standardize timestamp
    # --------------------------------------------------------

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    out = out.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Exact future AQI lookup
    # --------------------------------------------------------

    future_history = (
        out[
            [
                "city",
                "timestamp",
                "aqi",
            ]
        ]
        .drop_duplicates(
            subset=[
                "city",
                "timestamp",
            ],
            keep="last",
        )
        .set_index(
            [
                "city",
                "timestamp",
            ]
        )["aqi"]
        .sort_index()
    )

    # --------------------------------------------------------
    # Requested future timestamp
    # --------------------------------------------------------

    future_timestamp = (
        out["timestamp"]
        + pd.Timedelta(hours=horizon)
    )

    lookup = pd.MultiIndex.from_arrays(
        [
            out["city"].to_numpy(),
            future_timestamp.to_numpy(),
        ],
        names=[
            "city",
            "timestamp",
        ],
    )

    out[
        f"target_aqi_{horizon}h"
    ] = (
        future_history
        .reindex(lookup)
        .to_numpy()
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

    This function is retained for the normal/current
    AQICN + weather workflow.
    """

    logger.info(
        "Starting complete feature engineering pipeline."
    )

    # --------------------------------------------------------
    # Step 1: Merge
    # --------------------------------------------------------

    merged = merge_dataframes(
        aqicn_df,
        weather_df,
    )

    # --------------------------------------------------------
    # Step 2: Time features
    # --------------------------------------------------------

    features = add_time_features(
        merged
    )

    # --------------------------------------------------------
    # Step 3: Historical AQI features
    # --------------------------------------------------------

    features = add_aqi_features(
        features
    )

    # --------------------------------------------------------
    # Step 4: Future target
    # --------------------------------------------------------

    features = add_future_target(
        features,
        horizon=target_horizon,
    )

    # --------------------------------------------------------
    # Step 5: AQI category
    # --------------------------------------------------------

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


# ============================================================
# Training Feature Pipeline
# ============================================================

def build_training_features(
    merged_df: pd.DataFrame,
    target_horizons: list = [24, 48, 72],
    max_gap_hours: int = 3,
) -> pd.DataFrame:
    """
    Transform merged historical AQI + weather data into
    a model-ready training dataset.

    Historical AQI values are never forward-filled.

    Lag features and future targets use exact timestamps.

    Large timestamp gaps invalidate historical features
    immediately after the gap.

    Steps
    -----
    1. Validate input.
    2. Standardize timestamps.
    3. Sort by city + timestamp.
    4. Remove rows with missing AQI.
    5. Add time features.
    6. Add timestamp-aware historical AQI features.
    7. Add timestamp-aware future targets.
    8. Add AQI category.
    9. Detect large timestamp gaps.
    10. Invalidate historical features after large gaps.
    11. Require valid 24h historical AQI.
    12. Require all future targets.
    13. Return final training dataframe.
    """

    logger.info(
        "Building historical training features."
    )

    df = merged_df.copy()

    # --------------------------------------------------------
    # Step 1: Validate input
    # --------------------------------------------------------

    required_columns = {
        "timestamp",
        "city",
        "aqi",
        "pm25",
        "pm10",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Training data missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Validate target horizons
    # --------------------------------------------------------

    if not target_horizons:

        raise ValueError(
            "target_horizons cannot be empty."
        )

    if any(
        int(h) <= 0
        for h in target_horizons
    ):

        raise ValueError(
            "All target horizons must be greater than zero."
        )

    if max_gap_hours <= 0:

        raise ValueError(
            "max_gap_hours must be greater than zero."
        )

    # --------------------------------------------------------
    # Step 2: Standardize timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    before = len(df)

    df = df.dropna(
        subset=["timestamp"]
    )

    dropped = before - len(df)

    if dropped > 0:

        logger.info(
            "Removed %d rows with invalid timestamps.",
            dropped,
        )

    # --------------------------------------------------------
    # Step 3: Sort
    # --------------------------------------------------------

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Step 4: Remove missing AQI
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=["aqi"]
    ).reset_index(drop=True)

    dropped = before - len(df)

    logger.info(
        "Removed %d rows with missing AQI.",
        dropped,
    )

    if df.empty:

        raise ValueError(
            "No rows remain after removing missing AQI."
        )

    # --------------------------------------------------------
    # Step 5: Time features
    # --------------------------------------------------------

    df = add_time_features(
        df
    )

    # --------------------------------------------------------
    # Step 6: Historical AQI features
    # --------------------------------------------------------

    df = add_aqi_features(
        df
    )

    # --------------------------------------------------------
    # Step 7: Future targets
    # --------------------------------------------------------

    for horizon in target_horizons:

        df = add_future_target(
            df,
            horizon=int(horizon),
        )

    # --------------------------------------------------------
    # Step 8: AQI category
    # --------------------------------------------------------

    df = add_aqi_category(
        df
    )

    # --------------------------------------------------------
    # Step 9: Detect large timestamp gaps
    # --------------------------------------------------------

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    gap = (
        df.groupby(
            "city",
            sort=False,
        )["timestamp"]
        .diff()
    )

    large_gap = (
        gap > pd.Timedelta(
            hours=max_gap_hours
        )
    )

    large_gap_count = int(
        large_gap.sum()
    )

    logger.info(
        "Found %d large timestamp gap(s) > %dh.",
        large_gap_count,
        max_gap_hours,
    )

    # --------------------------------------------------------
    # Step 10: Invalidate historical features after gaps
    #
    # Example:
    #
    # Feb 15 02:00 -> AQI 85
    #
    #       61 days missing
    #
    # Apr 17 04:00 -> AQI 120
    #
    # The Apr 17 row cannot use Feb 15 as recent history.
    #
    # Timestamp-aware lags are already NaN if the exact
    # timestamp does not exist.
    #
    # We additionally invalidate ALL historical features
    # immediately after any gap > max_gap_hours.
    # --------------------------------------------------------

    historical_feature_cols = [
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
    ]

    for column in historical_feature_cols:

        if column in df.columns:

            df.loc[
                large_gap,
                column,
            ] = np.nan

    # --------------------------------------------------------
    # Step 11: Require valid 24h historical AQI
    #
    # This removes:
    #
    # - beginning of each city sequence
    # - rows immediately after large gaps
    # - rows where exact timestamp -24h does not exist
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=["aqi_lag_24h"]
    ).reset_index(drop=True)

    dropped = before - len(df)

    logger.info(
        "Dropped %d rows without valid 24h "
        "historical AQI.",
        dropped,
    )

    # --------------------------------------------------------
    # Step 12: Require complete future targets
    # --------------------------------------------------------

    target_columns = [
        f"target_aqi_{int(h)}h"
        for h in target_horizons
    ]

    before = len(df)

    df = df.dropna(
        subset=target_columns
    ).reset_index(drop=True)

    dropped = before - len(df)

    logger.info(
        "Dropped %d rows without complete "
        "future targets.",
        dropped,
    )

    # --------------------------------------------------------
    # Step 13: Final sort
    # --------------------------------------------------------

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Final logging
    # --------------------------------------------------------

    city_counts = (
        df["city"]
        .value_counts()
        .to_dict()
    )

    logger.info(
        "Training features complete. "
        "Shape: %s | Cities: %s",
        df.shape,
        city_counts,
    )

    return df