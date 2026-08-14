"""
Error analysis for the AQI Predictor.

Analyzes the tuned Random Forest model on the final
chronological test set.

For each forecast horizon:
    +24h
    +48h
    +72h

The script reports:
    - MAE
    - RMSE
    - R²
    - Mean error / bias
    - Largest absolute errors
    - Error by AQI category
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/training_data.parquet"

TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "pm25",
    "aqi",
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
]


# Best parameters found during Random Forest tuning
BEST_PARAMS = {
    24: {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    48: {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    72: {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
}


# ============================================================
# AQI category helper
# ============================================================

def aqi_category(aqi_value):
    """
    Approximate category based on AQI value.

    These categories are used only for error analysis.
    """

    if pd.isna(aqi_value):
        return "Unknown"

    if aqi_value <= 50:
        return "Good"

    if aqi_value <= 100:
        return "Moderate"

    if aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi_value <= 200:
        return "Unhealthy"

    if aqi_value <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# Prepare data
# ============================================================

def prepare_data(df, target_column):
    """Prepare X and y for one forecast horizon."""

    required = [
    "timestamp",
    "city",
    target_column,
    *FEATURE_COLUMNS,
]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = df[required].copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data = data.dropna(
        subset=required
    )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return data


# ============================================================
# Split
# ============================================================

def split_data(data):
    """Chronological 80/20 split."""

    split_index = int(
        len(data) * (1 - TEST_SIZE)
    )

    train = data.iloc[
        :split_index
    ].copy()

    test = data.iloc[
        split_index:
    ].copy()

    return train, test


# ============================================================
# Analyze horizon
# ============================================================

def analyze_horizon(df, horizon):
    """Train tuned Random Forest and analyze test errors."""

    target_column = (
        f"target_aqi_{horizon}h"
    )

    logger.info(
        "Starting error analysis for +%dh",
        horizon,
    )

    data = prepare_data(
        df,
        target_column,
    )

    train, test = split_data(data)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[target_column]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[target_column]

    model = RandomForestRegressor(
        **BEST_PARAMS[horizon],
        random_state=42,
        n_jobs=-1,
    )

    logger.info(
        "Training tuned Random Forest for +%dh...",
        horizon,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    analysis = test[
        [
            "timestamp",
            "city",
            "aqi",
            target_column,
        ]
    ].copy()

    analysis["prediction"] = predictions

    analysis["error"] = (
        analysis[target_column]
        - analysis["prediction"]
    )

    analysis["absolute_error"] = (
        analysis["error"].abs()
    )

    analysis["percentage_error"] = (
        analysis["absolute_error"]
        / analysis[target_column].abs().clip(lower=1)
        * 100
    )

    analysis["aqi_category"] = (
        analysis[target_column]
        .apply(aqi_category)
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    mean_error = analysis[
        "error"
    ].mean()

    print(
        "\n" + "=" * 70
    )

    print(
        f"ERROR ANALYSIS — +{horizon}H"
    )

    print(
        "=" * 70
    )

    print(
        f"MAE : {mae:.3f}"
    )

    print(
        f"RMSE: {rmse:.3f}"
    )

    print(
        f"R2  : {r2:.3f}"
    )

    print(
        f"Mean Error (bias): {mean_error:.3f}"
    )

    # --------------------------------------------------------
    # Largest errors
    # --------------------------------------------------------

    print(
        "\nTOP 10 LARGEST ERRORS"
    )

    largest_errors = analysis.sort_values(
        "absolute_error",
        ascending=False,
    ).head(10)

    print(
        largest_errors[
            [
                "timestamp",
                "city",
                "aqi",
                target_column,
                "prediction",
                "error",
                "absolute_error",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Error by AQI category
    # --------------------------------------------------------

    print(
        "\nERROR BY AQI CATEGORY"
    )

    category_analysis = (
        analysis
        .groupby("aqi_category")
        .agg(
            samples=(
                "absolute_error",
                "count",
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean",
            ),
            mean_error=(
                "error",
                "mean",
            ),
        )
        .reset_index()
    )

    category_analysis = category_analysis.sort_values(
        "mean_absolute_error",
        ascending=False,
    )

    print(
        category_analysis.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Error by city
    # --------------------------------------------------------

    print(
        "\nERROR BY CITY"
    )

    city_analysis = (
        analysis
        .groupby("city")
        .agg(
            samples=(
                "absolute_error",
                "count",
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean",
            ),
            mean_error=(
                "error",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_absolute_error",
            ascending=False,
        )
    )

    print(
        city_analysis.to_string(
            index=False
        )
    )

    return analysis


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "=" * 70
    )

    logger.info(
        "AQI MODEL ERROR ANALYSIS"
    )

    logger.info(
        "=" * 70
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    all_results = {}

    for horizon in [24, 48, 72]:

        all_results[
            f"{horizon}h"
        ] = analyze_horizon(
            df,
            horizon,
        )

    logger.info(
        "Error analysis completed."
    )


if __name__ == "__main__":
    main()