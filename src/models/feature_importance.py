"""
Feature importance analysis for the AQI Predictor.

Uses the tuned Random Forest configuration that currently
performs best on the AQI forecasting task.

Analyzes feature importance for:
    +24h
    +48h
    +72h

The script prints the top features for each horizon and
their correctly aligned average importance across horizons.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/training_data.parquet"

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

TEST_SIZE = 0.20

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
# Prepare data
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    target_column: str,
):
    """Prepare features and target."""

    required = [
        "timestamp",
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

    X = data[FEATURE_COLUMNS]
    y = data[target_column]

    return X, y


# ============================================================
# Chronological split
# ============================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """Split into chronological train and test sets."""

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Feature importance
# ============================================================

def analyze_horizon(
    df: pd.DataFrame,
    horizon: int,
):
    """Train tuned Random Forest and calculate feature importance."""

    target_column = (
        f"target_aqi_{horizon}h"
    )

    logger.info(
        "Analyzing feature importance for +%dh",
        horizon,
    )

    X, y = prepare_data(
        df,
        target_column,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    # y_test is intentionally kept here for a consistent
    # train/test split, although feature importance itself
    # is calculated from the trained model.
    _ = (
        X_test,
        y_test,
    )

    params = BEST_PARAMS[horizon]

    model = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
    )

    logger.info(
        "Training Random Forest for +%dh...",
        horizon,
    )

    model.fit(
        X_train,
        y_train,
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("AQI FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)

    df = pd.read_parquet(
        DATA_PATH
    )

    logger.info(
        "Loaded %d rows and %d columns.",
        len(df),
        len(df.columns),
    )

    results = {}

    # --------------------------------------------------------
    # Analyze each horizon
    # --------------------------------------------------------

    for horizon in [24, 48, 72]:

        importance_df = analyze_horizon(
            df,
            horizon,
        )

        results[f"{horizon}h"] = importance_df

        print(
            "\n" + "=" * 60
        )

        print(
            f"TOP 15 FEATURES — +{horizon}H"
        )

        print(
            "=" * 60
        )

        print(
            importance_df.head(15).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Combined average importance
    # --------------------------------------------------------

    combined = (
        results["24h"][
            ["feature", "importance"]
        ]
        .rename(
            columns={
                "importance": "importance_24h"
            }
        )
    )

    combined = combined.merge(
        results["48h"][
            ["feature", "importance"]
        ].rename(
            columns={
                "importance": "importance_48h"
            }
        ),
        on="feature",
    )

    combined = combined.merge(
        results["72h"][
            ["feature", "importance"]
        ].rename(
            columns={
                "importance": "importance_72h"
            }
        ),
        on="feature",
    )

    combined["mean_importance"] = combined[
        [
            "importance_24h",
            "importance_48h",
            "importance_72h",
        ]
    ].mean(axis=1)

    combined = combined.sort_values(
        "mean_importance",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Print overall importance
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "OVERALL FEATURE IMPORTANCE"
    )

    print(
        "=" * 60
    )

    print(
        combined.head(15).to_string(
            index=False
        )
    )

    print(
        "\nFeature importance analysis completed."
    )


if __name__ == "__main__":
    main()