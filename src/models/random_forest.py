"""
Random Forest Model for AQI Predictor.

Forecast horizons
-----------------
24h -> target_aqi_24h
48h -> target_aqi_48h
72h -> target_aqi_72h

Uses a chronological 80/20 train-test split and the same
feature set used by Ridge Regression.
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

N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
RANDOM_STATE = 42
N_JOBS = -1


# ============================================================
# Features
# ============================================================

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


# ============================================================
# Evaluation
# ============================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE and R²."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
    }


# ============================================================
# Model
# ============================================================

class RandomForestAQI:
    """Random Forest regression model for AQI forecasting."""

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split: int = MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
        random_state: int = RANDOM_STATE,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=N_JOBS,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the Random Forest."""

        logger.info(
            "Training Random Forest | trees=%d",
            self.model.n_estimators,
        )

        self.model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""

        return self.model.predict(X)


# ============================================================
# Data preparation
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    target_column: str,
):
    """Prepare features and target for one horizon."""

    required_columns = [
        "timestamp",
        target_column,
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[required_columns].copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data = data.dropna(
        subset=required_columns
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

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
):
    """Split the data without shuffling."""

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# One horizon
# ============================================================

def run_random_forest_for_horizon(
    df: pd.DataFrame,
    horizon: int,
):
    """Train/evaluate Random Forest for one horizon."""

    target_column = (
        f"target_aqi_{horizon}h"
    )

    logger.info(
        "Preparing Random Forest for +%dh...",
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
    ) = chronological_split(X, y)

    logger.info(
        "Horizon +%dh | train=%d test=%d features=%d",
        horizon,
        len(X_train),
        len(X_test),
        X_train.shape[1],
    )

    model = RandomForestAQI()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = evaluate(
        y_test.to_numpy(),
        predictions,
    )

    logger.info(
        "Random Forest +%dh | MAE=%.3f RMSE=%.3f R2=%.3f",
        horizon,
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )

    return metrics


# ============================================================
# All horizons
# ============================================================

def run_random_forest(df: pd.DataFrame) -> dict:
    """Run Random Forest for 24h, 48h and 72h."""

    if df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    results = {}

    for horizon in [24, 48, 72]:

        target_column = (
            f"target_aqi_{horizon}h"
        )

        if target_column not in df.columns:
            logger.warning(
                "Column %s not found — skipping.",
                target_column,
            )
            continue

        results[
            f"{horizon}h"
        ] = run_random_forest_for_horizon(
            df,
            horizon,
        )

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    logger.info(
        "=" * 60
    )

    logger.info(
        "AQI RANDOM FOREST"
    )

    logger.info(
        "=" * 60
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    logger.info(
        "Loaded %d rows and %d columns.",
        len(df),
        len(df.columns),
    )

    results = run_random_forest(df)

    print(
        "\n" + "=" * 50
    )

    print(
        "RANDOM FOREST RESULTS"
    )

    print(
        "=" * 50
    )

    for horizon, metrics in results.items():

        print(
            f"\nHorizon +{horizon}"
        )

        for metric, value in metrics.items():

            print(
                f"  {metric:6s}: {value}"
            )

    print(
        "=" * 50
    )

    print(
        "\nCompare these results against "
        "the Naive Baseline and Ridge Regression."
    )