"""
Ridge Regression Model for AQI Predictor.

Ridge regression uses multiple engineered AQI and weather features
to predict future AQI.

Forecast horizons
-----------------
24h -> target_aqi_24h
48h -> target_aqi_48h
72h -> target_aqi_72h

The model uses:
    StandardScaler -> Ridge Regression

A chronological 80/20 train-test split is used to avoid
future-data leakage.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/training_data.parquet"

TEST_SIZE = 0.20

RIDGE_ALPHA = 1.0


# ============================================================
# Feature configuration
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
# Metrics
# ============================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE and R².
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
    }


# ============================================================
# Ridge Model
# ============================================================

class RidgeAQI:
    """
    Ridge Regression model for AQI forecasting.

    The pipeline performs:

        StandardScaler
            ↓
        Ridge Regression

    Parameters
    ----------
    alpha : float
        Ridge regularization strength.
    """

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha = alpha

        self.model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=self.alpha,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Train the Ridge model.
        """

        logger.info(
            "Training Ridge Regression | alpha=%.2f",
            self.alpha,
        )

        self.model.fit(
            X,
            y,
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate AQI predictions.
        """

        return self.model.predict(X)


# ============================================================
# Data preparation
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    target_column: str,
):
    """
    Prepare features and target for one forecast horizon.
    """

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

    data = df[
        required_columns
    ].copy()

    # Sort chronologically
    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Remove missing values
    data = data.dropna(
        subset=[
            target_column,
            *FEATURE_COLUMNS,
        ]
    )

    X = data[
        FEATURE_COLUMNS
    ]

    y = data[
        target_column
    ]

    return X, y


# ============================================================
# Chronological split
# ============================================================

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Split data chronologically.

    First 80%  -> training
    Last 20%   -> testing
    """

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
# Run Ridge for one horizon
# ============================================================

def run_ridge_for_horizon(
    df: pd.DataFrame,
    horizon: int,
):
    """
    Train and evaluate Ridge Regression for one horizon.
    """

    target_column = (
        f"target_aqi_{horizon}h"
    )

    logger.info(
        "Preparing Ridge model for +%dh...",
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
    ) = chronological_split(
        X,
        y,
    )

    logger.info(
        "Horizon +%dh | train=%d test=%d features=%d",
        horizon,
        len(X_train),
        len(X_test),
        X_train.shape[1],
    )

    model = RidgeAQI()

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
        "Ridge +%dh | MAE=%.3f RMSE=%.3f R2=%.3f",
        horizon,
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )

    return metrics


# ============================================================
# Run all horizons
# ============================================================

def run_ridge(
    df: pd.DataFrame,
) -> dict:
    """
    Train and evaluate Ridge Regression
    for 24h, 48h and 72h forecasting.
    """

    df = df.copy()

    if df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    logger.info(
        "Starting Ridge Regression evaluation..."
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
        ] = run_ridge_for_horizon(
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
        "AQI RIDGE REGRESSION"
    )

    logger.info(
        "=" * 60
    )

    # Load dataset
    df = pd.read_parquet(
        DATA_PATH
    )

    logger.info(
        "Loaded %d rows and %d columns.",
        len(df),
        len(df.columns),
    )

    # Run model
    results = run_ridge(
        df
    )

    # Display results
    print(
        "\n" + "=" * 50
    )

    print(
        "RIDGE REGRESSION RESULTS"
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
        "\nCompare these results against the Naive Baseline."
    )