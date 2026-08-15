"""
City-aware ensemble for AQI forecasting.

Models:
    1. Ridge Regression
    2. Tuned Random Forest
    3. Tuned XGBoost

Forecast horizons:
    Day 1 -> +24h
    Day 2 -> +48h
    Day 3 -> +72h

Method:
    - 70% chronological training
    - 10% chronological validation
    - 20% chronological test
    - Ensemble weights are learned ONLY on validation data.
    - Final ensemble is evaluated ONLY on the untouched test set.

Primary metrics:
    MAE
    RMSE
    R2
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/training_data.parquet"

RANDOM_STATE = 42


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
# Tuned Random Forest parameters
# ============================================================

RF_PARAMS = {
    24: {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    48: {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    72: {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
}


# ============================================================
# Tuned XGBoost parameters
# ============================================================

XGB_PARAMS = {
    24: {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
    },
    48: {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    },
    72: {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    },
}


# ============================================================
# Load data
# ============================================================

def load_data() -> pd.DataFrame:
    """Load and validate the clean training dataset."""

    logger.info("Loading training dataset...")

    df = pd.read_parquet(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    required_columns = [
        "timestamp",
        "city",
        *FEATURE_COLUMNS,
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df = df.sort_values(
        ["timestamp", "city"]
    ).reset_index(drop=True)

    logger.info(
        "Loaded %d clean rows.",
        len(df),
    )

    return df


# ============================================================
# Feature preparation
# ============================================================

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model features including one-hot encoded city.

    City is added because previous error analysis showed
    materially different performance between Lahore and
    Islamabad.
    """

    X = df[
        FEATURE_COLUMNS + ["city"]
    ].copy()

    X = pd.get_dummies(
        X,
        columns=["city"],
        prefix="city",
        dtype=float,
    )

    return X


# ============================================================
# Regression metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:
    """Calculate MAE, RMSE and R2."""

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
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# Chronological split
# ============================================================

def split_data(df: pd.DataFrame):
    """
    Chronological split:

        70% train
        10% validation
        20% test
    """

    unique_timestamps = (
        df["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n = len(unique_timestamps)

    train_end = unique_timestamps.iloc[
        int(n * 0.70)
    ]

    validation_end = unique_timestamps.iloc[
        int(n * 0.80)
    ]

    train_df = df[
        df["timestamp"] < train_end
    ].copy()

    validation_df = df[
        (df["timestamp"] >= train_end)
        & (df["timestamp"] < validation_end)
    ].copy()

    test_df = df[
        df["timestamp"] >= validation_end
    ].copy()

    logger.info(
        "Train rows: %d",
        len(train_df),
    )

    logger.info(
        "Validation rows: %d",
        len(validation_df),
    )

    logger.info(
        "Test rows: %d",
        len(test_df),
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# Models
# ============================================================

def build_ridge() -> Ridge:
    """Create Ridge regression."""

    return Ridge(
        alpha=1.0
    )


def build_random_forest(
    horizon: int,
) -> RandomForestRegressor:
    """Create tuned Random Forest."""

    return RandomForestRegressor(
        **RF_PARAMS[horizon],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgboost(
    horizon: int,
) -> XGBRegressor:
    """Create tuned XGBoost."""

    return XGBRegressor(
        **XGB_PARAMS[horizon],
        objective="reg:squarederror",
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Ensemble weight search
# ============================================================

def find_best_weights(
    y_true,
    ridge_pred,
    rf_pred,
    xgb_pred,
):
    """
    Find the ensemble weights that minimize validation MAE.

    Weight constraints:
        w_ridge >= 0
        w_rf >= 0
        w_xgb >= 0
        w_ridge + w_rf + w_xgb = 1

    Search resolution:
        0.01

    The validation set is used only for selecting weights.
    """

    best_weights = None
    best_mae = float("inf")

    for w_ridge_int in range(0, 101):

        w_ridge = w_ridge_int / 100.0

        for w_rf_int in range(
            0,
            101 - w_ridge_int,
        ):

            w_rf = w_rf_int / 100.0

            w_xgb = round(
                1.0 - w_ridge - w_rf,
                2,
            )

            if w_xgb < 0:
                continue

            ensemble_prediction = (
                w_ridge * ridge_pred
                + w_rf * rf_pred
                + w_xgb * xgb_pred
            )

            mae = mean_absolute_error(
                y_true,
                ensemble_prediction,
            )

            if mae < best_mae:

                best_mae = mae

                best_weights = (
                    w_ridge,
                    w_rf,
                    w_xgb,
                )

    return (
        best_weights,
        best_mae,
    )


# ============================================================
# Run one horizon
# ============================================================

def run_horizon(
    df: pd.DataFrame,
    horizon: int,
):
    """
    Train the ensemble for one horizon.

    Validation:
        learn ensemble weights.

    Test:
        final untouched evaluation.
    """

    target_column = (
        f"target_aqi_{horizon}h"
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = split_data(df)

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train = make_features(
        train_df
    )

    X_validation = make_features(
        validation_df
    )

    X_test = make_features(
        test_df
    )

    # Ensure identical feature columns.
    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    y_train = train_df[
        target_column
    ]

    y_validation = validation_df[
        target_column
    ]

    y_test = test_df[
        target_column
    ]

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    ridge_scaler = StandardScaler()

    X_train_ridge = (
        ridge_scaler.fit_transform(
            X_train
        )
    )

    X_validation_ridge = (
        ridge_scaler.transform(
            X_validation
        )
    )

    X_test_ridge = (
        ridge_scaler.transform(
            X_test
        )
    )

    ridge = build_ridge()

    ridge.fit(
        X_train_ridge,
        y_train,
    )

    ridge_validation_pred = (
        ridge.predict(
            X_validation_ridge
        )
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf = build_random_forest(
        horizon
    )

    rf.fit(
        X_train,
        y_train,
    )

    rf_validation_pred = (
        rf.predict(
            X_validation
        )
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb = build_xgboost(
        horizon
    )

    xgb.fit(
        X_train,
        y_train,
        verbose=False,
    )

    xgb_validation_pred = (
        xgb.predict(
            X_validation
        )
    )

    # --------------------------------------------------------
    # Validation weights
    # --------------------------------------------------------

    (
        weights,
        validation_mae,
    ) = find_best_weights(
        y_validation.to_numpy(),
        ridge_validation_pred,
        rf_validation_pred,
        xgb_validation_pred,
    )

    (
        w_ridge,
        w_rf,
        w_xgb,
    ) = weights

    logger.info(
        "Day %d validation weights: "
        "Ridge=%.2f RF=%.2f XGB=%.2f",
        horizon // 24,
        w_ridge,
        w_rf,
        w_xgb,
    )

    logger.info(
        "Day %d validation ensemble MAE: %.3f",
        horizon // 24,
        validation_mae,
    )

    # --------------------------------------------------------
    # Retrain on train + validation
    # --------------------------------------------------------

    full_train_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    X_full = make_features(
        full_train_df
    )

    X_full_test = X_test.reindex(
        columns=X_full.columns,
        fill_value=0,
    )

    y_full = full_train_df[
        target_column
    ]

    # --------------------------------------------------------
    # Final Ridge
    # --------------------------------------------------------

    final_ridge_scaler = StandardScaler()

    X_full_ridge = (
        final_ridge_scaler.fit_transform(
            X_full
        )
    )

    X_full_test_ridge = (
        final_ridge_scaler.transform(
            X_full_test
        )
    )

    final_ridge = build_ridge()

    final_ridge.fit(
        X_full_ridge,
        y_full,
    )

    ridge_test_pred = (
        final_ridge.predict(
            X_full_test_ridge
        )
    )

    # --------------------------------------------------------
    # Final Random Forest
    # --------------------------------------------------------

    final_rf = build_random_forest(
        horizon
    )

    final_rf.fit(
        X_full,
        y_full,
    )

    rf_test_pred = (
        final_rf.predict(
            X_full_test
        )
    )

    # --------------------------------------------------------
    # Final XGBoost
    # --------------------------------------------------------

    final_xgb = build_xgboost(
        horizon
    )

    final_xgb.fit(
        X_full,
        y_full,
        verbose=False,
    )

    xgb_test_pred = (
        final_xgb.predict(
            X_full_test
        )
    )

    # --------------------------------------------------------
    # Final ensemble
    # --------------------------------------------------------

    ensemble_test_pred = (
        w_ridge * ridge_test_pred
        + w_rf * rf_test_pred
        + w_xgb * xgb_test_pred
    )

    final_metrics = calculate_metrics(
        y_test,
        ensemble_test_pred,
    )

    return {
        "weights": weights,
        "validation_mae": validation_mae,
        "metrics": final_metrics,
    }


# ============================================================
# Main
# ============================================================

def main():
    """Run the complete city-aware ensemble experiment."""

    logger.info(
        "Starting city-aware ensemble experiment..."
    )

    df = load_data()

    print()
    print("=" * 75)
    print(
        "CITY-AWARE ENSEMBLE RESULTS"
    )
    print("=" * 75)

    results = {}

    for horizon in [
        24,
        48,
        72,
    ]:

        result = run_horizon(
            df,
            horizon,
        )

        results[
            horizon
        ] = result

        (
            w_ridge,
            w_rf,
            w_xgb,
        ) = result["weights"]

        metrics = result[
            "metrics"
        ]

        print()
        print(
            f"Day {horizon // 24} (+{horizon}h)"
        )

        print(
            "Validation weights:"
        )

        print(
            f"  Ridge : {w_ridge:.2f}"
        )

        print(
            f"  RF    : {w_rf:.2f}"
        )

        print(
            f"  XGB   : {w_xgb:.2f}"
        )

        print()
        print(
            "Final test:"
        )

        print(
            f"  MAE : {metrics['MAE']:.3f}"
        )

        print(
            f"  RMSE: {metrics['RMSE']:.3f}"
        )

        print(
            f"  R2  : {metrics['R2']:.3f}"
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "ENSEMBLE FINAL SUMMARY"
    )
    print("=" * 75)

    for horizon in [
        24,
        48,
        72,
    ]:

        metrics = results[
            horizon
        ]["metrics"]

        print(
            f"Day {horizon // 24}: "
            f"MAE={metrics['MAE']:.3f}  "
            f"RMSE={metrics['RMSE']:.3f}  "
            f"R2={metrics['R2']:.3f}"
        )

    print()
    print(
        "Ensemble experiment completed."
    )


if __name__ == "__main__":
    main()