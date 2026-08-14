"""
Final unified AQI prediction pipeline.

Production model:
    Tuned Random Forest

Forecast horizons:
    Day 1 -> +24h
    Day 2 -> +48h
    Day 3 -> +72h

Evaluation metrics:
    MAE
    RMSE
    R2

The primary task is AQI regression forecasting.
No classification metrics are used.
"""

import os

import joblib
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

MODEL_DIR = "models/saved"

TEST_SIZE = 0.20

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
# Best tuned Random Forest parameters
# ============================================================

BEST_PARAMS = {
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
# Load data
# ============================================================

def load_data():
    """
    Load the clean model-ready training dataset.
    """

    logger.info(
        "Loading clean training dataset..."
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    required = [
        "timestamp",
        "city",
        *FEATURE_COLUMNS,
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=required
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
# Regression metrics
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):
    """
    Calculate regression evaluation metrics.
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
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# Chronological split
# ============================================================

def chronological_split(df):
    """
    Split data chronologically:

        80% train
        20% test
    """

    timestamps = (
        df["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(
        len(timestamps)
        * (1 - TEST_SIZE)
    )

    test_start = timestamps.iloc[
        split_index
    ]

    train_df = df[
        df["timestamp"] < test_start
    ].copy()

    test_df = df[
        df["timestamp"] >= test_start
    ].copy()

    logger.info(
        "Chronological split: train=%d, test=%d",
        len(train_df),
        len(test_df),
    )

    return (
        train_df,
        test_df,
    )


# ============================================================
# Model creation
# ============================================================

def create_model(
    horizon,
):
    """
    Create tuned Random Forest model for a horizon.
    """

    return RandomForestRegressor(
        **BEST_PARAMS[horizon],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Evaluate one horizon
# ============================================================

def evaluate_horizon(
    train_df,
    test_df,
    horizon,
):
    """
    Evaluate one horizon using the chronological test set.
    """

    target_column = (
        f"target_aqi_{horizon}h"
    )

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        target_column
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        target_column
    ]

    model = create_model(
        horizon
    )

    logger.info(
        "Training evaluation model for +%dh...",
        horizon,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = regression_metrics(
        y_test,
        predictions,
    )

    return (
        metrics,
        predictions,
    )


# ============================================================
# Print evaluation
# ============================================================

def print_evaluation(
    horizon,
    metrics,
):
    """
    Print regression-only evaluation.
    """

    day = horizon // 24

    print()
    print("=" * 70)
    print(
        f"DAY {day} (+{horizon}H) FINAL TEST EVALUATION"
    )
    print("=" * 70)

    print(
        f"MAE : {metrics['MAE']:.3f}"
    )

    print(
        f"RMSE: {metrics['RMSE']:.3f}"
    )

    print(
        f"R2  : {metrics['R2']:.3f}"
    )


# ============================================================
# Train production models
# ============================================================

def train_production_models(
    df,
):
    """
    Train one final Random Forest per horizon
    using all clean training data.
    """

    production_models = {}

    logger.info(
        "Training production models on all clean data..."
    )

    X = df[
        FEATURE_COLUMNS
    ]

    for horizon in [
        24,
        48,
        72,
    ]:

        target_column = (
            f"target_aqi_{horizon}h"
        )

        y = df[
            target_column
        ]

        model = create_model(
            horizon
        )

        logger.info(
            "Training production Day %d model...",
            horizon // 24,
        )

        model.fit(
            X,
            y,
        )

        production_models[
            horizon
        ] = model

    return production_models


# ============================================================
# Save production models
# ============================================================

def save_production_models(
    models,
):
    """
    Save production Random Forest models.
    """

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    for horizon, model in models.items():

        path = os.path.join(
            MODEL_DIR,
            f"final_random_forest_{horizon}h.joblib",
        )

        joblib.dump(
            model,
            path,
        )

        logger.info(
            "Saved production model: %s",
            path,
        )


# ============================================================
# Unified Day 1 / Day 2 / Day 3 forecast
# ============================================================

def predict_day_1_2_3(
    df,
    production_models,
):
    """
    Generate the final Day-1 / Day-2 / Day-3 forecast
    using the latest model-ready observation.
    """

    latest_row = (
        df.sort_values(
            "timestamp"
        )
        .iloc[-1]
    )

    X_latest = pd.DataFrame(
        [
            latest_row[
                FEATURE_COLUMNS
            ]
        ]
    )

    current_city = latest_row[
        "city"
    ]

    current_timestamp = latest_row[
        "timestamp"
    ]

    current_aqi = float(
        latest_row["aqi"]
    )

    forecasts = {}

    for horizon in [
        24,
        48,
        72,
    ]:

        model = production_models[
            horizon
        ]

        prediction = float(
            model.predict(
                X_latest
            )[0]
        )

        # AQI cannot be negative.
        prediction = max(
            prediction,
            0.0,
        )

        forecasts[
            horizon
        ] = prediction

    print()
    print("=" * 70)
    print(
        "UNIFIED AQI FORECAST"
    )
    print("=" * 70)

    print(
        f"City             : {current_city}"
    )

    print(
        f"Latest timestamp : {current_timestamp}"
    )

    print(
        f"Current AQI      : {current_aqi:.1f}"
    )

    print()
    print("Forecast:")

    for horizon in [
        24,
        48,
        72,
    ]:

        day = horizon // 24

        prediction = forecasts[
            horizon
        ]

        print(
            f"Day {day} (+{horizon}h): "
            f"{prediction:.1f} AQI"
        )

    print("=" * 70)

    return {
        "city": current_city,
        "timestamp": current_timestamp,
        "current_aqi": current_aqi,
        "day_1": forecasts[24],
        "day_2": forecasts[48],
        "day_3": forecasts[72],
    }


# ============================================================
# Main
# ============================================================

def main():
    """
    Complete final Random Forest prediction pipeline.
    """

    logger.info(
        "Starting final AQI regression pipeline..."
    )

    df = load_data()

    # --------------------------------------------------------
    # 1. Evaluate on untouched chronological test set
    # --------------------------------------------------------

    train_df, test_df = chronological_split(
        df
    )

    final_evaluation = {}

    for horizon in [
        24,
        48,
        72,
    ]:

        (
            metrics,
            _,
        ) = evaluate_horizon(
            train_df,
            test_df,
            horizon,
        )

        final_evaluation[
            horizon
        ] = metrics

        print_evaluation(
            horizon,
            metrics,
        )

    # --------------------------------------------------------
    # 2. Train production models on ALL clean data
    # --------------------------------------------------------

    production_models = (
        train_production_models(
            df
        )
    )

    # --------------------------------------------------------
    # 3. Save production models
    # --------------------------------------------------------

    save_production_models(
        production_models
    )

    # --------------------------------------------------------
    # 4. Generate unified Day 1 / Day 2 / Day 3 forecast
    # --------------------------------------------------------

    forecast = predict_day_1_2_3(
        df,
        production_models,
    )

    # --------------------------------------------------------
    # 5. Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL REGRESSION SUMMARY"
    )
    print("=" * 70)

    for horizon in [
        24,
        48,
        72,
    ]:

        day = horizon // 24

        metrics = final_evaluation[
            horizon
        ]

        print(
            f"\nDay {day} (+{horizon}h)"
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

    print()
    print(
        "Final AQI regression pipeline completed."
    )


if __name__ == "__main__":
    main()