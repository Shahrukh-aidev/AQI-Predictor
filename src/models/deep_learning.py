"""
Deep Learning models for the Pearls AQI Predictor.

This module trains three separate LSTM regression models:

    Day 1 -> +24h AQI
    Day 2 -> +48h AQI
    Day 3 -> +72h AQI

Evaluation:
    MAE
    RMSE
    R2

Data split:
    70% chronological training
    10% chronological validation
    20% chronological test

Important:
    - No random shuffling.
    - Scaling is fitted on training data only.
    - Sequences are created separately for each city.
    - Sequences crossing large time gaps are not allowed.
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/training_data.parquet"

MODEL_DIR = "models/saved"

SEQUENCE_LENGTH = 24

MAX_GAP_HOURS = 3

EPOCHS = 50

BATCH_SIZE = 32

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
# Reproducibility
# ============================================================

def set_seed(seed: int = RANDOM_STATE) -> None:
    """Make training as reproducible as practical."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Metrics
# ============================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate regression metrics."""

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
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 3),
    }


# ============================================================
# Load data
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the cleaned model-ready training data."""

    logger.info(
        "Loading training dataset..."
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    required = [
        "timestamp",
        "city",
        *FEATURE_COLUMNS,
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    logger.info(
        "Loaded %d clean rows.",
        len(df),
    )

    return df


# ============================================================
# Chronological split boundaries
# ============================================================

def get_split_boundaries(
    df: pd.DataFrame,
):
    """
    Calculate global chronological train/validation/test
    boundaries using timestamps.

    70% train
    10% validation
    20% test
    """

    unique_times = (
        df["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n = len(unique_times)

    train_index = int(
        n * 0.70
    )

    validation_index = int(
        n * 0.80
    )

    train_end = unique_times.iloc[
        train_index
    ]

    validation_end = unique_times.iloc[
        validation_index
    ]

    return (
        train_end,
        validation_end,
    )


# ============================================================
# Scale features
# ============================================================

def fit_scaler(
    train_df: pd.DataFrame,
) -> StandardScaler:
    """
    Fit scaler using training data only.
    """

    scaler = StandardScaler()

    scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    return scaler


# ============================================================
# Build model
# ============================================================

def build_lstm_model(
    sequence_length: int,
    n_features: int,
) -> Sequential:
    """Create an LSTM regression model."""

    model = Sequential(
        [
            LSTM(
                64,
                input_shape=(
                    sequence_length,
                    n_features,
                ),
                return_sequences=False,
            ),

            Dropout(
                0.20
            ),

            Dense(
                32,
                activation="relu",
            ),

            Dense(
                1
            ),
        ]
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ============================================================
# Sequence creation
# ============================================================

def create_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    target_column: str,
):
    """
    Create 24-hour sequences separately for each city.

    A sequence is valid only when:
        - It belongs to the same city.
        - It contains 24 rows.
        - Consecutive timestamps are no more than 3 hours apart.

    Returns:
        X
        y
        target timestamps
        target cities
    """

    sequences = []
    targets = []
    target_times = []
    target_cities = []

    for city, city_df in df.groupby(
        "city",
        sort=True,
    ):
        city_df = city_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        scaled_features = scaler.transform(
            city_df[FEATURE_COLUMNS]
        )

        target_values = city_df[
            target_column
        ].to_numpy(
            dtype=np.float32
        )

        timestamps = (
            city_df["timestamp"]
            .to_numpy()
        )

        for i in range(
            SEQUENCE_LENGTH,
            len(city_df),
        ):
            window_start = i - SEQUENCE_LENGTH
            window_end = i

            window_times = timestamps[
                window_start:window_end
            ]

            # Check chronological continuity.
            time_differences = (
                pd.Series(
                    window_times
                )
                .diff()
                .dropna()
                .dt.total_seconds()
                / 3600.0
            )

            if (
                not time_differences.empty
                and time_differences.max()
                > MAX_GAP_HOURS
            ):
                continue

            # Ensure the last observed timestamp
            # is actually before the prediction target.
            target_time = pd.Timestamp(
                timestamps[i]
            )

            last_input_time = pd.Timestamp(
                timestamps[i - 1]
            )

            if target_time <= last_input_time:
                continue

            sequences.append(
                scaled_features[
                    window_start:window_end
                ]
            )

            targets.append(
                target_values[i]
            )

            target_times.append(
                target_time
            )

            target_cities.append(
                city
            )

    if not sequences:
        raise ValueError(
            "No valid LSTM sequences were created."
        )

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    )

    target_times = pd.to_datetime(
        target_times,
        utc=True,
    )

    target_cities = np.asarray(
        target_cities
    )

    return (
        X,
        y,
        target_times,
        target_cities,
    )


# ============================================================
# Split sequences by target timestamp
# ============================================================

def split_sequences(
    X,
    y,
    target_times,
    target_cities,
    train_end,
    validation_end,
):
    """Split sequences according to prediction timestamp."""

    train_mask = (
        target_times < train_end
    )

    validation_mask = (
        (target_times >= train_end)
        & (target_times < validation_end)
    )

    test_mask = (
        target_times >= validation_end
    )

    return (
        X[train_mask],
        y[train_mask],
        target_times[train_mask],
        target_cities[train_mask],

        X[validation_mask],
        y[validation_mask],
        target_times[validation_mask],
        target_cities[validation_mask],

        X[test_mask],
        y[test_mask],
        target_times[test_mask],
        target_cities[test_mask],
    )


# ============================================================
# Train one horizon
# ============================================================

def train_horizon(
    df: pd.DataFrame,
    horizon: int,
):
    """Train one LSTM for one forecast horizon."""

    target_column = (
        f"target_aqi_{horizon}h"
    )

    logger.info("=" * 70)

    logger.info(
        "TRAINING LSTM — DAY %d (+%dh)",
        horizon // 24,
        horizon,
    )

    logger.info("=" * 70)

    train_end, validation_end = (
        get_split_boundaries(df)
    )

    train_rows = df[
        df["timestamp"] < train_end
    ]

    scaler = fit_scaler(
        train_rows
    )

    (
        X,
        y,
        target_times,
        target_cities,
    ) = create_sequences(
        df,
        scaler,
        target_column,
    )

    (
        X_train,
        y_train,
        _,
        _,

        X_validation,
        y_validation,
        _,
        _,

        X_test,
        y_test,
        test_times,
        test_cities,
    ) = split_sequences(
        X,
        y,
        target_times,
        target_cities,
        train_end,
        validation_end,
    )

    logger.info(
        "Day %d sequence shapes:",
        horizon // 24,
    )

    logger.info(
        "Train: %s",
        X_train.shape,
    )

    logger.info(
        "Validation: %s",
        X_validation.shape,
    )

    logger.info(
        "Test: %s",
        X_test.shape,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_lstm_model(
        SEQUENCE_LENGTH,
        len(FEATURE_COLUMNS),
    )

    model.summary()

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"lstm_{horizon}h.keras",
    )

    prediction_path = os.path.join(
        MODEL_DIR,
        f"lstm_{horizon}h_predictions.csv",
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    logger.info(
        "Training Day %d model...",
        horizon // 24,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    predictions = model.predict(
        X_test,
        verbose=0,
    ).reshape(-1)

    metrics = evaluate_predictions(
        y_test,
        predictions,
    )

    logger.info(
        "Day %d | MAE=%.3f | RMSE=%.3f | R2=%.3f",
        horizon // 24,
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "timestamp": test_times,
            "city": test_cities,
            "actual_aqi": y_test,
            "predicted_aqi": predictions,
        }
    )

    prediction_df["error"] = (
        prediction_df["actual_aqi"]
        - prediction_df["predicted_aqi"]
    )

    prediction_df["absolute_error"] = (
        prediction_df["error"].abs()
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    logger.info(
        "Saved model: %s",
        model_path,
    )

    logger.info(
        "Saved predictions: %s",
        prediction_path,
    )

    return {
        "metrics": metrics,
        "model_path": model_path,
        "prediction_path": prediction_path,
    }


# ============================================================
# Main
# ============================================================

def main():
    """Train Day 1, Day 2 and Day 3 LSTM models."""

    set_seed()

    df = load_data()

    results = {}

    for horizon in [
        24,
        48,
        72,
    ]:
        results[
            horizon
        ] = train_horizon(
            df,
            horizon,
        )

    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LSTM AQI FORECAST RESULTS"
    )

    print(
        "=" * 70
    )

    for horizon in [
        24,
        48,
        72,
    ]:

        day = horizon // 24

        metrics = results[
            horizon
        ]["metrics"]

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

    print(
        "\n" + "=" * 70
    )

    print(
        "LSTM TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()