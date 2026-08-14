"""
Random Forest hyperparameter tuning for AQI forecasting.

Uses:
    70% chronological train
    10% chronological validation
    20% chronological test

The test set is kept untouched during tuning.
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


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def prepare_data(df, target_column):
    columns = [
        "timestamp",
        target_column,
        *FEATURE_COLUMNS,
    ]

    data = df[columns].copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data = data.dropna(
        subset=columns
    )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return (
        data[FEATURE_COLUMNS],
        data[target_column],
    )


def split_time_series(X, y):
    """
    70% train
    10% validation
    20% test
    """

    n = len(X)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.80)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[
        train_end:validation_end
    ]
    y_validation = y.iloc[
        train_end:validation_end
    ]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def tune_horizon(df, horizon):
    target = f"target_aqi_{horizon}h"

    logger.info(
        "Tuning Random Forest for +%dh",
        horizon,
    )

    X, y = prepare_data(
        df,
        target,
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_time_series(
        X,
        y,
    )

    parameter_grid = [
        {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 500,
            "max_depth": 15,
            "min_samples_leaf": 1,
            "max_features": 0.8,
        },
    ]

    best_params = None
    best_metrics = None

    for i, params in enumerate(
        parameter_grid,
        start=1,
    ):
        logger.info(
            "Configuration %d/%d: %s",
            i,
            len(parameter_grid),
            params,
        )

        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = evaluate(
            y_validation,
            predictions,
        )

        logger.info(
            "Validation +%dh | MAE=%.3f RMSE=%.3f R2=%.3f",
            horizon,
            metrics["MAE"],
            metrics["RMSE"],
            metrics["R2"],
        )

        if (
            best_metrics is None
            or metrics["MAE"] < best_metrics["MAE"]
        ):
            best_metrics = metrics
            best_params = params

    logger.info(
        "BEST PARAMETERS +%dh: %s",
        horizon,
        best_params,
    )

    # Retrain best model using train + validation
    X_train_full = pd.concat(
        [X_train, X_validation]
    )

    y_train_full = pd.concat(
        [y_train, y_validation]
    )

    final_model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_train_full,
        y_train_full,
    )

    # Evaluate ONLY once on untouched test set
    test_predictions = final_model.predict(
        X_test
    )

    test_metrics = evaluate(
        y_test,
        test_predictions,
    )

    logger.info(
        "FINAL TEST +%dh | MAE=%.3f RMSE=%.3f R2=%.3f",
        horizon,
        test_metrics["MAE"],
        test_metrics["RMSE"],
        test_metrics["R2"],
    )

    return {
        "best_params": best_params,
        "validation": best_metrics,
        "test": test_metrics,
    }


def main():
    logger.info(
        "Loading training data..."
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    results = {}

    for horizon in [24, 48, 72]:
        results[f"{horizon}h"] = tune_horizon(
            df,
            horizon,
        )

    print("\n" + "=" * 70)
    print("RANDOM FOREST TUNING RESULTS")
    print("=" * 70)

    for horizon, result in results.items():
        print(f"\nHorizon +{horizon}")

        print(
            "Best parameters:"
        )

        for key, value in result[
            "best_params"
        ].items():
            print(
                f"  {key}: {value}"
            )

        validation = result[
            "validation"
        ]

        test = result[
            "test"
        ]

        print(
            "\nBest validation:"
        )

        print(
            f"  MAE : {validation['MAE']:.3f}"
        )

        print(
            f"  RMSE: {validation['RMSE']:.3f}"
        )

        print(
            f"  R2  : {validation['R2']:.3f}"
        )

        print(
            "\nFinal test:"
        )

        print(
            f"  MAE : {test['MAE']:.3f}"
        )

        print(
            f"  RMSE: {test['RMSE']:.3f}"
        )

        print(
            f"  R2  : {test['R2']:.3f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()