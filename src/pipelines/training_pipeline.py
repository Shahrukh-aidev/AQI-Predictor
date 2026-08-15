"""
Daily AQI Training + Automatic Model Promotion Pipeline.

Purpose
-------
Runs daily from GitHub Actions and:

1. Reads labeled historical data from Hopsworks Feature Group v6.
2. Removes live rows whose future targets are still NULL.
3. Performs a chronological 80/20 train/test split.
4. Trains candidate tuned Random Forest models for:
       +24h
       +48h
       +72h
5. Reads the current production model MAE directly from
   Hopsworks Model Registry metadata.
6. Does NOT download the production model just to compare MAE.
7. Promotes the candidate only when candidate MAE is lower.
8. When promoted:
       - retrains on ALL labeled data
       - saves the production .joblib locally
       - registers a new Model Registry version
9. Saves a training metrics report.

Production model names
-----------------------
    aqi_random_forest_24h
    aqi_random_forest_48h
    aqi_random_forest_72h

Feature Group
-------------
    aqi_features
    version 6

Important
---------
The hourly feature pipeline inserts recent live rows into the
same Feature Group, but those rows do not yet have future AQI
targets.

Only rows where all three target columns are available are
used for training.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
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

FEATURE_GROUP_NAME = "aqi_features"

FEATURE_GROUP_VERSION = int(
    os.getenv(
        "HOPSWORKS_FEATURE_GROUP_VERSION",
        "6",
    )
)

MODEL_DIR = Path(
    "models/saved"
)

METRICS_PATH = Path(
    "models/training_metrics.json"
)

TEST_SIZE = 0.20

RANDOM_STATE = 42

HORIZONS = [
    24,
    48,
    72,
]

MODEL_NAME_TEMPLATE = (
    "aqi_random_forest_{horizon}h"
)


# ============================================================
# Tuned Random Forest parameters
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
# Exact model features
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


TARGET_COLUMNS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


# ============================================================
# Data validation
# ============================================================

def validate_training_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and clean the labeled training dataset.

    Only rows with:
        - valid timestamp
        - valid features
        - all three future targets
    are kept.
    """

    required_columns = [
        "timestamp",
        "city",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training data is missing required columns: "
            f"{missing_columns}"
        )

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove incomplete rows
    #
    # This excludes new live rows because their target columns
    # are NULL until the future observations actually exist.
    # --------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    ).copy()

    # --------------------------------------------------------
    # Remove duplicate city/timestamp rows
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=[
                "city",
                "timestamp",
            ],
            keep="last",
        )
        .sort_values(
            [
                "timestamp",
                "city",
            ]
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            "No labeled training rows remain after validation."
        )

    logger.info(
        "Validated labeled training dataset: %d rows.",
        len(df),
    )

    return df


# ============================================================
# Load training data from Hopsworks
# ============================================================

def load_training_data_from_hopsworks(
    project,
) -> pd.DataFrame:
    """
    Read the Feature Group from Hopsworks and return only
    labeled historical training rows.
    """

    logger.info(
        "Opening Feature Group '%s' version %d.",
        FEATURE_GROUP_NAME,
        FEATURE_GROUP_VERSION,
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    logger.info(
        "Reading Feature Group into pandas..."
    )

    df = fg.read(
        dataframe_type="pandas"
    )

    if df is None:
        raise RuntimeError(
            "Hopsworks Feature Group returned no dataframe."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        df = pd.DataFrame(df)

    logger.info(
        "Hopsworks returned %d rows and %d columns.",
        len(df),
        len(df.columns),
    )

    return validate_training_data(
        df
    )


# ============================================================
# Chronological split
# ============================================================

def chronological_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Perform an 80/20 chronological split.

    The split is performed on timestamps so observations from
    the same timestamp stay together.
    """

    timestamps = (
        df["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(timestamps) < 10:
        raise ValueError(
            "Not enough unique timestamps for "
            "chronological evaluation."
        )

    split_index = int(
        len(timestamps)
        * (1 - TEST_SIZE)
    )

    split_index = max(
        1,
        min(
            split_index,
            len(timestamps) - 1,
        ),
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

    if train_df.empty:
        raise RuntimeError(
            "Training split is empty."
        )

    if test_df.empty:
        raise RuntimeError(
            "Test split is empty."
        )

    logger.info(
        "Chronological split | train=%d | test=%d | "
        "test_start=%s",
        len(train_df),
        len(test_df),
        test_start,
    )

    return (
        train_df,
        test_df,
    )


# ============================================================
# Model creation
# ============================================================

def create_model(
    horizon: int,
) -> RandomForestRegressor:
    """
    Create tuned Random Forest model for one horizon.
    """

    if horizon not in BEST_PARAMS:
        raise ValueError(
            f"Unsupported forecast horizon: {horizon}"
        )

    return RandomForestRegressor(
        **BEST_PARAMS[horizon],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate regression metrics.
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
# Candidate training + evaluation
# ============================================================

def train_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
) -> tuple[
    RandomForestRegressor,
    dict[str, float],
]:
    """
    Train and evaluate one candidate Random Forest model.
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
        "Training candidate +%dh | train=%d | test=%d",
        horizon,
        len(X_train),
        len(X_test),
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = compute_metrics(
        y_test.to_numpy(),
        predictions,
    )

    logger.info(
        "Candidate +%dh | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        horizon,
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )

    return (
        model,
        metrics,
    )


# ============================================================
# Registry lookup
# ============================================================

def get_latest_registered_model(
    mr,
    model_name: str,
):
    """
    Return the latest registered version of a model.

    No model artifact is downloaded.
    """

    try:

        models = mr.get_models(
            name=model_name
        )

    except Exception as exc:

        logger.warning(
            "Unable to read registry versions for %s: %s",
            model_name,
            exc,
        )

        return None

    if not models:

        logger.info(
            "No registered model exists for %s.",
            model_name,
        )

        return None

    models = sorted(
        models,
        key=lambda model: int(
            model.version
        ),
    )

    latest = models[-1]

    logger.info(
        "Latest registered model: %s v%s",
        latest.name,
        latest.version,
    )

    return latest


# ============================================================
# Read production MAE from registry metadata
# ============================================================

def get_production_mae(
    mr,
    model_name: str,
) -> tuple[
    float | None,
    int | None,
]:
    """
    Read the latest production model MAE directly from
    Model Registry metadata.

    IMPORTANT:
    This function does NOT download the model file.
    """

    latest = get_latest_registered_model(
        mr,
        model_name,
    )

    if latest is None:
        return (
            None,
            None,
        )

    try:

        metrics = (
            latest.training_metrics
            or {}
        )

        mae = (
            metrics.get("MAE")
            or metrics.get("mae")
        )

        if mae is None:

            logger.warning(
                "Production model %s v%s has no MAE metadata.",
                model_name,
                latest.version,
            )

            return (
                None,
                int(
                    latest.version
                ),
            )

        mae = float(
            mae
        )

        logger.info(
            "Production %s v%s | stored MAE=%.4f",
            model_name,
            latest.version,
            mae,
        )

        return (
            mae,
            int(
                latest.version
            ),
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        logger.warning(
            "Invalid MAE metadata for %s v%s: %s",
            model_name,
            latest.version,
            exc,
        )

        return (
            None,
            int(
                latest.version
            ),
        )

    except Exception as exc:

        logger.warning(
            "Could not read production metadata for %s: %s",
            model_name,
            exc,
        )

        return (
            None,
            int(
                latest.version
            ),
        )


# ============================================================
# Promotion decision
# ============================================================

def should_promote(
    candidate_mae: float,
    production_mae: float | None,
    horizon: int,
) -> bool:
    """
    Decide whether candidate should become the new production
    model.

    Rules:

    1. No existing model -> promote.
    2. Existing model with MAE -> promote only when the new
       candidate has strictly lower MAE.
    3. Existing model with missing MAE -> DO NOT automatically
       replace production. This is the safe behavior.
    """

    if production_mae is None:

        logger.warning(
            "+%dh: Production MAE is unavailable.",
            horizon,
        )

        logger.warning(
            "+%dh: Candidate will NOT be automatically "
            "promoted without a production baseline.",
            horizon,
        )

        return False

    if candidate_mae < production_mae:

        improvement = (
            production_mae
            - candidate_mae
        )

        logger.info(
            "+%dh: PROMOTE | candidate MAE=%.4f | "
            "production MAE=%.4f | improvement=%.4f",
            horizon,
            candidate_mae,
            production_mae,
            improvement,
        )

        return True

    logger.info(
        "+%dh: KEEP PRODUCTION | candidate MAE=%.4f | "
        "production MAE=%.4f",
        horizon,
        candidate_mae,
        production_mae,
    )

    return False


# ============================================================
# Train final production model
# ============================================================

def train_final_production_model(
    df: pd.DataFrame,
    horizon: int,
) -> RandomForestRegressor:
    """
    Train the promoted production model using ALL labeled data.
    """

    target_column = (
        f"target_aqi_{horizon}h"
    )

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        target_column
    ]

    model = create_model(
        horizon
    )

    logger.info(
        "Training FINAL production +%dh model on %d rows...",
        horizon,
        len(df),
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# Save local production artifact
# ============================================================

def save_production_model(
    model: RandomForestRegressor,
    horizon: int,
) -> Path:
    """
    Save promoted model locally.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        MODEL_DIR
        / f"final_random_forest_{horizon}h.joblib"
    )

    joblib.dump(
        model,
        path,
    )

    logger.info(
        "Saved promoted production model: %s",
        path,
    )

    return path


# ============================================================
# Register promoted model
# ============================================================

def register_model(
    mr,
    model: RandomForestRegressor,
    model_name: str,
    horizon: int,
    metrics: dict[str, float],
    local_path: Path,
) -> int:
    """
    Register the promoted production model.

    Hopsworks automatically creates the next version.
    """

    input_example = pd.DataFrame(
        [
            {
                feature: 0.0
                for feature in FEATURE_COLUMNS
            }
        ]
    )

    hops_model = mr.sklearn.create_model(
        name=model_name,
        metrics={
            "MAE": round(
                metrics["MAE"],
                6,
            ),
            "RMSE": round(
                metrics["RMSE"],
                6,
            ),
            "R2": round(
                metrics["R2"],
                6,
            ),
        },
        description=(
            "Automatically promoted tuned Random Forest "
            f"AQI model for +{horizon}h forecasting. "
            f"Training run: "
            f"{datetime.now(timezone.utc).isoformat()}"
        ),
        input_example=input_example,
    )

    registered = hops_model.save(
        str(local_path)
    )

    logger.info(
        "REGISTERED %s v%s | MAE=%.4f",
        model_name,
        registered.version,
        metrics["MAE"],
    )

    return int(
        registered.version
    )


# ============================================================
# Metrics report
# ============================================================

def save_metrics_report(
    report: dict,
) -> None:
    """
    Save the daily training report.
    """

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    logger.info(
        "Training metrics saved to %s",
        METRICS_PATH,
    )


# ============================================================
# Main training pipeline
# ============================================================

def run_training_pipeline() -> None:
    """
    Execute the complete daily training process.
    """

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING DAILY AQI TRAINING PIPELINE"
    )

    logger.info(
        "=" * 70
    )

    load_dotenv()

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    project_name = os.getenv(
        "HOPSWORKS_PROJECT"
    )

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is not set."
        )

    if not project_name:
        raise ValueError(
            "HOPSWORKS_PROJECT is not set."
        )

    # --------------------------------------------------------
    # Connect to Hopsworks
    # --------------------------------------------------------

    logger.info(
        "Connecting to Hopsworks project: %s",
        project_name,
    )

    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name,
    )

    try:

        mr = project.get_model_registry()

        # ----------------------------------------------------
        # Load labeled data
        # ----------------------------------------------------

        df = load_training_data_from_hopsworks(
            project
        )

        logger.info(
            "Training period: %s -> %s",
            df["timestamp"].min(),
            df["timestamp"].max(),
        )

        # ----------------------------------------------------
        # Chronological split
        # ----------------------------------------------------

        train_df, test_df = chronological_split(
            df
        )

        # ----------------------------------------------------
        # Training report
        # ----------------------------------------------------

        report = {
            "run_timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "feature_group": FEATURE_GROUP_NAME,
            "feature_group_version": (
                FEATURE_GROUP_VERSION
            ),
            "training_rows": int(
                len(df)
            ),
            "train_rows": int(
                len(train_df)
            ),
            "test_rows": int(
                len(test_df)
            ),
            "training_start": (
                df["timestamp"]
                .min()
                .isoformat()
            ),
            "training_end": (
                df["timestamp"]
                .max()
                .isoformat()
            ),
            "horizons": {},
        }

        # ----------------------------------------------------
        # Process horizons
        # ----------------------------------------------------

        for horizon in HORIZONS:

            model_name = (
                MODEL_NAME_TEMPLATE.format(
                    horizon=horizon
                )
            )

            logger.info(
                "-" * 70
            )

            logger.info(
                "PROCESSING +%dh | %s",
                horizon,
                model_name,
            )

            # ------------------------------------------------
            # Train candidate
            # ------------------------------------------------

            (
                candidate_model,
                candidate_metrics,
            ) = train_candidate(
                train_df,
                test_df,
                horizon,
            )

            # ------------------------------------------------
            # Read production MAE ONLY from registry
            #
            # No 200+ MB model download.
            # ------------------------------------------------

            (
                production_mae,
                production_version,
            ) = get_production_mae(
                mr,
                model_name,
            )

            # ------------------------------------------------
            # Promotion decision
            # ------------------------------------------------

            promoted = should_promote(
                candidate_mae=(
                    candidate_metrics["MAE"]
                ),
                production_mae=production_mae,
                horizon=horizon,
            )

            registered_version = None

            # ------------------------------------------------
            # Promote if better
            # ------------------------------------------------

            if promoted:

                final_model = (
                    train_final_production_model(
                        df,
                        horizon,
                    )
                )

                local_path = (
                    save_production_model(
                        final_model,
                        horizon,
                    )
                )

                registered_version = (
                    register_model(
                        mr=mr,
                        model=final_model,
                        model_name=model_name,
                        horizon=horizon,
                        metrics=candidate_metrics,
                        local_path=local_path,
                    )
                )

            # ------------------------------------------------
            # Store report
            # ------------------------------------------------

            report[
                "horizons"
            ][str(horizon)] = {
                "candidate_metrics": (
                    candidate_metrics
                ),
                "production_mae": (
                    production_mae
                ),
                "production_version": (
                    production_version
                ),
                "promoted": promoted,
                "registered_version": (
                    registered_version
                ),
                "model_name": model_name,
            }

        # ----------------------------------------------------
        # Save report
        # ----------------------------------------------------

        save_metrics_report(
            report
        )

        # ----------------------------------------------------
        # Console summary
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print(
            "DAILY AQI TRAINING PIPELINE — SUMMARY"
        )
        print("=" * 80)

        for horizon in HORIZONS:

            result = report[
                "horizons"
            ][str(horizon)]

            candidate_mae = (
                result[
                    "candidate_metrics"
                ]["MAE"]
            )

            production_mae = (
                result[
                    "production_mae"
                ]
            )

            promoted = result[
                "promoted"
            ]

            print()
            print(
                f"+{horizon}h"
            )

            print(
                f"  Candidate MAE : "
                f"{candidate_mae:.4f}"
            )

            if production_mae is None:

                print(
                    "  Production MAE: unavailable"
                )

            else:

                print(
                    f"  Production MAE: "
                    f"{production_mae:.4f}"
                )

            print(
                f"  Status        : "
                f"{'PROMOTED' if promoted else 'KEPT PRODUCTION'}"
            )

            if result[
                "registered_version"
            ] is not None:

                print(
                    f"  Registry v    : "
                    f"{result['registered_version']}"
                )

        print()
        print("=" * 80)

        logger.info(
            "Daily AQI training pipeline completed successfully."
        )

    finally:

        try:
            project.close()
        except Exception:
            pass


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_training_pipeline()