"""
Naive Baseline Model for AQI Predictor.

Strategy
--------
Predict future AQI = current AQI.

This is the simplest possible forecast.
Any ML model that cannot beat this baseline
has a fundamental problem.

Horizons
--------
24h  → predicted AQI = current AQI
48h  → predicted AQI = current AQI
72h  → predicted AQI = current AQI
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import logger


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, R² for a set of predictions.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}


class NaiveBaseline:
    """
    Naive baseline: predict current AQI for all horizons.

    Parameters
    ----------
    None

    Usage
    -----
    model = NaiveBaseline()
    model.fit(X_train, y_train)   # no-op, kept for API consistency
    preds = model.predict(X_test)
    """

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """No training needed — naive model uses current AQI directly."""
        logger.info("NaiveBaseline.fit() called — no training required.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return current AQI as prediction for every horizon.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain 'aqi' column (current AQI).

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if "aqi" not in X.columns:
            raise ValueError("'aqi' column required for NaiveBaseline.")
        return X["aqi"].to_numpy()


def run_baseline(df: pd.DataFrame) -> dict:
    """
    Train/evaluate naive baseline on all three horizons.

    Uses a chronological 80/20 train-test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset with target columns.

    Returns
    -------
    dict
        Results per horizon: {horizon: {MAE, RMSE, R2}}
    """

    # ── Chronological split ────────────────────────────────────────────────────
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split     = int(len(df_sorted) * 0.8)
    test_df   = df_sorted.iloc[split:]

    logger.info(
        "Baseline evaluation | train=%d  test=%d",
        split, len(test_df),
    )

    horizons = [24, 48, 72]
    results  = {}
    model    = NaiveBaseline()

    for h in horizons:
        target_col = f"target_aqi_{h}h"

        if target_col not in test_df.columns:
            logger.warning("Column %s not found — skipping.", target_col)
            continue

        subset = test_df.dropna(subset=[target_col, "aqi"])
        y_true = subset[target_col].to_numpy()
        y_pred = model.predict(subset)

        metrics         = evaluate(y_true, y_pred)
        results[f"{h}h"] = metrics

        logger.info(
            "Naive Baseline +%dh | MAE=%.2f  RMSE=%.2f  R2=%.3f",
            h, metrics["MAE"], metrics["RMSE"], metrics["R2"],
        )

    return results


if __name__ == "__main__":

    import json

    df = pd.read_parquet("data/processed/training_data.parquet")
    results = run_baseline(df)

    print("\n" + "=" * 50)
    print("NAIVE BASELINE RESULTS")
    print("=" * 50)
    for horizon, metrics in results.items():
        print(f"\nHorizon +{horizon}")
        for metric, value in metrics.items():
            print(f"  {metric:6s}: {value}")
    print("=" * 50)
    print("\nThis is the benchmark every ML model must beat.")