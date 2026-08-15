"""
Register production AQI Random Forest models in Hopsworks Model Registry.
"""

from __future__ import annotations

import os
from pathlib import Path

import hopsworks
from dotenv import load_dotenv

from src.utils.logger import logger


MODEL_DIR = Path("models/saved")

MODELS = {
    "aqi_random_forest_24h": (
        MODEL_DIR / "final_random_forest_24h.joblib"
    ),
    "aqi_random_forest_48h": (
        MODEL_DIR / "final_random_forest_48h.joblib"
    ),
    "aqi_random_forest_72h": (
        MODEL_DIR / "final_random_forest_72h.joblib"
    ),
}

# Numeric evaluation metadata.
#
# These are kept numeric because Hopsworks validates the
# `metrics` dictionary as numerical model metrics.
MODEL_METRICS = {
    "aqi_random_forest_24h": {
        "mae": 28.121,
        "rmse": 39.343,
        "r2": 0.621,
        "forecast_horizon_hours": 24.0,
    },
    "aqi_random_forest_48h": {
        "mae": 30.779,
        "rmse": 42.761,
        "r2": 0.548,
        "forecast_horizon_hours": 48.0,
    },
    "aqi_random_forest_72h": {
        "mae": 32.259,
        "rmse": 44.387,
        "r2": 0.507,
        "forecast_horizon_hours": 72.0,
    },
}


def main() -> None:
    """Register all production Random Forest models."""

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is not set."
        )

    if not project_name:
        raise ValueError(
            "HOPSWORKS_PROJECT is not set."
        )

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

        logger.info(
            "Connected to Hopsworks Model Registry."
        )

        for model_name, model_path in MODELS.items():

            # ------------------------------------------------
            # Validate local model file
            # ------------------------------------------------

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )

            if model_path.stat().st_size == 0:
                raise ValueError(
                    f"Model file is empty: {model_path}"
                )

            logger.info(
                "Registering %s from %s",
                model_name,
                model_path,
            )

            metrics = MODEL_METRICS.get(
                model_name,
                {},
            )

            # ------------------------------------------------
            # Create registry model
            # ------------------------------------------------

            model = mr.python.create_model(
                name=model_name,
                description=(
                    "Production Random Forest regression "
                    "model for AQI forecasting."
                ),
                metrics=metrics,
            )

            # ------------------------------------------------
            # Upload/register artifact
            # ------------------------------------------------

            registered_model = model.save(
                str(model_path),
                await_registration=480,
            )

            logger.info(
                "Registered model successfully: %s "
                "version=%s",
                registered_model.name,
                registered_model.version,
            )

            try:
                logger.info(
                    "Model URL: %s",
                    registered_model.get_url(),
                )
            except Exception:
                pass

        logger.info(
            "All production AQI models registered successfully."
        )

    finally:
        try:
            project.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()