"""
Download production AQI models from Hopsworks Model Registry.

Registered models:
    aqi_random_forest_24h v1
    aqi_random_forest_48h v1
    aqi_random_forest_72h v1

The downloaded artifacts are normalized to:

    models/saved/final_random_forest_24h.joblib
    models/saved/final_random_forest_48h.joblib
    models/saved/final_random_forest_72h.joblib
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import hopsworks
from dotenv import load_dotenv

from src.utils.logger import logger


MODEL_DIR = Path("models/saved")

REGISTERED_MODELS = {
    24: "aqi_random_forest_24h",
    48: "aqi_random_forest_48h",
    72: "aqi_random_forest_72h",
}


def find_joblib_file(path: Path) -> Path:
    """
    Find the downloaded .joblib artifact.

    Hopsworks may return either:
        - a direct file path
        - a directory containing the model artifact
    """

    if path.is_file() and path.suffix.lower() == ".joblib":
        return path

    if path.is_dir():
        candidates = list(path.rglob("*.joblib"))

        if not candidates:
            raise FileNotFoundError(
                f"No .joblib file found inside: {path}"
            )

        # Pick the largest joblib artifact.
        candidates.sort(
            key=lambda p: p.stat().st_size,
            reverse=True,
        )

        return candidates[0]

    raise FileNotFoundError(
        f"Downloaded model artifact not found: {path}"
    )


def download_one_model(
    registry,
    model_name: str,
    horizon: int,
) -> None:
    """Download one registered model safely."""

    expected_path = (
        MODEL_DIR
        / f"final_random_forest_{horizon}h.joblib"
    )

    logger.info(
        "Retrieving registered model: %s v1",
        model_name,
    )

    model = registry.get_model(
        name=model_name,
        version=1,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Hopsworks 5.0.3 does not expose overwrite=True on
    # Model.download().
    #
    # Therefore download into a temporary directory and
    # replace the production file ourselves.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix=f"aqi_model_{horizon}h_"
    ) as temp_dir:

        temp_path = Path(temp_dir)

        logger.info(
            "Downloading %s into temporary directory: %s",
            model_name,
            temp_path,
        )

        downloaded = model.download(
            local_path=str(temp_path),
        )

        downloaded_path = Path(
            downloaded
        )

        logger.info(
            "Hopsworks download returned: %s",
            downloaded_path,
        )

        artifact = find_joblib_file(
            downloaded_path
        )

        logger.info(
            "Downloaded artifact: %s",
            artifact,
        )

        # ----------------------------------------------------
        # Validate downloaded artifact
        # ----------------------------------------------------

        if not artifact.exists():
            raise FileNotFoundError(
                f"Downloaded artifact does not exist: "
                f"{artifact}"
            )

        artifact_size = artifact.stat().st_size

        if artifact_size == 0:
            raise ValueError(
                f"Downloaded model is empty: {artifact}"
            )

        logger.info(
            "Downloaded model size: %.2f MB",
            artifact_size / (1024 * 1024),
        )

        # ----------------------------------------------------
        # Replace existing production artifact
        # ----------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if expected_path.exists():
            logger.info(
                "Replacing existing production model: %s",
                expected_path,
            )
            expected_path.unlink()

        shutil.copy2(
            artifact,
            expected_path,
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Production model was not created: "
            f"{expected_path}"
        )

    final_size = expected_path.stat().st_size

    if final_size == 0:
        raise ValueError(
            f"Production model is empty: {expected_path}"
        )

    logger.info(
        "Production model ready: %s (%.2f MB)",
        expected_path,
        final_size / (1024 * 1024),
    )


def main() -> None:
    """Download all production models from Hopsworks."""

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

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

        registry = (
            project.get_model_registry()
        )

        logger.info(
            "Connected to Hopsworks Model Registry."
        )

        for horizon, model_name in (
            REGISTERED_MODELS.items()
        ):
            download_one_model(
                registry=registry,
                model_name=model_name,
                horizon=horizon,
            )

        logger.info(
            "All production AQI models downloaded successfully."
        )

    finally:

        try:
            project.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()