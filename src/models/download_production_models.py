"""
Download the latest production AQI models from Hopsworks.

Registered model names:
    aqi_random_forest_24h
    aqi_random_forest_48h
    aqi_random_forest_72h

The newest registered version is automatically selected.

Downloaded artifacts are normalized to:

    models/saved/final_random_forest_24h.joblib
    models/saved/final_random_forest_48h.joblib
    models/saved/final_random_forest_72h.joblib

The local production file is replaced ONLY after the downloaded
model has been successfully validated with joblib.load().
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import hopsworks
import joblib
from dotenv import load_dotenv

from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path(
    "models/saved"
)

REGISTERED_MODELS = {
    24: "aqi_random_forest_24h",
    48: "aqi_random_forest_48h",
    72: "aqi_random_forest_72h",
}

MAX_DOWNLOAD_ATTEMPTS = 3

MIN_MODEL_SIZE_BYTES = 1024


# ============================================================
# Find downloaded artifact
# ============================================================

def find_joblib_file(
    path: Path,
) -> Path:
    """
    Find the .joblib model artifact.

    Hopsworks may return either:
        - a direct file path
        - a directory containing the model files
    """

    path = Path(
        path
    )

    if (
        path.is_file()
        and path.suffix.lower() == ".joblib"
    ):
        return path

    if path.is_dir():

        candidates = list(
            path.rglob("*.joblib")
        )

        if not candidates:

            raise FileNotFoundError(
                f"No .joblib artifact found inside: {path}"
            )

        # The actual model artifact should normally be the
        # largest .joblib file.
        candidates.sort(
            key=lambda item: item.stat().st_size,
            reverse=True,
        )

        return candidates[0]

    raise FileNotFoundError(
        f"Downloaded model path does not exist: {path}"
    )


# ============================================================
# Get latest registered version
# ============================================================

def get_latest_registered_model(
    registry,
    model_name: str,
):
    """
    Retrieve the newest registered version for a model name.

    No model artifact is downloaded here.
    """

    logger.info(
        "Searching Model Registry for %s...",
        model_name,
    )

    models = registry.get_models(
        name=model_name
    )

    if not models:

        raise RuntimeError(
            f"No registered versions found for "
            f"'{model_name}'."
        )

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
# Clear Hopsworks model cache
# ============================================================

def clear_model_cache(
    model,
) -> None:
    """
    Clear cached copies of the selected Hopsworks model.

    This is especially useful after an incomplete download.
    """

    try:

        from hsml.model import Model

        removed = Model.clear_cache(
            project_name=model.project_name,
            model_name=model.name,
            version=int(
                model.version
            ),
        )

        logger.info(
            "Cleared Hopsworks cache for %s v%s "
            "(removed=%s).",
            model.name,
            model.version,
            removed,
        )

    except Exception as exc:

        logger.warning(
            "Could not clear Hopsworks cache for %s v%s: %s",
            model.name,
            model.version,
            exc,
        )


# ============================================================
# Validate downloaded model
# ============================================================

def validate_model_artifact(
    artifact: Path,
) -> None:
    """
    Validate the downloaded joblib artifact.

    Validation includes:
        - file exists
        - file size is reasonable
        - joblib can deserialize it
        - loaded object exposes predict()
    """

    if not artifact.exists():

        raise FileNotFoundError(
            f"Model artifact does not exist: {artifact}"
        )

    file_size = artifact.stat().st_size

    if file_size < MIN_MODEL_SIZE_BYTES:

        raise ValueError(
            f"Downloaded model is too small: "
            f"{file_size} bytes."
        )

    logger.info(
        "Downloaded artifact size: %.2f MB",
        file_size / (1024 * 1024),
    )

    logger.info(
        "Validating joblib artifact..."
    )

    loaded_model = joblib.load(
        artifact
    )

    if not hasattr(
        loaded_model,
        "predict",
    ):

        raise ValueError(
            "Downloaded artifact does not expose "
            "a predict() method."
        )

    logger.info(
        "Model artifact validation successful."
    )


# ============================================================
# Download one model
# ============================================================

def download_one_model(
    registry,
    model_name: str,
    horizon: int,
) -> None:
    """
    Download the latest registered version safely.

    The existing production artifact is never deleted until
    the new artifact has been downloaded and validated.
    """

    expected_path = (
        MODEL_DIR
        / f"final_random_forest_{horizon}h.joblib"
    )

    model = get_latest_registered_model(
        registry,
        model_name,
    )

    latest_version = int(
        model.version
    )

    logger.info(
        "Preparing production download: %s v%s",
        model_name,
        latest_version,
    )

    # --------------------------------------------------------
    # Try multiple times because large model downloads can be
    # interrupted.
    # --------------------------------------------------------

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1,
    ):

        logger.info(
            "Download attempt %d/%d for %s v%s",
            attempt,
            MAX_DOWNLOAD_ATTEMPTS,
            model_name,
            latest_version,
        )

        # Clear cache before retry.
        if attempt > 1:
            clear_model_cache(
                model
            )

        try:

            with tempfile.TemporaryDirectory(
                prefix=f"aqi_model_{horizon}h_"
            ) as temp_dir:

                temp_path = Path(
                    temp_dir
                )

                logger.info(
                    "Downloading %s v%s into %s",
                    model_name,
                    latest_version,
                    temp_path,
                )

                downloaded = model.download(
                    local_path=str(
                        temp_path
                    )
                )

                downloaded_path = Path(
                    downloaded
                )

                logger.info(
                    "Hopsworks returned: %s",
                    downloaded_path,
                )

                artifact = find_joblib_file(
                    downloaded_path
                )

                logger.info(
                    "Found model artifact: %s",
                    artifact,
                )

                # Validate BEFORE touching production file.
                validate_model_artifact(
                    artifact
                )

                # ------------------------------------------------
                # Copy the validated model to a staging file.
                # ------------------------------------------------

                MODEL_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                staging_path = (
                    MODEL_DIR
                    / (
                        f".final_random_forest_"
                        f"{horizon}h.joblib.tmp"
                    )
                )

                if staging_path.exists():
                    staging_path.unlink()

                shutil.copy2(
                    artifact,
                    staging_path,
                )

                # Validate the staged copy too.
                validate_model_artifact(
                    staging_path
                )

                # ------------------------------------------------
                # Atomic replacement.
                #
                # Existing production model is preserved until
                # the new file has passed validation.
                # ------------------------------------------------

                os.replace(
                    staging_path,
                    expected_path,
                )

                logger.info(
                    "Production model updated successfully: "
                    "%s",
                    expected_path,
                )

                logger.info(
                    "Using registered version: %s v%s",
                    model_name,
                    latest_version,
                )

                return

        except Exception as exc:

            last_error = exc

            logger.warning(
                "Download attempt %d/%d failed for %s v%s: %s",
                attempt,
                MAX_DOWNLOAD_ATTEMPTS,
                model_name,
                latest_version,
                exc,
            )

            # Remove staging file if something failed.
            staging_path = (
                MODEL_DIR
                / (
                    f".final_random_forest_"
                    f"{horizon}h.joblib.tmp"
                )
            )

            try:

                if staging_path.exists():
                    staging_path.unlink()

            except Exception:
                pass

    raise RuntimeError(
        f"Failed to download and validate "
        f"{model_name} v{latest_version} after "
        f"{MAX_DOWNLOAD_ATTEMPTS} attempts."
    ) from last_error


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Download the latest registered production version for
    all three AQI forecast horizons.
    """

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
        "=" * 70
    )

    logger.info(
        "STARTING PRODUCTION MODEL DOWNLOAD"
    )

    logger.info(
        "Hopsworks project: %s",
        project_name,
    )

    logger.info(
        "=" * 70
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

        # ----------------------------------------------------
        # Final verification of all production artifacts
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "PRODUCTION MODELS READY"
        )
        print("=" * 70)

        for horizon, model_name in (
            REGISTERED_MODELS.items()
        ):

            path = (
                MODEL_DIR
                / f"final_random_forest_{horizon}h.joblib"
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Production model missing: {path}"
                )

            size_mb = (
                path.stat().st_size
                / (1024 * 1024)
            )

            print(
                f"+{horizon}h | "
                f"{model_name} | "
                f"{size_mb:.2f} MB | "
                f"{path}"
            )

        print("=" * 70)

        logger.info(
            "All latest production AQI models downloaded "
            "and validated successfully."
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
    main()