import os
import re
import time

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import logger


def normalize_column_names(df):
    """Normalize DataFrame column names for Hopsworks compatibility."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower())
        .strip("_")
        for col in df.columns
    ]
    return df


def upload_feature_group():
    # ---------------------------------------------------------
    # Load environment variables
    # ---------------------------------------------------------
    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is not set")

    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT_NAME is not set")

    # ---------------------------------------------------------
    # Connect to Hopsworks
    # ---------------------------------------------------------
    logger.info("Connecting to Hopsworks...")

    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name,
    )

    fs = project.get_feature_store()

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------
    logger.info("Loading training data...")

    data_path = "data/processed/training_data.parquet"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found: {data_path}"
        )

    df = pd.read_parquet(data_path)

    if df.empty:
        raise ValueError("Training data is empty")

    logger.info(
        f"Loaded {len(df)} rows and {len(df.columns)} columns"
    )

    # ---------------------------------------------------------
    # Clean column names
    # ---------------------------------------------------------
    df = normalize_column_names(df)

    logger.info(
        f"Columns: {df.columns.tolist()}"
    )

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------
    required_columns = ["city", "timestamp"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # Ensure timestamp is datetime
    # ---------------------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamp values could not be converted to datetime."
        )

    # ---------------------------------------------------------
    # Create / get Feature Group
    # ---------------------------------------------------------
    logger.info("Creating feature group...")

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI prediction features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="DELTA",
    )

    logger.info(
        f"Feature group: {fg.name}, "
        f"version: {fg.version}, "
        f"time travel: {fg.time_travel_format}"
    )

    # ---------------------------------------------------------
    # Upload data
    # ---------------------------------------------------------
    logger.info(
        f"Uploading {len(df)} rows..."
    )

    last_error = None
    for attempt in range(3):
        try:
            fg.insert(
                df,
                overwrite=True,
                operation="upsert",
                write_options={"wait_for_job": True},
            )
            logger.info(
                "Feature upload completed successfully!"
            )
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"Upload attempt {attempt + 1} failed: {exc}. Retrying..."
            )
            time.sleep(5)

    raise RuntimeError(f"Feature upload failed after 3 attempts: {last_error}")


if __name__ == "__main__":
    upload_feature_group()