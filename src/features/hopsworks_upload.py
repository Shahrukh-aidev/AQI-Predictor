import os
import pandas as pd
import hopsworks

from dotenv import load_dotenv

from src.utils.logger import logger


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

    ddata_path = "data/processed/training_data.parquet"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Training data is empty")

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # ---------------------------------------------------------
    # Clean column names
    # ---------------------------------------------------------
    df.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    ]

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
    logger.info(f"Uploading {len(df)} rows...")

    fg.insert(df)

    logger.info("Feature upload completed successfully!")


if __name__ == "__main__":
    upload_feature_group()