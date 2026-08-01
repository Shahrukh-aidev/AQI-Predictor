"""
Upload training features to Hopsworks Feature Store.
"""

import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import logger


load_dotenv()


def upload_feature_group():
    logger.info("Connecting to Hopsworks...")

    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )

    fs = project.get_feature_store()

    logger.info("Loading training data...")

    df = pd.read_parquet(
        "data/processed/training_data.parquet"
    )

    # Hopsworks requires timestamp to be timezone-naive.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    ).dt.tz_localize(None)

    logger.info("Creating feature group...")

    # Version 2 is intentional.
    #
    # Version 1 was created with DELTA and failed during fg.insert()
    # with:
    # Generic HdfsObjectStore error
    # RPC listener disconnected
    #
    # Therefore, use a new feature-group version without Delta.
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="NONE",
        online_enabled=False,
        description="AQI + weather features for Pakistani cities",
    )

    logger.info("Uploading %d rows...", len(df))

    # Keep the normal insert.
    fg.insert(df)

    logger.info(
        "Upload complete! Feature group ready at Hopsworks."
    )


if __name__ == "__main__":
    upload_feature_group()