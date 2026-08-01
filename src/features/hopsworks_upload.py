"""
Upload training features to Hopsworks Feature Store.
"""

import os
import pandas as pd
import hopsworks
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

    df = pd.read_parquet("data/processed/training_data.parquet")

    # Hopsworks needs timestamp as datetime without timezone
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    ).dt.tz_localize(None)

    logger.info("Creating feature group...")

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="DELTA",
        description="AQI + weather features for Pakistani cities",
    )

    logger.info("Uploading %d rows...", len(df))

    fg.insert(df)

    logger.info("Upload complete!")
    print("Feature group 'aqi_features' uploaded successfully.")


if __name__ == "__main__":
    upload_feature_group()