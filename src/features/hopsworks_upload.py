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
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(col).strip().lower(),
        ).strip("_")
        for col in df.columns
    ]

    return df


def build_feature_group_name():
    """
    Use a stable feature group name.

    We don't create a new feature group for every retry.
    """
    return "aqi_features"


def upload_feature_group():
    # =========================================================
    # 1. Environment
    # =========================================================

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is not set")

    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT_NAME is not set")

    # =========================================================
    # 2. Connect to Hopsworks
    # =========================================================

    logger.info("Connecting to Hopsworks...")

    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name,
    )

    try:
        fs = project.get_feature_store()

        # =====================================================
        # 3. Load training data
        # =====================================================

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

        # =====================================================
        # 4. Normalize columns
        # =====================================================

        df = normalize_column_names(df)

        logger.info(
            f"Columns: {df.columns.tolist()}"
        )

        # =====================================================
        # 5. Validate required columns
        # =====================================================

        required_columns = [
            "city",
            "timestamp",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # =====================================================
        # 6. Clean timestamp
        # =====================================================

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        if df["timestamp"].isna().any():
            raise ValueError(
                "Some timestamp values could not be converted "
                "to datetime."
            )

        # =====================================================
        # 7. Remove duplicate primary keys
        # =====================================================

        before = len(df)

        df = df.drop_duplicates(
            subset=["city", "timestamp"],
            keep="last",
        ).reset_index(drop=True)

        removed = before - len(df)

        if removed:
            logger.info(
                f"Removed {removed} duplicate "
                f"(city, timestamp) rows"
            )

        # =====================================================
        # 8. Create/get Feature Group
        # =====================================================

        feature_group_name = build_feature_group_name()

        logger.info(
            f"Creating/getting feature group "
            f"'{feature_group_name}'..."
        )

        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=1,
            description="AQI prediction features",
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=False,
            stream=False,
            time_travel_format="HUDI",
        )

        logger.info(
            f"Feature group ready: "
            f"{fg.name}, version: {fg.version}"
        )

        # =====================================================
        # 9. Upload
        # =====================================================

        logger.info(
            f"Uploading {len(df)} rows..."
        )

        fg.insert(
            df,
            overwrite=True,
            operation="upsert",
            storage="offline",
            write_options={
                "wait_for_job": True,
            },
            validation_options={
                "run_validation": False,
                "save_report": False,
            },
        )

        logger.info(
            "Feature upload completed successfully!"
        )

    finally:
        logger.info("Closing Hopsworks connection...")

        try:
            project.close()
        except Exception:
            pass


if __name__ == "__main__":
    upload_feature_group()