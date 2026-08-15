import os
import re
import time

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import logger


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
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


def build_feature_group_name() -> str:
    """Return the canonical feature group name."""
    return "aqi_features"


def build_feature_group_version() -> int:
    """
    Use a fresh version well beyond the corrupted v1-v5 range.
    Override via HOPSWORKS_FEATURE_GROUP_VERSION env var if needed.
    """
    return int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "6"))


def normalize_dtypes_for_hopsworks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize dtypes to improve compatibility with the
    existing Hopsworks Feature Group schema.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="raise",
            utc=True,
        )

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    if "city" in df.columns:
        df["city"] = df["city"].astype(str)

    # --------------------------------------------------------
    # is_weekend
    #
    # Hopsworks schema expects int.
    # pandas may otherwise expose this as int64/bigint.
    # --------------------------------------------------------

    if "is_weekend" in df.columns:
        df["is_weekend"] = (
            pd.to_numeric(
                df["is_weekend"],
                errors="raise",
            )
            .astype("int32")
        )

    return df


def insert_feature_group_with_retry(fg, df: pd.DataFrame, max_retries: int = 3):
    """
    Retry transient Hopsworks/HDFS write errors.

    NOTE: overwrite=True is intentionally omitted. It triggers
    _delete_content internally which POSTs to /featuregroups/<id>/clear
    and 500s on any corrupted metadata. Since we are always writing to
    a fresh version, there is no prior data to clear.
    """
    transient_tokens = (
        "RPC listener disconnected",
        "Generic HdfsObjectStore error",
        "ConnectionReset",
        "ConnectionAborted",
        "Failed to libgssapi_krb5",
    )

    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(
                df,
                write_options={"wait_for_job": True},
                validation_options={
                    "run_validation": False,
                    "save_report": False,
                },
            )
            return
        except Exception as exc:  # pragma: no cover - network-side failure path
            message = str(exc)
            is_transient = any(token in message for token in transient_tokens)

            if not is_transient or attempt == max_retries:
                raise

            logger.warning(
                "Transient Hopsworks write error on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
            time.sleep(5)


def upload_feature_group():
    # =========================================================
    # 1. Environment
    # =========================================================

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is not set")

    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT is not set")

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
            "Loaded %d rows and %d columns",
            len(df),
            len(df.columns),
        )

        # =====================================================
        # 4. Normalize column names
        # =====================================================

        df = normalize_column_names(df)

        logger.info("Columns: %s", df.columns.tolist())

        # =====================================================
        # 5. Validate required columns
        # =====================================================

        required_columns = ["city", "timestamp"]

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
        # 6. Normalize dtypes
        # =====================================================

        df = normalize_dtypes_for_hopsworks(df)

        logger.info("timestamp dtype: %s", df["timestamp"].dtype)

        if "is_weekend" in df.columns:
            logger.info("is_weekend dtype: %s", df["is_weekend"].dtype)

        # =====================================================
        # 7. Validate timestamps
        # =====================================================

        if df["timestamp"].isna().any():
            raise ValueError(
                "Some timestamp values could not be converted to datetime."
            )

        # =====================================================
        # 8. Remove duplicate primary keys
        # =====================================================

        before = len(df)

        df = df.drop_duplicates(
            subset=["city", "timestamp"],
            keep="last",
        ).reset_index(drop=True)

        removed = before - len(df)

        if removed:
            logger.info(
                "Removed %d duplicate (city, timestamp) rows",
                removed,
            )

        # =====================================================
        # 9. Create/get Feature Group
        #
        # No stale-version deletion loop — that pattern caused
        # ghost DB records (duplicate key on 'name_version')
        # which made _delete_content 500 on every insert attempt.
        # We now write to a clean version (default: 6) that has
        # never been touched. Override via env var if needed:
        #   HOPSWORKS_FEATURE_GROUP_VERSION=7
        # =====================================================

        feature_group_name = build_feature_group_name()
        feature_group_version = build_feature_group_version()

        logger.info(
            "Creating/getting feature group '%s' version %s...",
            feature_group_name,
            feature_group_version,
        )

        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            description="AQI prediction features",
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=False,
            stream=False,
        )

        logger.info(
            "Feature group ready: %s, version: %s",
            fg.name,
            fg.version,
        )

        # =====================================================
        # 10. Upload
        # =====================================================

        logger.info("Uploading %d rows...", len(df))

        insert_feature_group_with_retry(fg, df)

        logger.info("Feature upload completed successfully!")

    finally:
        logger.info("Closing Hopsworks connection...")

        try:
            project.close()
        except Exception:
            pass


if __name__ == "__main__":
    upload_feature_group()