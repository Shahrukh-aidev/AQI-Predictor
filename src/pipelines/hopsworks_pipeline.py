"""
Hopsworks Feature Store Pipeline

Uploads processed features to Hopsworks Feature Store for:
- Version control
- Feature lineage tracking
- Training data management
- Online serving (future)
"""

import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from datetime import datetime
from src.utils.logger import logger

# Load environment variables
load_dotenv()


class HopsworksFeatureStore:
    """Upload features to Hopsworks Feature Store"""

    def __init__(self):
        """Initialize Hopsworks connection"""
        self.api_key = os.getenv("HOPSWORKS_API_KEY")
        self.project_name = os.getenv("HOPSWORKS_PROJECT")
        
        if not self.api_key or not self.project_name:
            raise ValueError("HOPSWORKS_API_KEY and HOPSWORKS_PROJECT must be set in .env")
        
        logger.info(f"Connecting to Hopsworks project: {self.project_name}")
        self.project = hopsworks.login(
            api_key_value=self.api_key,
            project=self.project_name
        )
        self.fs = self.project.get_feature_store()
        logger.info("✅ Connected to Hopsworks Feature Store")

    def upload_historical_aqi(self, data_path: str = "data/processed/historical_aqi.parquet"):
        """Upload historical AQI data"""
        logger.info(f"Reading {data_path}")
        df = pd.read_parquet(data_path)
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        logger.info(f"Uploading {len(df)} historical AQI records")
        
        fg = self.fs.get_or_create_feature_group(
            name="historical_aqi",
            version=1,
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=False,
            stream=False,
            description="Historical AQI measurements from AQICN"
        )
        
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info("✅ Historical AQI uploaded")
        return fg

    def upload_aqi_weather_merged(self, data_path: str = "data/processed/aqi_weather_merged.parquet"):
        """Upload merged AQI + Weather features"""
        logger.info(f"Reading {data_path}")
        df = pd.read_parquet(data_path)
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        logger.info(f"Uploading {len(df)} AQI+Weather records")
        
        fg = self.fs.get_or_create_feature_group(
            name="aqi_weather_features",
            version=1,
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=False,
            stream=False,
            description="Merged AQI and Weather measurements"
        )
        
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info("✅ AQI+Weather features uploaded")
        return fg

    def upload_training_data(self, data_path: str = "data/processed/training_data.parquet"):
        """Upload training dataset"""
        logger.info(f"Reading {data_path}")
        df = pd.read_parquet(data_path)
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        logger.info(f"Uploading {len(df)} training records")
        
        # Create feature group for training data
        fg = self.fs.get_or_create_feature_group(
            name="aqi_training_data",
            version=1,
            primary_key=["city", "timestamp"] if "city" in df.columns else ["timestamp"],
            event_time="timestamp",
            online_enabled=False,
            stream=False,
            description="Training dataset with features and target"
        )
        
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info("✅ Training data uploaded")
        return fg

    def get_training_dataset(self, name: str = "aqi_training_features"):
        """Create or get training dataset"""
        logger.info(f"Creating training dataset: {name}")
        
        # Get feature groups
        historical_fg = self.fs.get_feature_group("historical_aqi", version=1)
        weather_fg = self.fs.get_feature_group("aqi_weather_features", version=1)
        
        # Create training dataset
        query = historical_fg.select(["*"]).join(
            weather_fg.select(["*"]),
            on=["city", "timestamp"],
            how="inner"
        )
        
        td = self.fs.create_training_dataset(
            name=name,
            version=1,
            data_format="parquet",
            description="Combined AQI and weather features for training"
        )
        
        td.insert(query)
        logger.info("✅ Training dataset created")
        return td


def push_features_to_hopsworks():
    """Main function to push all features"""
    try:
        store = HopsworksFeatureStore()
        
        # Upload all feature groups
        logger.info("=" * 60)
        logger.info("UPLOADING FEATURES TO HOPSWORKS")
        logger.info("=" * 60)
        
        store.upload_historical_aqi()
        store.upload_aqi_weather_merged()
        store.upload_training_data()
        
        logger.info("=" * 60)
        logger.info("✅ ALL FEATURES UPLOADED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"View at: https://app.hopsworks.ai/p/{store.project.id}/fs")
        
    except Exception as e:
        logger.error(f"❌ Error uploading features: {e}")
        raise


if __name__ == "__main__":
    push_features_to_hopsworks()
