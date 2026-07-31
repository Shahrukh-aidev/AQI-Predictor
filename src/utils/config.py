"""
Central configuration for the AQI Predictor project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv



# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Load Environment Variables
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# Project Directories
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"


DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# API Configuration
# ============================================================

AQICN_API_KEY = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")


# ============================================================
# Hopsworks Configuration
# ============================================================

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")


# ============================================================
# Model Configuration
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

FORECAST_DAYS = 3