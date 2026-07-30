# docstring
from pathlib import Path 
from dotenv import load_dotenv
import os

# loading the environment variables
load_dotenv()                   #loads environment variables from a .env file.

# here are the project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# API Keys
# ==========================

AQICN_API_KEY = os.getenv("38dd22f695cc249b32fb7fa2f76d1939b3966b7c")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Hopsworks
# ==========================

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

# Model Settings
# ==========================

RANDOM_STATE = 42

TEST_SIZE = 0.20

FORECAST_DAYS = 3