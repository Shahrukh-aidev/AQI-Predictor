"""
Logging utility for the AQI Predictor project.
"""

import logging

from src.utils.config import LOG_DIR

LOG_FILE = LOG_DIR / "aqi_predictor.log"

logger = logging.getLogger("AQI Predictor")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = logging.FileHandler(LOG_FILE)

file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)