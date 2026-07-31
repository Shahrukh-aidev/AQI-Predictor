"""
Logging utility for the AQI Predictor project.
"""

import logging

from src.utils.config import LOG_DIR


LOG_FILE = LOG_DIR / "aqi_predictor.log"


logger = logging.getLogger("AQI Predictor")

logger.setLevel(logging.INFO)


# Prevent duplicate handlers if the module is imported multiple times.
if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)

    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)