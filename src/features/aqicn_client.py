"""
AQICN API client for the AQI Predictor project.

This module retrieves air-quality information from AQICN
and converts the API response into a pandas DataFrame.
"""

from typing import Optional

import pandas as pd
import requests

from src.utils.config import AQICN_API_KEY
from src.utils.logger import logger


class AQICNClient:
    """
    Client for interacting with the AQICN API.
    """

    BASE_URL = "https://api.waqi.info"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 10,
    ):
        """
        Initialize the AQICN client.

        Parameters
        ----------
        api_key : str, optional
            AQICN API token. If not provided, the value from
            the environment configuration is used.

        timeout : int
            Maximum number of seconds to wait for an API response.
        """

        self.api_key = api_key or AQICN_API_KEY
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "AQICN API key is missing. "
                "Add AQICN_API_KEY to your .env file."
            )

    def fetch_city_data(self, city: str) -> pd.DataFrame:
        """
        Fetch current AQI and pollutant data for a city.

        Parameters
        ----------
        city : str
            City name, for example "Lahore".

        Returns
        -------
        pandas.DataFrame
            DataFrame containing AQI and pollutant information.
        """

        if not city or not city.strip():
            raise ValueError("City name cannot be empty.")

        url = f"{self.BASE_URL}/feed/{city}/"

        params = {
            "token": self.api_key,
        }

        logger.info("Fetching AQI data for city: %s", city)

        try:

            response = requests.get(
                url,
                params=params,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.exceptions.Timeout as exc:

            logger.error(
                "AQICN request timed out for city: %s",
                city,
            )

            raise RuntimeError(
                "AQICN API request timed out."
            ) from exc

        except requests.exceptions.RequestException as exc:

            logger.error(
                "AQICN request failed: %s",
                exc,
            )

            raise RuntimeError(
                "Failed to communicate with AQICN API."
            ) from exc

        try:

            payload = response.json()

        except ValueError as exc:

            logger.error("AQICN returned invalid JSON.")

            raise RuntimeError(
                "AQICN returned an invalid JSON response."
            ) from exc

        if payload.get("status") != "ok":

            message = payload.get(
                "data",
                "Unknown AQICN API error.",
            )

            logger.error(
                "AQICN API error: %s",
                message,
            )

            raise RuntimeError(
                f"AQICN API error: {message}"
            )

        data = payload.get("data", {})

        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict) -> pd.DataFrame:
        """
        Convert AQICN JSON data into a DataFrame.
        """

        iaqi = data.get("iaqi", {})

        row = {
            "timestamp": data.get("time", {}).get("iso"),
            "city": data.get("city", {}).get("name"),
            "aqi": data.get("aqi"),
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "so2": iaqi.get("so2", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
        }

        dataframe = pd.DataFrame([row])

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        return dataframe