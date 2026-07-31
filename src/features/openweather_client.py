"""
OpenWeather API client for the AQI Predictor project.

This module retrieves weather information from OpenWeather
and converts the API response into a pandas DataFrame.
"""

from typing import Optional

import pandas as pd
import requests

from src.utils.config import OPENWEATHER_API_KEY
from src.utils.logger import logger


class OpenWeatherClient:
    """
    Client for interacting with the OpenWeather API.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 10,
    ):
        """
        Initialize the OpenWeather client.
        """

        self.api_key = api_key or OPENWEATHER_API_KEY
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenWeather API key is missing. "
                "Add OPENWEATHER_API_KEY to your .env file."
            )

    def fetch_city_weather(self, city: str) -> pd.DataFrame:
        """
        Fetch current weather information for a city.

        Parameters
        ----------
        city : str
            City name, for example "Lahore".

        Returns
        -------
        pandas.DataFrame
            DataFrame containing weather information.
        """

        if not city or not city.strip():
            raise ValueError("City name cannot be empty.")

        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
        }

        logger.info(
            "Fetching weather data for city: %s",
            city,
        )

        try:

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.exceptions.Timeout as exc:

            logger.error(
                "OpenWeather request timed out for city: %s",
                city,
            )

            raise RuntimeError(
                "OpenWeather API request timed out."
            ) from exc

        except requests.exceptions.RequestException as exc:

            logger.error(
                "OpenWeather request failed: %s",
                exc,
            )

            raise RuntimeError(
                "Failed to communicate with OpenWeather API."
            ) from exc

        try:

            data = response.json()

        except ValueError as exc:

            logger.error(
                "OpenWeather returned invalid JSON."
            )

            raise RuntimeError(
                "OpenWeather returned invalid JSON."
            ) from exc

        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict) -> pd.DataFrame:
        """
        Convert OpenWeather JSON response into a DataFrame.
        """

        main = data.get("main", {})
        wind = data.get("wind", {})
        rain = data.get("rain", {})

        row = {
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "city": data.get("name"),
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "visibility": data.get("visibility"),
            "rain_1h": rain.get("1h", 0.0),
            "rain_3h": rain.get("3h", 0.0),
        }

        dataframe = pd.DataFrame([row])

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        return dataframe