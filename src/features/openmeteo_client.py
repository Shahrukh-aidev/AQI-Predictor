"""
Open-Meteo historical weather client for the AQI Predictor project.

Why Open-Meteo instead of OpenWeather for historical data:
- OpenWeather historical API requires a paid plan (~$40/month).
- Open-Meteo historical archive is completely free, no API key needed.
- Covers 1940 to present, hourly resolution.

This client is used ONLY for the one-time historical backfill.
The real-time hourly pipeline still uses OpenWeatherClient.

API docs: https://open-meteo.com/en/docs/historical-weather-api
"""

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from src.utils.logger import logger


# ── City coordinates ───────────────────────────────────────────────────────────
# Open-Meteo needs lat/lon, not city names.
# Add more Pakistani cities here as needed.

CITY_COORDINATES = {
    "Lahore":    {"lat": 31.5204, "lon": 74.3587},
    "Karachi":   {"lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Sukkur":    {"lat": 27.7052, "lon": 68.8574},
}

# ── Variables to fetch from Open-Meteo ────────────────────────────────────────
# These map 1-to-1 with OpenWeather columns so both clients
# produce the same output schema.

HOURLY_VARIABLES = [
    "temperature_2m",        # °C  →  temperature
    "relative_humidity_2m",  # %   →  humidity
    "pressure_msl",          # hPa →  pressure
    "wind_speed_10m",        # km/h → wind_speed
    "wind_direction_10m",    # °   →  wind_direction
    "precipitation",         # mm  →  rain_1h
]

# ── Column rename map ──────────────────────────────────────────────────────────
# Rename Open-Meteo variable names to match OpenWeatherClient output schema.

COLUMN_RENAME = {
    "temperature_2m":       "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl":         "pressure",
    "wind_speed_10m":       "wind_speed",
    "wind_direction_10m":   "wind_direction",
    "precipitation":        "rain_1h",
}


class OpenMeteoClient:
    """
    Client for fetching historical weather data from Open-Meteo.

    Usage
    -----
    client = OpenMeteoClient()

    # Fetch 6 months of hourly weather for Lahore
    df = client.fetch_historical(
        city="Lahore",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 7, 30),
    )
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, timeout: int = 30):
        """
        Parameters
        ----------
        timeout : int
            Request timeout in seconds.
            Set higher than OpenWeather (30s) because historical
            requests return large payloads.
        """
        self.timeout = timeout

    def fetch_historical(
        self,
        city: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly historical weather for a city.

        Parameters
        ----------
        city : str
            City name. Must be in CITY_COORDINATES.
            Example: "Lahore", "Karachi", "Islamabad", "Sukkur"

        start_date : date, optional
            First date to fetch. Defaults to 180 days ago.

        end_date : date, optional
            Last date to fetch. Defaults to yesterday.
            (Open-Meteo archive lags by ~1-2 days.)

        Returns
        -------
        pd.DataFrame
            Hourly weather DataFrame with columns matching
            OpenWeatherClient output schema:
            timestamp, city, temperature, humidity, pressure,
            wind_speed, wind_direction, rain_1h
        """

        # ── Validate city ──────────────────────────────────────────────────────
        if city not in CITY_COORDINATES:
            raise ValueError(
                f"City '{city}' not in CITY_COORDINATES. "
                f"Available: {list(CITY_COORDINATES.keys())}"
            )

        coords = CITY_COORDINATES[city]

        # ── Default date range: last 6 months ──────────────────────────────────
        if end_date is None:
            end_date = date.today() - timedelta(days=2)

        if start_date is None:
            start_date = end_date - timedelta(days=180)

        logger.info(
    "Fetching Open-Meteo historical weather: city=%s  %s to %s",
            city,
            start_date,
            end_date,
        )

        # ── Build request ──────────────────────────────────────────────────────
        params = {
            "latitude":   coords["lat"],
            "longitude":  coords["lon"],
            "start_date": start_date.isoformat(),
            "end_date":   end_date.isoformat(),
            "hourly":     ",".join(HOURLY_VARIABLES),
            "timezone":   "UTC",
            "wind_speed_unit": "ms",
        }

        # ── Call API ───────────────────────────────────────────────────────────
        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()

        except requests.exceptions.Timeout as exc:
            logger.error("Open-Meteo request timed out for city: %s", city)
            raise RuntimeError("Open-Meteo request timed out.") from exc

        except requests.exceptions.RequestException as exc:
            logger.error("Open-Meteo request failed: %s", exc)
            raise RuntimeError("Failed to reach Open-Meteo API.") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("Open-Meteo returned invalid JSON.")
            raise RuntimeError("Open-Meteo returned invalid JSON.") from exc

        # ── Parse response ─────────────────────────────────────────────────────
        df = self._parse_response(payload, city)

        logger.info(
            "Open-Meteo: fetched %d hourly rows for %s.",
            len(df),
            city,
        )

        return df

    def fetch_multiple_cities(
        self,
        cities: Optional[list] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical weather for multiple cities and combine.

        Parameters
        ----------
        cities : list, optional
            List of city names. Defaults to all cities in
            CITY_COORDINATES.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with a 'city' column identifying
            each row's origin.
        """

        if cities is None:
            cities = list(CITY_COORDINATES.keys())

        frames = []

        for city in cities:
            try:
                df = self.fetch_historical(
                    city=city,
                    start_date=start_date,
                    end_date=end_date,
                )
                frames.append(df)

            except RuntimeError as exc:
                logger.error(
                    "Skipping %s due to error: %s",
                    city,
                    exc,
                )

        if not frames:
            raise RuntimeError(
                "No data fetched for any city."
            )

        combined = pd.concat(frames, ignore_index=True)

        logger.info(
            "Combined %d rows across %d cities.",
            len(combined),
            len(frames),
        )

        return combined

    @staticmethod
    def _parse_response(payload: dict, city: str) -> pd.DataFrame:
        """
        Convert Open-Meteo JSON into a DataFrame matching
        OpenWeatherClient output schema.
        """

        hourly = payload.get("hourly", {})

        if not hourly:
            raise RuntimeError(
                "Open-Meteo response missing 'hourly' data."
            )

        df = pd.DataFrame(hourly)

        # ── Rename columns to match OpenWeather schema ─────────────────────────
        df = df.rename(columns=COLUMN_RENAME)

        # ── Parse timestamp ────────────────────────────────────────────────────
        df["timestamp"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce",
        )

        df = df.drop(columns=["time"])

        # ── Add city column ────────────────────────────────────────────────────
        df["city"] = city

        # ── Add rain_3h (not available in Open-Meteo, set to 0) ───────────────
        # OpenWeatherClient has rain_3h — keeping schema consistent.
        df["rain_3h"] = 0.0

        # ── Reorder columns to match OpenWeatherClient output ──────────────────
        col_order = [
            "timestamp",
            "city",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "rain_1h",
            "rain_3h",
        ]

        df = df[col_order]

        # ── Drop rows with null timestamps ─────────────────────────────────────
        df = df.dropna(subset=["timestamp"])

        return df.reset_index(drop=True)