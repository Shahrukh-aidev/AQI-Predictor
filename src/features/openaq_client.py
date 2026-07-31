"""
OpenAQ historical air-quality client.

This module retrieves historical pollutant measurements from
the OpenAQ API v3.

OpenAQ provides pollutant concentrations such as:
    PM2.5
    PM10
    NO2
    SO2
    O3
    CO

We convert these pollutant concentrations into AQI values
using the EPA formula in aqi_calculator.py.

Architecture:

OpenAQ API
    ↓
Locations
    ↓
Sensors
    ↓
Hourly measurements
    ↓
Pollutant DataFrame
    ↓
AQI calculation
"""

from typing import Optional

import pandas as pd
import requests

from src.utils.config import OPENAQ_API_KEY
from src.utils.logger import logger


class OpenAQClient:
    """
    Client for interacting with OpenAQ API v3.
    """

    BASE_URL = "https://api.openaq.org/v3"

    # ── Best stations per city ─────────────────────────────────────────────
    # These sensors were selected after testing OpenAQ availability.
    #
    # Training cities:
    #   Lahore
    #   Karachi
    #   Islamabad
    #
    # Sukkur is intentionally excluded because its historical coverage
    # is not sufficient/reliable for our two-year training requirement.

    CITY_SENSORS = {
        "Lahore": {
            "location_id": 1894641,
            "pm25_sensor": 7466365,
            "pm10_sensor": 7466364,
        },

        "Karachi": {
            "location_id": 8156,
            "pm25_sensor": 23747,
            "pm10_sensor": None,
        },

        "Islamabad": {
            "location_id": 233470,
            "pm25_sensor": 1343270,
            "pm10_sensor": None,
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize OpenAQ client.

        Parameters
        ----------
        api_key : str, optional
            OpenAQ API key. If not supplied, it is loaded
            from the .env configuration.

        timeout : int
            HTTP request timeout in seconds.
        """

        self.api_key = api_key or OPENAQ_API_KEY
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenAQ API key is missing. "
                "Add OPENAQ_API_KEY to your .env file."
            )

        self.headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }

    # ── Internal request helper ────────────────────────────────────────────

    def _get(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Perform authenticated GET request to OpenAQ.
        """

        url = f"{self.BASE_URL}{endpoint}"

        logger.info("OpenAQ request: %s", endpoint)

        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.exceptions.Timeout as exc:
            logger.error(
                "OpenAQ request timed out: %s",
                endpoint,
            )

            raise RuntimeError(
                "OpenAQ API request timed out."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            logger.error(
                "OpenAQ HTTP error %s: %s",
                response.status_code,
                response.text[:500],
            )

            raise RuntimeError(
                f"OpenAQ API returned HTTP {response.status_code}."
            ) from exc

        except requests.exceptions.RequestException as exc:
            logger.error(
                "OpenAQ request failed: %s",
                exc,
            )

            raise RuntimeError(
                "Failed to communicate with OpenAQ API."
            ) from exc

        try:
            return response.json()

        except ValueError as exc:
            logger.error(
                "OpenAQ returned invalid JSON."
            )

            raise RuntimeError(
                "OpenAQ returned invalid JSON."
            ) from exc

    # ── Discovery methods ──────────────────────────────────────────────────

    def search_locations(
        self,
        country: str = "PK",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Find OpenAQ monitoring locations in a country.

        Parameters
        ----------
        country : str
            ISO country code. Pakistan = PK.

        limit : int
            Maximum number of locations to retrieve.

        Returns
        -------
        pandas.DataFrame
            Location information.
        """

        params = {
            "iso": country,
            "limit": min(limit, 1000),
        }

        payload = self._get(
            "/locations",
            params=params,
        )

        results = payload.get(
            "results",
            [],
        )

        if not results:
            logger.warning(
                "No OpenAQ locations found for country=%s",
                country,
            )

            return pd.DataFrame()

        rows = []

        for location in results:
            rows.append(
                {
                    "id": location.get("id"),
                    "name": location.get("name"),
                    "city": location.get("city"),
                    "country": location.get(
                        "country",
                        {},
                    ).get("code"),
                    "latitude": location.get(
                        "coordinates",
                        {},
                    ).get("latitude"),
                    "longitude": location.get(
                        "coordinates",
                        {},
                    ).get("longitude"),
                }
            )

        dataframe = pd.DataFrame(rows)

        logger.info(
            "Found %d OpenAQ locations in %s.",
            len(dataframe),
            country,
        )

        return dataframe

    def get_location(
        self,
        location_id: int,
    ) -> dict:
        """
        Retrieve detailed information about one location.
        """

        payload = self._get(
            f"/locations/{location_id}"
        )

        return payload.get(
            "results",
            {},
        )

    def get_location_sensors(
        self,
        location_id: int,
    ) -> pd.DataFrame:
        """
        Retrieve sensors belonging to a location.

        Returns
        -------
        pandas.DataFrame
            Sensor information including parameter names.
        """

        payload = self._get(
            f"/locations/{location_id}/sensors"
        )

        results = payload.get(
            "results",
            [],
        )

        if not results:
            logger.warning(
                "No sensors found for location=%s",
                location_id,
            )

            return pd.DataFrame()

        rows = []

        for sensor in results:
            parameter = sensor.get(
                "parameter",
                {},
            )

            rows.append(
                {
                    "sensor_id": sensor.get("id"),
                    "parameter_id": parameter.get("id"),
                    "parameter_name": parameter.get("name"),
                    "unit": parameter.get("units"),
                }
            )

        dataframe = pd.DataFrame(rows)

        logger.info(
            "Found %d sensors for location=%s.",
            len(dataframe),
            location_id,
        )

        return dataframe

    # ── Measurement fetch ──────────────────────────────────────────────────

    def get_hourly_measurements(
        self,
        sensor_id: int,
        date_from: str,
        date_to: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Retrieve hourly aggregated measurements for a sensor.

        Automatically paginates through all results so that
        date ranges longer than 1000 hours are fully fetched.

        OpenAQ hard-caps limit at 1000 per request.

        Parameters
        ----------
        sensor_id : int
            OpenAQ sensor ID.

        date_from : str
            Start date in YYYY-MM-DD format.

        date_to : str
            End date in YYYY-MM-DD format.

        limit : int
            Records per page. Maximum allowed is 1000.

        Returns
        -------
        pandas.DataFrame
            Columns: timestamp, value
        """

        # OpenAQ allows a maximum of 1000 records per request.
        page_limit = min(max(limit, 1), 1000)

        all_rows = []
        page = 1

        while True:

            params = {
                "datetime_from": (
                    f"{date_from}T00:00:00Z"
                ),
                "datetime_to": (
                    f"{date_to}T23:59:59Z"
                ),
                "limit": page_limit,
                "page": page,
            }

            payload = self._get(
                f"/sensors/{sensor_id}/hours",
                params=params,
            )

            results = payload.get(
                "results",
                [],
            )

            if not results:
                break

            for item in results:
                all_rows.append(
                    {
                        "timestamp": item.get(
                            "period",
                            {},
                        ).get(
                            "datetimeFrom",
                            {},
                        ).get("utc"),
                        "value": item.get("value"),
                    }
                )

            logger.info(
                "OpenAQ sensor=%s page=%d fetched %d records.",
                sensor_id,
                page,
                len(results),
            )

            # If fewer than page_limit records are returned,
            # this is the final page.
            if len(results) < page_limit:
                break

            page += 1

        if not all_rows:
            logger.warning(
                "No hourly measurements found for sensor=%s",
                sensor_id,
            )

            return pd.DataFrame()

        dataframe = pd.DataFrame(
            all_rows
        )

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
            errors="coerce",
        )

        dataframe["value"] = pd.to_numeric(
            dataframe["value"],
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=["timestamp"]
        )

        dataframe = dataframe.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        logger.info(
            "Retrieved %d total hourly records for sensor=%s.",
            len(dataframe),
            sensor_id,
        )

        return dataframe

    # ── High-level historical fetch ────────────────────────────────────────

    def fetch_city_historical(
        self,
        city: str,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame:
        """
        Fetch historical hourly PM2.5 and PM10 data
        for one city.

        PM10 is optional because not all selected stations
        provide PM10 historical measurements.

        Parameters
        ----------
        city : str
            One of:
                Lahore
                Karachi
                Islamabad

        date_from : str
            Start date in YYYY-MM-DD format.

        date_to : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        pandas.DataFrame
            Columns:
                timestamp
                city
                pm25
                pm10
        """

        if city not in self.CITY_SENSORS:
            raise ValueError(
                f"City '{city}' not in CITY_SENSORS. "
                f"Available: {list(self.CITY_SENSORS.keys())}"
            )

        sensors = self.CITY_SENSORS[city]

        logger.info(
            "Fetching OpenAQ historical for %s: %s to %s",
            city,
            date_from,
            date_to,
        )

        # ── PM2.5 ──────────────────────────────────────────────────────────

        pm25_df = self.get_hourly_measurements(
            sensor_id=sensors["pm25_sensor"],
            date_from=date_from,
            date_to=date_to,
            limit=1000,
        ).rename(
            columns={"value": "pm25"}
        )

        if pm25_df.empty:
            logger.warning(
                "No PM2.5 data returned for %s.",
                city,
            )

            return pd.DataFrame()

        # ── PM10 ───────────────────────────────────────────────────────────

        pm10_sensor = sensors.get(
            "pm10_sensor"
        )

        if pm10_sensor:

            pm10_df = self.get_hourly_measurements(
                sensor_id=pm10_sensor,
                date_from=date_from,
                date_to=date_to,
                limit=1000,
            ).rename(
                columns={"value": "pm10"}
            )

            if not pm10_df.empty:

                merged = pd.merge(
                    pm25_df,
                    pm10_df,
                    on="timestamp",
                    how="left",
                )

            else:

                logger.warning(
                    "PM10 sensor=%s returned no data for %s. "
                    "Filling with NaN.",
                    pm10_sensor,
                    city,
                )

                merged = pm25_df.copy()
                merged["pm10"] = float("nan")

        else:

            merged = pm25_df.copy()
            merged["pm10"] = float("nan")

        # ── Add city ───────────────────────────────────────────────────────

        merged["city"] = city

        merged = merged[
            [
                "timestamp",
                "city",
                "pm25",
                "pm10",
            ]
        ]

        merged = merged.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        logger.info(
            "OpenAQ historical: %d rows for %s.",
            len(merged),
            city,
        )

        return merged

    # ── Multiple-city historical fetch ─────────────────────────────────────

    def fetch_multiple_cities_historical(
        self,
        cities: Optional[list] = None,
        date_from: str = "2024-07-31",
        date_to: str = "2026-07-30",
    ) -> pd.DataFrame:
        """
        Fetch historical PM2.5/PM10 data for multiple cities
        and combine into one DataFrame.

        Parameters
        ----------
        cities : list, optional
            List of cities.

            Defaults to:
                Lahore
                Karachi
                Islamabad

        date_from : str
            Start date in YYYY-MM-DD format.

        date_to : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        pandas.DataFrame
            Combined DataFrame sorted by city + timestamp.
        """

        if cities is None:
            cities = list(
                self.CITY_SENSORS.keys()
            )

        frames = []

        for city in cities:

            try:

                df = self.fetch_city_historical(
                    city=city,
                    date_from=date_from,
                    date_to=date_to,
                )

                if not df.empty:
                    frames.append(df)

            except RuntimeError as exc:

                logger.error(
                    "Skipping %s: %s",
                    city,
                    exc,
                )

        if not frames:
            raise RuntimeError(
                "No historical data fetched for any city."
            )

        combined = pd.concat(
            frames,
            ignore_index=True,
        )

        combined = combined.sort_values(
            [
                "city",
                "timestamp",
            ]
        ).reset_index(drop=True)

        logger.info(
            "Combined OpenAQ historical: %d rows, %d cities.",
            len(combined),
            len(frames),
        )

        return combined