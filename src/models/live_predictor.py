"""
Live AQI inference pipeline.

Purpose
-------
Build live AQI features from recent OpenAQ observations and
current OpenWeather data, then generate:

    Day 1 -> +24h
    Day 2 -> +48h
    Day 3 -> +72h

using the saved tuned Random Forest production models.

Important
---------
This pipeline does NOT use training_data.parquet to determine
the current prediction timestamp.

It uses:

    OpenAQ recent PM2.5
          +
    OpenWeather current weather
          ↓
    hourly AQI history preparation
          ↓
    existing production feature engineering
          ↓
    latest real AQI observation
          ↓
    Random Forest Day 1 / Day 2 / Day 3

Because OpenAQ sensors can have missing hourly observations,
short historical AQI gaps are interpolated before calculating
the same timestamp-aware lag/rolling features used by training.

The latest observation used as the prediction anchor MUST be
a real OpenAQ observation, not an interpolated value.

Cities without a current/recent PM2.5 source are reported as
DATA UNAVAILABLE rather than receiving fabricated predictions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.features.aqi_calculator import add_aqi_from_pm25
from src.features.feature_engineering import (
    add_aqi_features,
    add_time_features,
)
from src.features.openaq_client import OpenAQClient
from src.features.openweather_client import OpenWeatherClient
from src.utils.logger import logger


# ============================================================
# Configuration
# ============================================================

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
]

# Fetch enough history for 24-hour lag/rolling features.
HISTORY_HOURS = 30 * 24

# Maximum age of the actual AQI observation used as the
# live prediction anchor.
MAX_AQI_AGE_HOURS = 12

# Only short gaps are interpolated in the historical AQI
# timeline. Larger gaps remain missing.
MAX_INTERPOLATION_GAP_HOURS = 6

# OpenAQ fallback sensor discovery.
# The default radius was too small for these cities. Some active
# PM2.5 stations exist just outside the narrow search radius and
# only a few days old, so we relax the search window to keep the
# live pipeline robust.
OPENAQ_SENSOR_SEARCH_RADIUS_M = 100_000
OPENAQ_SENSOR_MAX_AGE_HOURS = 168

# Weather can be matched to the latest valid AQI observation
# if the timestamps differ by up to this amount.
WEATHER_MATCH_TOLERANCE_HOURS = 6

MODEL_DIR = Path(
    "models/saved"
)

PREDICTION_OUTPUT = Path(
    "predictions/latest.json"
)

MODEL_PATHS = {
    24: MODEL_DIR / "final_random_forest_24h.joblib",
    48: MODEL_DIR / "final_random_forest_48h.joblib",
    72: MODEL_DIR / "final_random_forest_72h.joblib",
}


# ============================================================
# City coordinates for OpenAQ fallback discovery
# ============================================================

CITY_COORDINATES = {
    "Lahore": {
        "latitude": 31.5204,
        "longitude": 74.3587,
    },
    "Karachi": {
        "latitude": 24.8607,
        "longitude": 67.0011,
    },
    "Islamabad": {
        "latitude": 33.6844,
        "longitude": 73.0479,
    },
}


# ============================================================
# EXACT MODEL INPUT FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "pm25",
    "aqi",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "rain_1h",
    "rain_3h",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_roll_mean_3h",
    "aqi_roll_mean_6h",
    "aqi_roll_mean_24h",
    "aqi_roll_std_24h",
    "aqi_change_1h",
    "aqi_change_rate",
]


HISTORICAL_FEATURE_COLUMNS = [
    "aqi",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_roll_mean_3h",
    "aqi_roll_mean_6h",
    "aqi_roll_mean_24h",
    "aqi_roll_std_24h",
    "aqi_change_1h",
    "aqi_change_rate",
]


# ============================================================
# MODEL LOADING
# ============================================================

def load_models() -> dict[int, object]:
    """Load the three production Random Forest models."""

    models: dict[int, object] = {}

    for horizon, path in MODEL_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Production model not found: {path}"
            )

        logger.info(
            "Loading production model: %s",
            path,
        )

        models[horizon] = joblib.load(path)

    return models


# ============================================================
# OPENAQ SENSOR DISCOVERY
# ============================================================

def discover_active_pm25_sensor(
    client: OpenAQClient,
    city: str,
) -> int | None:
    """
    Find a recently reporting PM2.5 sensor near a city.

    OpenAQ parameter ID 2 corresponds to PM2.5.
    """

    coordinates = CITY_COORDINATES.get(
        city
    )

    if coordinates is None:
        logger.warning(
            "No coordinates configured for %s.",
            city,
        )
        return None

    params = {
        "coordinates": (
            f"{coordinates['latitude']},"
            f"{coordinates['longitude']}"
        ),
        "radius": OPENAQ_SENSOR_SEARCH_RADIUS_M,
        "parameters_id": 2,
        "limit": 1000,
        "page": 1,
    }

    logger.info(
        "Searching for active PM2.5 sensors near %s...",
        city,
    )

    try:
        payload = client._get(
            "/locations",
            params=params,
        )
    except Exception as exc:
        logger.warning(
            "OpenAQ sensor discovery failed for %s: %s",
            city,
            exc,
        )
        return None

    locations = payload.get(
        "results",
        [],
    )

    if not locations:
        logger.warning(
            "No OpenAQ locations found near %s.",
            city,
        )
        return None

    now = pd.Timestamp.now(
        tz="UTC"
    )

    candidates: list[dict] = []

    for location in locations:

        for sensor in location.get(
            "sensors",
            [],
        ):

            parameter = sensor.get(
                "parameter",
                {},
            )

            parameter_id = parameter.get(
                "id"
            )

            parameter_name = str(
                parameter.get(
                    "name",
                    "",
                )
            ).lower()

            if not (
                parameter_id == 2
                or parameter_name in {
                    "pm25",
                    "pm2.5",
                }
            ):
                continue

            sensor_id = sensor.get(
                "id"
            )

            datetime_last = sensor.get(
                "datetimeLast"
            )

            if not sensor_id or not datetime_last:
                continue

            last_utc = datetime_last.get(
                "utc"
            )

            if not last_utc:
                continue

            try:
                last_time = pd.to_datetime(
                    last_utc,
                    utc=True,
                )
            except Exception:
                continue

            age_hours = (
                now - last_time
            ).total_seconds() / 3600

            if age_hours <= OPENAQ_SENSOR_MAX_AGE_HOURS:

                candidates.append(
                    {
                        "sensor_id": int(
                            sensor_id
                        ),
                        "location_id": location.get(
                            "id"
                        ),
                        "location_name": location.get(
                            "name"
                        ),
                        "last_time": last_time,
                        "age_hours": age_hours,
                    }
                )

    if not candidates:
        logger.warning(
            "No recently reporting PM2.5 sensor found "
            "near %s within %dh.",
            city,
            OPENAQ_SENSOR_MAX_AGE_HOURS,
        )
        return None

    candidates.sort(
        key=lambda item: item["last_time"],
        reverse=True,
    )

    selected = candidates[0]

    logger.info(
        "Selected PM2.5 sensor for %s: sensor=%s "
        "location=%s latest=%s age=%.1fh",
        city,
        selected["sensor_id"],
        selected["location_id"],
        selected["last_time"],
        selected["age_hours"],
    )

    return selected["sensor_id"]


# ============================================================
# SENSOR PREPARATION
# ============================================================

def prepare_openaq_sensor(
    client: OpenAQClient,
    city: str,
) -> None:
    """
    Use the configured PM2.5 sensor when active.

    Otherwise search OpenAQ for a recently reporting
    PM2.5 sensor.
    """

    if city not in client.CITY_SENSORS:
        raise ValueError(
            f"City '{city}' not configured in OpenAQClient."
        )

    configured_sensor = (
        client.CITY_SENSORS[city]
        .get("pm25_sensor")
    )

    if configured_sensor:

        logger.info(
            "Testing configured PM2.5 sensor for %s: %s",
            city,
            configured_sensor,
        )

        now = datetime.now(
            timezone.utc
        )

        probe_start = (
            now - timedelta(days=3)
        ).date()

        probe_end = now.date()

        try:

            probe = client.get_hourly_measurements(
                sensor_id=int(
                    configured_sensor
                ),
                date_from=probe_start.isoformat(),
                date_to=probe_end.isoformat(),
                limit=1000,
            )

        except Exception as exc:

            logger.warning(
                "Configured sensor probe failed for %s: %s",
                city,
                exc,
            )

            probe = pd.DataFrame()

        if not probe.empty:

            logger.info(
                "Configured PM2.5 sensor is active for %s.",
                city,
            )

            return

    logger.warning(
        "Configured PM2.5 sensor is unavailable for %s.",
        city,
    )

    discovered_sensor = discover_active_pm25_sensor(
        client,
        city,
    )

    if discovered_sensor is None:

        raise RuntimeError(
            f"No active PM2.5 OpenAQ sensor is currently "
            f"available for {city}."
        )

    client.CITY_SENSORS[
        city
    ]["pm25_sensor"] = discovered_sensor

    logger.info(
        "Using discovered PM2.5 sensor=%s for %s.",
        discovered_sensor,
        city,
    )


# ============================================================
# FETCH RECENT AQI
# ============================================================

def fetch_recent_aqi(
    client: OpenAQClient,
    city: str,
) -> pd.DataFrame:
    """Fetch recent PM2.5 observations for a city."""

    prepare_openaq_sensor(
        client,
        city,
    )

    now_utc = datetime.now(
        timezone.utc
    )

    end_date = now_utc.date()

    start_date = (
        now_utc
        - timedelta(
            hours=HISTORY_HOURS
        )
    ).date()

    logger.info(
        "Fetching recent OpenAQ history for %s: %s -> %s",
        city,
        start_date,
        end_date,
    )

    df = client.fetch_city_historical(
        city=city,
        date_from=start_date.isoformat(),
        date_to=end_date.isoformat(),
    )

    if df.empty:
        raise RuntimeError(
            f"No OpenAQ observations returned for {city}."
        )

    logger.info(
        "Retrieved %d OpenAQ rows for %s.",
        len(df),
        city,
    )

    return df


# ============================================================
# HOURLY AQI REGULARIZATION
# ============================================================

def prepare_hourly_aqi_history(
    aqi_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert sparse OpenAQ observations to an hourly timeline.

    Important:
    - Real OpenAQ observations remain marked as actual.
    - Short internal gaps are time-interpolated.
    - Long gaps remain missing.
    - The latest prediction anchor MUST be actual data.
    """

    df = aqi_history.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "pm25",
        ]
    )

    if df.empty:
        raise ValueError(
            "No valid PM2.5 observations are available."
        )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
    )

    # Keep the actual observation timestamps.
    actual_timestamps = set(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # Create the hourly timeline.
    # --------------------------------------------------------

    df = (
        df
        .set_index("timestamp")
        .sort_index()
    )

    start = df.index.min().floor("h")
    end = df.index.max().floor("h")

    hourly_index = pd.date_range(
        start=start,
        end=end,
        freq="1h",
        tz="UTC",
    )

    hourly = df.reindex(
        hourly_index
    )

    hourly.index.name = "timestamp"

    # City is constant for this function.
    if "city" in df.columns:

        city_values = (
            df["city"]
            .dropna()
            .astype(str)
        )

        if not city_values.empty:
            hourly["city"] = (
                city_values.iloc[-1]
            )

    # --------------------------------------------------------
    # Interpolate PM2.5 only across SHORT internal gaps.
    # --------------------------------------------------------

    hourly["pm25"] = (
        hourly["pm25"]
        .interpolate(
            method="time",
            limit=MAX_INTERPOLATION_GAP_HOURS,
            limit_area="inside",
        )
    )

    # --------------------------------------------------------
    # Calculate AQI from regularized PM2.5.
    # --------------------------------------------------------

    hourly = hourly.reset_index()

    hourly = add_aqi_from_pm25(
        hourly
    )

    # --------------------------------------------------------
    # Mark actual vs interpolated rows.
    # --------------------------------------------------------

    hourly["is_actual_aqi"] = (
        hourly["timestamp"].isin(
            actual_timestamps
        )
    )

    # Only AQI itself is needed for historical features.
    hourly = hourly.dropna(
        subset=["aqi"]
    )

    logger.info(
        "Hourly AQI timeline: %d rows, %d actual observations.",
        len(hourly),
        int(
            hourly["is_actual_aqi"].sum()
        ),
    )

    return hourly


# ============================================================
# CURRENT WEATHER
# ============================================================

def fetch_current_weather(
    client: OpenWeatherClient,
    city: str,
) -> pd.DataFrame:
    """Fetch current OpenWeather observation."""

    logger.info(
        "Fetching current OpenWeather data for %s",
        city,
    )

    df = client.fetch_city_weather(
        city
    )

    if df.empty:
        raise RuntimeError(
            f"No current OpenWeather data returned for {city}."
        )

    return df


# ============================================================
# BUILD LIVE MODEL FEATURES
# ============================================================

def build_current_features(
    aqi_history: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the final one-row model input.

    Steps:
        1. Regularize recent AQI history to hourly frequency.
        2. Apply the existing feature-engineering functions.
        3. Select the latest COMPLETE historical row that
           corresponds to an ACTUAL OpenAQ observation.
        4. Attach current OpenWeather values.
        5. Validate all model inputs.
    """

    weather = weather.copy()

    # --------------------------------------------------------
    # Regularize sparse AQI history.
    # --------------------------------------------------------

    hourly_aqi = prepare_hourly_aqi_history(
        aqi_history
    )

    # --------------------------------------------------------
    # Existing production feature engineering.
    # --------------------------------------------------------

    hourly_aqi = (
        hourly_aqi
        .sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    features = add_time_features(
        hourly_aqi
    )

    features = add_aqi_features(
        features
    )

    features = (
        features
        .sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Confirm expected historical features exist.
    # --------------------------------------------------------

    missing_historical = [
        column
        for column in HISTORICAL_FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_historical:
        raise ValueError(
            "Historical feature engineering did not create: "
            f"{missing_historical}"
        )

    # --------------------------------------------------------
    # Find complete historical rows.
    # --------------------------------------------------------

    complete_rows = (
        features
        .dropna(
            subset=HISTORICAL_FEATURE_COLUMNS
        )
        .copy()
    )

    # IMPORTANT:
    # We only want a real OpenAQ observation to serve as the
    # current anchor.
    complete_rows = complete_rows[
        complete_rows["is_actual_aqi"]
    ].copy()

    if complete_rows.empty:
        raise RuntimeError(
            "No actual OpenAQ observation has complete "
            "historical features after hourly regularization."
        )

    # --------------------------------------------------------
    # Latest actual complete row.
    # --------------------------------------------------------

    latest = (
        complete_rows
        .sort_values("timestamp")
        .iloc[-1:]
        .copy()
        .reset_index(drop=True)
    )

    latest_timestamp = pd.to_datetime(
        latest["timestamp"].iloc[0],
        utc=True,
    )

    # --------------------------------------------------------
    # Check age.
    # --------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    age_hours = (
        now - latest_timestamp
    ).total_seconds() / 3600

    logger.info(
        "Latest actual complete AQI observation: %s",
        latest_timestamp,
    )

    logger.info(
        "Age of AQI observation: %.2f hours",
        age_hours,
    )

    if age_hours > MAX_AQI_AGE_HOURS:

        raise RuntimeError(
            f"Latest complete actual AQI observation "
            f"is {age_hours:.1f} hours old. "
            f"Maximum allowed age is "
            f"{MAX_AQI_AGE_HOURS} hours."
        )

    # --------------------------------------------------------
    # Current weather.
    # --------------------------------------------------------

    required_weather = [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "rain_1h",
        "rain_3h",
    ]

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        utc=True,
        errors="coerce",
    )

    weather = weather.dropna(
        subset=["timestamp"]
    )

    if weather.empty:
        raise RuntimeError(
            "Current weather has no valid timestamp."
        )

    missing_weather = [
        column
        for column in required_weather
        if column not in weather.columns
    ]

    if missing_weather:
        raise ValueError(
            "Current weather missing columns: "
            f"{missing_weather}"
        )

    weather = (
        weather[
            required_weather
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Attach current weather to latest actual AQI row.
    # --------------------------------------------------------

    latest = pd.merge_asof(
        latest.sort_values("timestamp"),
        weather,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(
            hours=WEATHER_MATCH_TOLERANCE_HOURS
        ),
    )

    # --------------------------------------------------------
    # Check final model columns.
    # --------------------------------------------------------

    missing_model_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in latest.columns
    ]

    if missing_model_columns:
        raise ValueError(
            "Final model row missing columns: "
            f"{missing_model_columns}"
        )

    missing_model_values = (
        latest[
            FEATURE_COLUMNS
        ]
        .isna()
        .any()
    )

    missing_model_value_columns = (
        missing_model_values[
            missing_model_values
        ]
        .index
        .tolist()
    )

    if missing_model_value_columns:
        raise ValueError(
            "Final model row contains missing values: "
            f"{missing_model_value_columns}"
        )

    logger.info(
        "Final live feature timestamp: %s",
        latest_timestamp,
    )

    return latest


# ============================================================
# PREDICT ONE CITY
# ============================================================

def predict_city(
    city: str,
    models: dict[int, object],
) -> dict:
    """Generate Day-1, Day-2 and Day-3 predictions."""

    openaq = OpenAQClient()
    openweather = OpenWeatherClient()

    # --------------------------------------------------------
    # AQI history
    # --------------------------------------------------------

    aqi_history = fetch_recent_aqi(
        openaq,
        city,
    )

    # --------------------------------------------------------
    # Current weather
    # --------------------------------------------------------

    weather = fetch_current_weather(
        openweather,
        city,
    )

    # --------------------------------------------------------
    # Final model row
    # --------------------------------------------------------

    features = build_current_features(
        aqi_history,
        weather,
    )

    model_input = features[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Prediction timestamp
    # --------------------------------------------------------

    latest_timestamp = pd.to_datetime(
        features["timestamp"].iloc[0],
        utc=True,
    )

    current_aqi = float(
        features["aqi"].iloc[0]
    )

    predictions: dict[int, dict] = {}

    # --------------------------------------------------------
    # Day 1 / Day 2 / Day 3
    # --------------------------------------------------------

    for horizon, model in models.items():

        prediction = float(
            model.predict(
                model_input
            )[0]
        )

        prediction = max(
            0.0,
            prediction,
        )

        forecast_timestamp = (
            latest_timestamp
            + pd.Timedelta(
                hours=horizon
            )
        )

        predictions[horizon] = {
            "forecast_timestamp":
                forecast_timestamp.isoformat(),
            "predicted_aqi":
                round(
                    prediction,
                    2,
                ),
        }

    return {
        "status": "success",
        "city": city,
        "latest_observation":
            latest_timestamp.isoformat(),
        "current_aqi":
            round(
                current_aqi,
                2,
            ),
        "predictions":
            predictions,
    }


# ============================================================
# SAVE
# ============================================================

def save_predictions(
    results: list[dict],
) -> None:
    """Save prediction results to JSON."""

    PREDICTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "model":
            "Tuned Random Forest",
        "forecasts":
            results,
    }

    with PREDICTION_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )

    logger.info(
        "Saved predictions to %s",
        PREDICTION_OUTPUT,
    )


# ============================================================
# DISPLAY
# ============================================================

def print_success_result(
    result: dict,
) -> None:
    """Print a successful forecast."""

    print()
    print("=" * 68)
    print("LIVE AQI FORECAST")
    print("=" * 68)

    print(
        f"City              : "
        f"{result['city']}"
    )

    print(
        f"Latest observation: "
        f"{result['latest_observation']}"
    )

    print(
        f"Current AQI       : "
        f"{result['current_aqi']}"
    )

    print()
    print("Forecast:")

    for horizon in [
        24,
        48,
        72,
    ]:

        prediction = result[
            "predictions"
        ][horizon]

        day = horizon // 24

        print(
            f"Day {day} (+{horizon}h)"
        )

        print(
            f"  Timestamp : "
            f"{prediction['forecast_timestamp']}"
        )

        print(
            f"  AQI       : "
            f"{prediction['predicted_aqi']}"
        )

        print()

    print("=" * 68)


def print_unavailable_result(
    city: str,
    reason: str,
) -> None:
    """Print an unavailable city."""

    print()
    print("=" * 68)
    print("LIVE AQI DATA UNAVAILABLE")
    print("=" * 68)

    print(
        f"City   : {city}"
    )

    print(
        f"Reason : {reason}"
    )

    print("=" * 68)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run live inference independently for each city.

    A source outage in one city does not prevent another city
    from producing a valid forecast.
    """

    logger.info(
        "Starting live AQI inference pipeline..."
    )

    models = load_models()

    results: list[dict] = []

    for city in CITIES:

        try:

            result = predict_city(
                city=city,
                models=models,
            )

            results.append(
                result
            )

            print_success_result(
                result
            )

        except Exception as exc:

            reason = str(exc)

            logger.error(
                "Live inference unavailable for %s: %s",
                city,
                reason,
            )

            print_unavailable_result(
                city,
                reason,
            )

            # Store the status so the dashboard can distinguish
            # "unavailable" from "not processed".
            results.append(
                {
                    "status": "unavailable",
                    "city": city,
                    "reason": reason,
                    "predictions": {},
                }
            )

    # --------------------------------------------------------
    # Save ALL city statuses.
    #
    # This is deliberately not an all-or-nothing pipeline.
    # --------------------------------------------------------

    save_predictions(
        results
    )

    successful = sum(
        1
        for result in results
        if result.get("status") == "success"
    )

    unavailable = len(results) - successful

    logger.info(
        "Live inference completed: %d successful, "
        "%d unavailable.",
        successful,
        unavailable,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()