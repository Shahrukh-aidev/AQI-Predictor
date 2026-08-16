from datetime import timedelta

import pandas as pd

from src.models.live_predictor import build_current_features


def test_build_current_features_handles_sparse_hourly_aqi():
    """
    Verify sparse AQI observations are regularized correctly and
    a recent real OpenAQ observation can become a valid live row.
    """

    now = pd.Timestamp.now(tz="UTC").floor("h")

    timestamps = [
        now - pd.Timedelta(hours=48),
        now - pd.Timedelta(hours=45),
        now - pd.Timedelta(hours=42),
        now - pd.Timedelta(hours=39),
        now - pd.Timedelta(hours=36),
        now - pd.Timedelta(hours=33),
        now - pd.Timedelta(hours=30),
        now - pd.Timedelta(hours=27),
        now - pd.Timedelta(hours=24),
        now - pd.Timedelta(hours=21),
        now - pd.Timedelta(hours=18),
        now - pd.Timedelta(hours=15),
        now - pd.Timedelta(hours=12),
        now - pd.Timedelta(hours=9),
        now - pd.Timedelta(hours=6),
        now,
    ]

    aqi_history = pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": ["Lahore"] * len(timestamps),
            "pm25": [
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
                29.0,
                28.0,
                27.0,
                26.0,
                25.0,
            ],
            "pm10": [None] * len(timestamps),
        }
    )

    weather = pd.DataFrame(
        {
            "timestamp": [now],
            "city": ["Lahore"],
            "temperature": [33.0],
            "humidity": [45.0],
            "pressure": [1000.0],
            "wind_speed": [3.4],
            "wind_direction": [180.0],
            "rain_1h": [0.0],
            "rain_3h": [0.0],
        }
    )

    features = build_current_features(
        aqi_history,
        weather,
    )

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1

    # The anchor must be a real OpenAQ observation.
    assert bool(features["is_actual_aqi"].iloc[0])

    required_columns = [
        "pm25",
        "aqi",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "rain_1h",
        "rain_3h",
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

    for column in required_columns:
        assert column in features.columns
        assert pd.notna(features[column].iloc[0])