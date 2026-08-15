import pandas as pd

from src.models.live_predictor import build_current_features


def test_build_current_features_handles_sparse_hourly_aqi():
    aqi_history = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([
                "2026-08-14 00:00:00+00:00",
                "2026-08-14 03:00:00+00:00",
                "2026-08-14 08:00:00+00:00",
                "2026-08-14 12:00:00+00:00",
                "2026-08-14 16:00:00+00:00",
                "2026-08-14 20:00:00+00:00",
            ], utc=True),
            "city": ["Lahore"] * 6,
            "pm25": [20.0, 22.0, 25.0, 30.0, 28.0, 26.0],
            "pm10": [None] * 6,
        }
    )

    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-14 12:00:00+00:00"], utc=True),
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

    features = build_current_features(aqi_history, weather)

    assert not features.empty
    assert 'aqi_lag_24h' in features.columns
    assert 'aqi_roll_mean_24h' in features.columns
    assert features['aqi_lag_24h'].notna().any()
