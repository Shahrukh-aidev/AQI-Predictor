import pandas as pd

from src.features.feature_engineering import build_features


def main():

    # --------------------------------------------------------
    # Create 72 hours of sample AQI data
    # --------------------------------------------------------

    timestamps = pd.date_range(
        start="2026-07-01 00:00:00",
        periods=72,
        freq="h",
        tz="UTC",
    )

    aqicn_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": ["Lahore"] * 72,
            "aqi": [50 + (i % 30) for i in range(72)],
            "pm25": [30 + (i % 20) for i in range(72)],
            "pm10": [50 + (i % 25) for i in range(72)],
            "co": [1.0] * 72,
            "no2": [20.0] * 72,
            "so2": [5.0] * 72,
            "o3": [30.0] * 72,
        }
    )

    # --------------------------------------------------------
    # Create weather data
    # --------------------------------------------------------

    weather_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": ["Lahore"] * 72,
            "temperature": [32.0 + (i % 5) for i in range(72)],
            "feels_like": [34.0 + (i % 5) for i in range(72)],
            "humidity": [55.0 + (i % 10) for i in range(72)],
            "pressure": [1008.0] * 72,
            "wind_speed": [3.5] * 72,
            "wind_direction": [180.0] * 72,
            "visibility": [10000.0] * 72,
            "rain_1h": [0.0] * 72,
            "rain_3h": [0.0] * 72,
        }
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = build_features(
        aqicn_df,
        weather_df,
        target_horizon=1,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING TEST")
    print("=" * 70)

    print("\nDataset shape:")
    print(features.shape)

    print("\nColumns:")
    print(features.columns.tolist())

    print("\nFirst 5 rows:")
    print(features.head())

    print("\nLast 5 rows:")
    print(features.tail())

    print("\nMissing values:")
    print(features.isna().sum())

    print("\nTarget:")
    print(
        features[
            [
                "timestamp",
                "aqi",
                "target_aqi_1h",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()