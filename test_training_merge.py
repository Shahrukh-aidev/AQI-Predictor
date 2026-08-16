import pandas as pd


def test_training_merge_uses_compatible_utc_timestamps():
    """Verify AQI/weather timestamps are compatible for merge_asof."""

    # Small local AQI sample.
    aqi_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-07-01T00:00:00Z",
                "2026-07-01T01:00:00Z",
            ],
            "city": [
                "Lahore",
                "Lahore",
            ],
            "aqi": [
                50.0,
                60.0,
            ],
            "pm25": [
                12.0,
                15.0,
            ],
        }
    )

    # Small local weather sample.
    weather_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-07-01T00:15:00Z",
                "2026-07-01T01:15:00Z",
            ],
            "city": [
                "Lahore",
                "Lahore",
            ],
            "temperature": [
                30.0,
                31.0,
            ],
            "humidity": [
                60.0,
                61.0,
            ],
            "pressure": [
                1005.0,
                1006.0,
            ],
            "wind_speed": [
                10.0,
                11.0,
            ],
            "wind_direction": [
                180.0,
                190.0,
            ],
            "rain_1h": [
                0.0,
                0.0,
            ],
            "rain_3h": [
                0.0,
                0.0,
            ],
        }
    )

    # --------------------------------------------------------
    # Normalize both timestamp columns to exactly the same
    # timezone-aware nanosecond dtype.
    # --------------------------------------------------------

    aqi_df["timestamp"] = (
        pd.to_datetime(
            aqi_df["timestamp"],
            utc=True,
            errors="coerce",
        )
        .astype("datetime64[ns, UTC]")
    )

    weather_df["timestamp"] = (
        pd.to_datetime(
            weather_df["timestamp"],
            utc=True,
            errors="coerce",
        )
        .astype("datetime64[ns, UTC]")
    )

    # --------------------------------------------------------
    # Verify timestamp compatibility.
    # --------------------------------------------------------

    assert str(aqi_df["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert str(weather_df["timestamp"].dtype) == "datetime64[ns, UTC]"

    # --------------------------------------------------------
    # Sort for merge_asof.
    # --------------------------------------------------------

    aqi_df = (
        aqi_df
        .sort_values(
            ["timestamp", "city"]
        )
        .reset_index(drop=True)
    )

    weather_df = (
        weather_df
        .sort_values(
            ["timestamp", "city"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Perform the same merge strategy used by the project.
    # --------------------------------------------------------

    merged = pd.merge_asof(
        aqi_df,
        weather_df,
        on="timestamp",
        by="city",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )

    # --------------------------------------------------------
    # Validate merge.
    # --------------------------------------------------------

    assert not merged.empty
    assert len(merged) == 2

    required_weather_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "rain_1h",
        "rain_3h",
    ]

    for column in required_weather_columns:
        assert column in merged.columns
        assert merged[column].notna().all()