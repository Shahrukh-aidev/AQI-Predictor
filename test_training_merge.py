import pandas as pd

from src.features.openmeteo_client import OpenMeteoClient


# ============================================================
# 1. Load historical AQI
# ============================================================

aqi_df = pd.read_parquet(
    "data/processed/historical_aqi.parquet"
)

print("\nAQI dataset:")
print(aqi_df.shape)
print(aqi_df.columns.tolist())


# ============================================================
# 2. Fetch Open-Meteo historical weather
# ============================================================

client = OpenMeteoClient()

weather_df = client.fetch_multiple_cities(
    cities=["Lahore", "Karachi", "Islamabad"],
    start_date=aqi_df["timestamp"].min().date(),
    end_date=aqi_df["timestamp"].max().date(),
)

print("\nWeather dataset:")
print(weather_df.shape)


# ============================================================
# 3. Standardize timestamps
# ============================================================

aqi_df["timestamp"] = pd.to_datetime(
    aqi_df["timestamp"],
    utc=True,
)

weather_df["timestamp"] = pd.to_datetime(
    weather_df["timestamp"],
    utc=True,
)


# ============================================================
# 4. Sort for merge_asof
# ============================================================

aqi_df = aqi_df.sort_values(
    ["timestamp", "city"]
).reset_index(drop=True)

weather_df = weather_df.sort_values(
    ["timestamp", "city"]
).reset_index(drop=True)


# ============================================================
# 5. Merge AQI with nearest weather observation
# ============================================================

merged = pd.merge_asof(
    aqi_df,
    weather_df,
    on="timestamp",
    by="city",
    direction="nearest",
    tolerance=pd.Timedelta("30min"),
)


# ============================================================
# 6. Check merge quality
# ============================================================

print("\nMerged dataset:")
print(merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nRows per city:")
print(merged["city"].value_counts())

print("\nMissing weather values after merge:")
weather_columns = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "rain_1h",
    "rain_3h",
]

print(
    merged[weather_columns].isna().sum()
)

print("\nMissing AQI values:")
print(merged["aqi"].isna().sum())


# ============================================================
# 7. Display sample
# ============================================================

print("\nSample:")
print(
    merged[
        [
            "timestamp",
            "city",
            "aqi",
            "pm25",
            "pm10",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "rain_1h",
        ]
    ].head(10)
)


# ============================================================
# 8. Save temporary merged dataset
# ============================================================

output_path = "data/processed/aqi_weather_merged.parquet"

merged.to_parquet(
    output_path,
    index=False,
)

print(f"\nSaved: {output_path}")