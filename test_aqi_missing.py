import pandas as pd

df = pd.read_parquet(
    "data/processed/aqi_weather_merged.parquet"
)

print("\n=== Missing AQI by city ===")
print(
    df.loc[df["aqi"].isna(), "city"]
    .value_counts()
)

print("\n=== Missing AQI percentage by city ===")
print(
    df.groupby("city")["aqi"]
    .apply(lambda x: x.isna().sum())
)

print("\n=== Total missing AQI ===")
print(df["aqi"].isna().sum())

print("\n=== Rows with missing AQI ===")
print(
    df[df["aqi"].isna()]
    [["timestamp", "city", "pm25", "pm10", "aqi"]]
    .head(30)
)

print("\n=== Missing AQI date ranges ===")

missing = df[df["aqi"].isna()].copy()

for city in missing["city"].unique():

    city_missing = missing[
        missing["city"] == city
    ]

    print(
        f"\n{city}: "
        f"{city_missing['timestamp'].min()} "
        f"→ "
        f"{city_missing['timestamp'].max()}"
    )