from datetime import date

from src.features.openmeteo_client import OpenMeteoClient


client = OpenMeteoClient()

df = client.fetch_multiple_cities(
    cities=["Lahore", "Karachi", "Islamabad"],
    start_date=date(2024, 8, 1),
    end_date=date(2026, 7, 31),
)

print("\nShape:")
print(df.shape)

print("\nRows per city:")
print(df["city"].value_counts())

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst rows:")
print(df.head())

print("\nLast rows:")
print(df.tail())

print("\nMissing values:")
print(df.isna().sum())