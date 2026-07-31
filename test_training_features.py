import pandas as pd

from src.features.feature_engineering import (
    build_training_features,
)


# ------------------------------------------------------------
# Load merged AQI + weather data
# ------------------------------------------------------------

df = pd.read_parquet(
    "data/processed/aqi_weather_merged.parquet"
)

print("Input shape:", df.shape)
print("Input cities:")
print(df["city"].value_counts())
print()


# ------------------------------------------------------------
# Build training features
# ------------------------------------------------------------

training = build_training_features(df)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n========================================")
print("FINAL TRAINING DATASET")
print("========================================")

print("Final shape:", training.shape)

print("\nCity counts:")
print(
    training["city"].value_counts()
)

print("\nColumns:")
print(training.columns.tolist())

print("\nMissing values:")

missing = training.isnull().sum()

print(
    missing[missing > 0]
)

print("\nSample:")
print(
    training[
        [
            "city",
            "timestamp",
            "aqi",
            "aqi_lag_1h",
            "aqi_lag_24h",
            "aqi_roll_mean_24h",
            "target_aqi_24h",
            "target_aqi_48h",
            "target_aqi_72h",
        ]
    ].head(10)
)

# test_training_features.py ke end mein add karo

# Drop useless columns
training = training.drop(columns=["pm10", "pm25_to_pm10_ratio"])

# Fill remaining NaN lag/rolling features within city
lag_roll_cols = [c for c in training.columns if "lag" in c or "roll" in c or "change" in c]
training[lag_roll_cols] = training.groupby("city")[lag_roll_cols].transform(lambda x: x.ffill())

# Drop any remaining NaN
before = len(training)
training = training.dropna().reset_index(drop=True)
print(f"Dropped {before - len(training)} rows after imputation")

# Save
training.to_parquet("data/processed/training_data.parquet", index=False)
print("Saved: data/processed/training_data.parquet")
print("Final shape:", training.shape)
print("Missing values:", training.isnull().sum().sum())