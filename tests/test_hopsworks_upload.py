import pandas as pd

from src.features.hopsworks_upload import normalize_column_names


def test_normalize_column_names_standardizes_and_strips_spaces():
    df = pd.DataFrame(
        {
            "City Name": ["A"],
            "Timestamp": ["2024-01-01 00:00:00"],
            "PM2.5": [10.0],
        }
    )

    normalized = normalize_column_names(df)

    assert normalized.columns.tolist() == ["city_name", "timestamp", "pm2_5"]
