import pandas as pd
import pytest

# This test module depends on the Hopsworks client.
# Skip the module cleanly when the optional dependency is unavailable.
pytest.importorskip(
    "hopsworks",
    reason="Hopsworks integration tests require the Hopsworks client.",
)

from src.features.hopsworks_upload import (
    build_feature_group_name,
    normalize_column_names,
)


def test_normalize_column_names_standardizes_and_strips_spaces():
    df = pd.DataFrame(
        {
            "City Name": ["A"],
            "Timestamp": ["2024-01-01 00:00:00"],
            "PM2.5": [10.0],
        }
    )

    normalized = normalize_column_names(df)

    assert normalized.columns.tolist() == [
        "city_name",
        "timestamp",
        "pm2_5",
    ]


def test_build_feature_group_name_uses_github_run_id(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")

    assert build_feature_group_name() == "aqi_features_12345"