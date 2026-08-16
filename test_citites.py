import os

import pandas as pd
import pytest

from src.features.aqi_calculator import add_aqi_from_pm25
from src.features.openaq_client import OpenAQClient


def test_configured_cities_have_valid_configuration():
    """Verify every production city is present in OpenAQ configuration."""

    client = OpenAQClient()

    expected_cities = {
        "Lahore",
        "Karachi",
        "Islamabad",
    }

    assert set(client.CITY_SENSORS.keys()) == expected_cities


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_TESTS") != "1",
    reason="External API tests are disabled by default.",
)
def test_lahore_historical_data_can_be_processed():
    """Test the currently active Lahore OpenAQ source."""

    client = OpenAQClient()

    df = client.fetch_city_historical(
        "Lahore",
        "2026-07-01",
        "2026-07-07",
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    df = add_aqi_from_pm25(df)

    assert "aqi" in df.columns
    assert df["aqi"].notna().any()