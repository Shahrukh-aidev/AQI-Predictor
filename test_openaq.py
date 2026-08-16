import os

import pandas as pd
import pytest

from src.features.openaq_client import OpenAQClient


def test_openaq_client_exposes_required_methods():
    """Verify methods used by the application exist."""

    client = OpenAQClient()

    assert callable(client.fetch_city_historical)
    assert callable(client.search_locations)


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_TESTS") != "1",
    reason="OpenAQ external API test disabled by default.",
)
def test_openaq_search_locations():
    """Optional integration test for the OpenAQ API."""

    client = OpenAQClient()

    result = client.search_locations(
        iso="PK",
        limit=100,
    )

    assert isinstance(result, pd.DataFrame)