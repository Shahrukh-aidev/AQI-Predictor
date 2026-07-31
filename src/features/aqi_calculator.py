"""
AQI Calculator for the Pearls AQI Predictor project.

Converts raw pollutant concentrations into AQI values
using the US EPA standard breakpoints.

Why PM2.5 only?
---------------
OpenAQ Pakistan stations predominantly measure PM2.5.
PM2.5 is also the dominant pollutant driving AQI in
Pakistani cities (industrial + vehicular emissions).

EPA PM2.5 AQI Formula
----------------------
AQI = ((AQI_high - AQI_low) / (C_high - C_low))
      * (C - C_low) + AQI_low

Where C is the 24-hour average PM2.5 concentration.

Reference:
https://www.airnow.gov/sites/default/files/2020-05/
aqi-technical-assistance-document-sept2018.pdf
"""

import numpy as np
import pandas as pd

from src.utils.logger import logger


# ── PM2.5 breakpoints (24-hour average, µg/m³) ────────────────────────────────
# Each tuple: (C_low, C_high, AQI_low, AQI_high, category)

PM25_BREAKPOINTS = [
    (0.0,   12.0,   0,   50,  "Good"),
    (12.1,  35.4,  51,  100,  "Moderate"),
    (35.5,  55.4, 101,  150,  "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, 151,  200,  "Unhealthy"),
    (150.5, 250.4, 201, 300,  "Very Unhealthy"),
    (250.5, 350.4, 301, 400,  "Hazardous"),
    (350.5, 500.4, 401, 500,  "Hazardous"),
]


def pm25_to_aqi(concentration: float) -> float:
    """
    Convert a PM2.5 concentration (µg/m³) to an AQI value.

    Parameters
    ----------
    concentration : float
        PM2.5 concentration in µg/m³.
        Negative values and NaN return NaN.

    Returns
    -------
    float
        AQI value, or NaN if conversion is not possible.

    Examples
    --------
    >>> pm25_to_aqi(10.0)
    42.0
    >>> pm25_to_aqi(55.0)
    151.0
    >>> pm25_to_aqi(300.0)
    350.0
    """

    try:
        c = float(concentration)
    except (TypeError, ValueError):
        return float("nan")

    if np.isnan(c) or c < 0:
        return float("nan")

    # Cap at maximum breakpoint
    if c > 500.4:
        return 500.0

    for c_low, c_high, aqi_low, aqi_high, _ in PM25_BREAKPOINTS:
        if c_low <= c <= c_high:
            aqi = (
                (aqi_high - aqi_low) / (c_high - c_low)
            ) * (c - c_low) + aqi_low
            return round(aqi)

    return float("nan")


def aqi_to_category(aqi: float) -> str:
    """
    Return the EPA AQI category string for an AQI value.

    Parameters
    ----------
    aqi : float
        Numeric AQI value.

    Returns
    -------
    str
        EPA category name.
    """

    try:
        val = float(aqi)
    except (TypeError, ValueError):
        return "Unknown"

    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Moderate"
    elif val <= 150:
        return "Unhealthy for Sensitive Groups"
    elif val <= 200:
        return "Unhealthy"
    elif val <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def add_aqi_from_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'aqi' and 'aqi_category' columns to a DataFrame
    that contains a 'pm25' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'pm25' column (µg/m³).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'aqi' and 'aqi_category' added.
    """

    if "pm25" not in df.columns:
        raise ValueError(
            "'pm25' column is required to compute AQI."
        )

    out = df.copy()

    out["aqi"] = (
        pd.to_numeric(out["pm25"], errors="coerce")
        .apply(pm25_to_aqi)
    )

    out["aqi_category"] = out["aqi"].apply(aqi_to_category)

    valid = out["aqi"].notna().sum()

    logger.info(
        "AQI computed for %d/%d rows from PM2.5.",
        valid,
        len(out),
    )

    return out