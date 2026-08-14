"""
Historical AQI data backfill for the Pearls AQI Predictor project.

Fetches two years of historical PM2.5 data from OpenAQ for:
    - Lahore
    - Karachi
    - Islamabad

Converts PM2.5 -> AQI using the EPA AQI calculator and saves
the combined historical dataset as a Parquet file.

Historical period:
    2024-08-01 -> 2026-07-31

The script fetches data month-by-month so that:
    1. API requests remain manageable.
    2. Missing periods do not stop the entire process.
    3. We can see exactly which city/month has data.
"""

from __future__ import annotations

import calendar
from datetime import date

import pandas as pd

from src.features.aqi_calculator import add_aqi_from_pm25
from src.features.openaq_client import OpenAQClient
from src.utils.logger import logger


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = date(2024, 8, 1)
END_DATE = date(2026, 7, 31)

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
]

OUTPUT_PATH = "data/processed/historical_aqi.parquet"


# ============================================================
# MONTH GENERATOR
# ============================================================

def generate_month_ranges(
    start_date: date,
    end_date: date,
):
    """
    Generate inclusive monthly date ranges.

    Example:
        2024-08-01 -> 2024-08-31
        2024-09-01 -> 2024-09-30
    """

    current = date(
        start_date.year,
        start_date.month,
        1,
    )

    while current <= end_date:

        last_day = calendar.monthrange(
            current.year,
            current.month,
        )[1]

        month_end = date(
            current.year,
            current.month,
            last_day,
        )

        # Do not go beyond requested END_DATE.
        if month_end > end_date:
            month_end = end_date

        yield current, month_end

        # Move to next month.
        if current.month == 12:
            current = date(
                current.year + 1,
                1,
                1,
            )
        else:
            current = date(
                current.year,
                current.month + 1,
                1,
            )


# ============================================================
# FETCH ONE CITY
# ============================================================

def fetch_city(
    client: OpenAQClient,
    city: str,
) -> pd.DataFrame:
    """
    Fetch the complete historical period for one city.

    Missing months are skipped rather than causing the
    entire backfill to fail.
    """

    city_frames = []

    logger.info("=" * 70)
    logger.info(
        "Starting historical backfill for %s",
        city,
    )
    logger.info("=" * 70)

    for month_start, month_end in generate_month_ranges(
        START_DATE,
        END_DATE,
    ):

        start_str = month_start.isoformat()
        end_str = month_end.isoformat()

        logger.info(
            "%s | Fetching %s -> %s",
            city,
            start_str,
            end_str,
        )

        try:
            df = client.fetch_city_historical(
                city=city,
                date_from=start_str,
                date_to=end_str,
            )

        except Exception as exc:
            logger.error(
                "%s | Failed for %s -> %s: %s",
                city,
                start_str,
                end_str,
                exc,
            )
            continue

        if df.empty:
            logger.warning(
                "%s | No data for %s -> %s",
                city,
                start_str,
                end_str,
            )
            continue

        city_frames.append(df)

        logger.info(
            "%s | %s -> %s | %d rows",
            city,
            start_str,
            end_str,
            len(df),
        )

    if not city_frames:
        logger.warning(
            "%s | No historical data found.",
            city,
        )
        return pd.DataFrame()

    city_df = pd.concat(
        city_frames,
        ignore_index=True,
    )

    return city_df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_historical_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and standardize the combined historical dataset.

    PM2.5 values <= 0 are treated as invalid/missing.
    """

    if df.empty:
        return df

    out = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
        utc=True,
    )

    # Remove rows with invalid timestamps.
    out = out.dropna(
        subset=["timestamp"],
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    out["pm25"] = pd.to_numeric(
        out["pm25"],
        errors="coerce",
    )

    if "pm10" in out.columns:
        out["pm10"] = pd.to_numeric(
            out["pm10"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Invalid PM2.5 values
    # --------------------------------------------------------

    # PM2.5 values <= 0 are treated as invalid/missing
    # for this forecasting dataset. A zero PM2.5 value
    # would otherwise become AQI = 0 and could propagate
    # into future AQI targets.
    out.loc[
        out["pm25"] <= 0,
        "pm25",
    ] = pd.NA

    # --------------------------------------------------------
    # Remove duplicate observations
    # --------------------------------------------------------

    out = out.drop_duplicates(
        subset=["city", "timestamp"],
        keep="last",
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    out = out.sort_values(
        ["city", "timestamp"],
    ).reset_index(drop=True)

    return out


# ============================================================
# COVERAGE REPORT
# ============================================================

def print_coverage_report(
    df: pd.DataFrame,
) -> None:
    """
    Print a simple historical coverage report.
    """

    print()
    print("=" * 80)
    print("HISTORICAL DATA COVERAGE")
    print("=" * 80)

    if df.empty:
        print("No data available.")
        return

    for city in CITIES:

        city_df = df[
            df["city"] == city
        ]

        if city_df.empty:
            print(
                f"{city:12s} | NO DATA"
            )
            continue

        first_timestamp = city_df[
            "timestamp"
        ].min()

        last_timestamp = city_df[
            "timestamp"
        ].max()

        rows = len(city_df)

        valid_pm25 = city_df[
            "pm25"
        ].notna().sum()

        print(
            f"{city:12s} | "
            f"rows={rows:6d} | "
            f"PM2.5={valid_pm25:6d} | "
            f"{first_timestamp} -> {last_timestamp}"
        )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete historical backfill.
    """

    logger.info(
        "Starting AQI historical backfill."
    )

    logger.info(
        "Period: %s -> %s",
        START_DATE,
        END_DATE,
    )

    logger.info(
        "Cities: %s",
        ", ".join(CITIES),
    )

    client = OpenAQClient()

    all_frames = []

    # --------------------------------------------------------
    # Fetch each city
    # --------------------------------------------------------

    for city in CITIES:

        df = fetch_city(
            client=client,
            city=city,
        )

        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        raise RuntimeError(
            "Historical backfill produced no data."
        )

    # --------------------------------------------------------
    # Combine cities
    # --------------------------------------------------------

    historical_df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    historical_df = clean_historical_data(
        historical_df,
    )

    # --------------------------------------------------------
    # Calculate AQI
    # --------------------------------------------------------

    historical_df = add_aqi_from_pm25(
        historical_df,
    )

    # --------------------------------------------------------
    # Final ordering
    # --------------------------------------------------------

    historical_df = historical_df[
        [
            "timestamp",
            "city",
            "pm25",
            "pm10",
            "aqi",
            "aqi_category",
        ]
    ]

    historical_df = historical_df.sort_values(
        ["city", "timestamp"],
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    historical_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_coverage_report(
        historical_df,
    )

    print()
    print("=" * 80)
    print("BACKFILL COMPLETE")
    print("=" * 80)
    print(
        f"Total rows: {len(historical_df):,}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )
    print(
        f"Columns: {list(historical_df.columns)}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()