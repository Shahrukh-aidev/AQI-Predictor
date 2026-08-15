"""
AQI Predictor - Streamlit Dashboard

Reads the latest live prediction results from:

    predictions/latest.json

The dashboard does NOT run the ML models itself.
GitHub Actions/live_predictor.py generates the JSON,
and Streamlit displays the latest results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Configuration
# ============================================================

PREDICTIONS_FILE = Path(
    "predictions/latest.json"
)

APP_TITLE = "AQI Predictor"

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
]


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .aqi-card {
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #ddd;
            background: #ffffff;
            margin-bottom: 15px;
        }

        .status-success {
            color: #15803d;
            font-weight: 700;
        }

        .status-warning {
            color: #b45309;
            font-weight: 700;
        }

        .status-error {
            color: #b91c1c;
            font-weight: 700;
        }

        .forecast-value {
            font-size: 30px;
            font-weight: 700;
        }

        .small-text {
            color: #666;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AQI classification
# ============================================================

def classify_aqi(aqi: float) -> tuple[str, str]:
    """
    Classify AQI using standard AQI ranges.

    Returns:
        category, status
    """

    if aqi <= 50:
        return "Good", "success"

    if aqi <= 100:
        return "Moderate", "success"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "warning"

    if aqi <= 200:
        return "Unhealthy", "error"

    if aqi <= 300:
        return "Very Unhealthy", "error"

    return "Hazardous", "error"


def category_icon(category: str) -> str:
    """Return an icon for an AQI category."""

    mapping = {
        "Good": "🟢",
        "Moderate": "🟡",
        "Unhealthy for Sensitive Groups": "🟠",
        "Unhealthy": "🔴",
        "Very Unhealthy": "🟣",
        "Hazardous": "☠️",
    }

    return mapping.get(
        category,
        "⚪",
    )


# ============================================================
# Load predictions
# ============================================================

@st.cache_data(ttl=60)
def load_predictions() -> dict:
    """Load latest prediction JSON."""

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{PREDICTIONS_FILE}"
        )

    with PREDICTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Prediction file must contain a JSON object."
        )

    if "forecasts" not in data:
        raise ValueError(
            "Prediction file is missing 'forecasts'."
        )

    return data


# ============================================================
# Timestamp formatting
# ============================================================

def format_timestamp(value: str) -> str:
    """Convert ISO timestamp into readable UTC text."""

    try:
        timestamp = datetime.fromisoformat(
            value
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        timestamp = timestamp.astimezone(
            timezone.utc
        )

        return timestamp.strftime(
            "%d %b %Y, %H:%M UTC"
        )

    except Exception:
        return value


# ============================================================
# Find city
# ============================================================

def get_city_result(
    forecasts: list[dict],
    city: str,
) -> dict | None:
    """Return the JSON result for a city."""

    for result in forecasts:

        if result.get("city") == city:
            return result

    return None


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🌍 AQI Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Live Air Quality Forecast for Pakistani Cities"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Load data
# ============================================================

try:

    data = load_predictions()

except FileNotFoundError:

    st.error(
        "No prediction file found."
    )

    st.info(
        "Run the live inference pipeline first:"
    )

    st.code(
        "python -m src.models.live_predictor",
        language="powershell",
    )

    st.stop()

except Exception as exc:

    st.error(
        f"Unable to load predictions: {exc}"
    )

    st.stop()


forecasts = data.get(
    "forecasts",
    [],
)

generated_at = data.get(
    "generated_at"
)

model_name = data.get(
    "model",
    "Unknown model",
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Dashboard")

    selected_city = st.selectbox(
        "Select city",
        CITIES,
    )

    st.divider()

    st.subheader(
        "System Information"
    )

    st.write(
        f"**Model:** {model_name}"
    )

    if generated_at:
        st.write(
            "**Generated:** "
            f"{format_timestamp(generated_at)}"
        )

    st.caption(
        "Data source: OpenAQ + OpenWeather"
    )

    if st.button(
        "🔄 Refresh data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# Selected city
# ============================================================

city_result = get_city_result(
    forecasts,
    selected_city,
)

if city_result is None:

    st.error(
        f"No result exists for {selected_city}."
    )

    st.stop()


status = city_result.get(
    "status",
    "unknown",
)


# ============================================================
# Unavailable city
# ============================================================

if status != "success":

    st.warning(
        f"Live AQI data is currently unavailable "
        f"for {selected_city}."
    )

    reason = city_result.get(
        "reason",
        "No reason supplied.",
    )

    st.error(
        f"Reason: {reason}"
    )

    st.markdown(
        """
        ### What this means

        The system does not generate a fabricated forecast when
        the required live PM2.5 data source is unavailable.

        This keeps the prediction pipeline trustworthy.
        """
    )

    st.stop()


# ============================================================
# Successful city
# ============================================================

current_aqi = float(
    city_result["current_aqi"]
)

latest_observation = city_result[
    "latest_observation"
]

predictions = city_result.get(
    "predictions",
    {},
)

category, category_status = classify_aqi(
    current_aqi
)

icon = category_icon(
    category
)


# ============================================================
# Current AQI
# ============================================================

st.subheader(
    f"{selected_city} — Current Air Quality"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Current AQI",
        f"{current_aqi:.0f}",
    )

with col2:

    st.metric(
        "Category",
        category,
    )

with col3:

    st.metric(
        "Live Status",
        "Available",
    )

st.markdown(
    f"""
    <div class="aqi-card">
        <div style="font-size: 20px;">
            {icon} <strong>{category}</strong>
        </div>
        <div class="small-text">
            Latest real OpenAQ observation:
            {format_timestamp(latest_observation)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Forecast cards
# ============================================================

st.subheader(
    "🔮 AQI Forecast"
)

forecast_columns = st.columns(3)

horizons = [
    ("24", "+24h", "Tomorrow"),
    ("48", "+48h", "Day 2"),
    ("72", "+72h", "Day 3"),
]

for column, (key, horizon_label, day_label) in zip(
    forecast_columns,
    horizons,
):

    prediction_data = predictions.get(
        key
    )

    if prediction_data is None:

        with column:

            st.warning(
                f"{horizon_label} forecast unavailable."
            )

        continue

    predicted_aqi = float(
        prediction_data[
            "predicted_aqi"
        ]
    )

    forecast_timestamp = prediction_data[
        "forecast_timestamp"
    ]

    forecast_category, _ = classify_aqi(
        predicted_aqi
    )

    forecast_icon = category_icon(
        forecast_category
    )

    with column:

        st.markdown(
            '<div class="aqi-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"### {day_label}"
        )

        st.markdown(
            f'<div class="forecast-value">'
            f"{predicted_aqi:.1f}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.write(
            f"{horizon_label} AQI"
        )

        st.write(
            f"{forecast_icon} "
            f"{forecast_category}"
        )

        st.caption(
            format_timestamp(
                forecast_timestamp
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# Forecast table
# ============================================================

st.subheader(
    "📊 Forecast Details"
)

table_rows = []

for key, horizon_label, day_label in horizons:

    prediction_data = predictions.get(
        key
    )

    if prediction_data is None:
        continue

    predicted_aqi = float(
        prediction_data[
            "predicted_aqi"
        ]
    )

    forecast_category, _ = classify_aqi(
        predicted_aqi
    )

    table_rows.append(
        {
            "Forecast": day_label,
            "Horizon": horizon_label,
            "AQI": round(
                predicted_aqi,
                2,
            ),
            "Category": forecast_category,
            "Timestamp (UTC)": format_timestamp(
                prediction_data[
                    "forecast_timestamp"
                ]
            ),
        }
    )

if table_rows:

    forecast_df = pd.DataFrame(
        table_rows
    )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Hazard alert
# ============================================================

max_forecast_aqi = current_aqi

for prediction_data in predictions.values():

    try:

        predicted = float(
            prediction_data[
                "predicted_aqi"
            ]
        )

        max_forecast_aqi = max(
            max_forecast_aqi,
            predicted,
        )

    except (
        TypeError,
        ValueError,
        KeyError,
    ):
        continue


st.subheader(
    "⚠️ Air Quality Alert"
)

if max_forecast_aqi > 300:

    st.error(
        "☠️ Hazardous AQI detected. "
        "Immediate health precautions are recommended."
    )

elif max_forecast_aqi > 200:

    st.error(
        "🟣 Very unhealthy AQI is expected. "
        "Reduce outdoor exposure."
    )

elif max_forecast_aqi > 150:

    st.error(
        "🔴 Unhealthy AQI detected or forecast. "
        "Consider limiting prolonged outdoor activity."
    )

elif max_forecast_aqi > 100:

    st.warning(
        "🟠 AQI may be unhealthy for sensitive groups. "
        "Sensitive individuals should consider reducing "
        "prolonged outdoor activity."
    )

elif max_forecast_aqi > 50:

    st.info(
        "🟡 Air quality is currently in the moderate range."
    )

else:

    st.success(
        "🟢 Air quality is currently good."
    )


# ============================================================
# City availability overview
# ============================================================

st.subheader(
    "🏙️ City Availability"
)

availability_columns = st.columns(
    len(CITIES)
)

for column, city in zip(
    availability_columns,
    CITIES,
):

    result = get_city_result(
        forecasts,
        city,
    )

    with column:

        if (
            result
            and result.get("status") == "success"
        ):

            st.success(
                f"✅ {city}"
            )

            st.caption(
                f"AQI: {result['current_aqi']}"
            )

        else:

            st.warning(
                f"⚠️ {city}"
            )

            if result:
                st.caption(
                    result.get(
                        "reason",
                        "Data unavailable.",
                    )
                )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "AQI Predictor • Live forecasts generated by the "
    "production Random Forest inference pipeline."
)