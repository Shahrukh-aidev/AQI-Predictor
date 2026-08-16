"""
Pearls AQI Predictor — Lightweight Streamlit Dashboard.

Reads:
    predictions/latest.json

The dashboard only displays the latest prediction output.
It does not run ML inference itself.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Configuration
# ============================================================

PREDICTIONS_FILE = Path("predictions/latest.json")

APP_TITLE = "Pearls AQI Predictor"

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
]

HORIZONS = [
    ("24", "+24h", "Tomorrow"),
    ("48", "+48h", "Day 2"),
    ("72", "+72h", "Day 3"),
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
# Helpers
# ============================================================

def render_html(content: str) -> None:
    """Render trusted application HTML."""
    st.html(content)


def safe_text(value: object) -> str:
    """Escape dynamic text before inserting into HTML."""
    return html.escape(str(value))


def format_timestamp(value: str | None) -> str:
    """Format an ISO timestamp as readable UTC text."""

    if not value:
        return "Unknown"

    try:
        timestamp = datetime.fromisoformat(value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        timestamp = timestamp.astimezone(
            timezone.utc
        )

        return timestamp.strftime(
            "%d %b %Y • %H:%M UTC"
        )

    except (TypeError, ValueError):
        return str(value)


def get_city_result(
    forecasts: list[dict],
    city: str,
) -> dict | None:
    """Return the prediction record for a city."""

    for result in forecasts:
        if result.get("city") == city:
            return result

    return None


# ============================================================
# AQI helpers
# ============================================================

def classify_aqi(
    aqi: float,
) -> tuple[str, str, str]:
    """
    Return:
        category,
        display color,
        icon
    """

    if aqi <= 50:
        return "Good", "#22c55e", "🟢"

    if aqi <= 100:
        return "Moderate", "#eab308", "🟡"

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "#f97316",
            "🟠",
        )

    if aqi <= 200:
        return "Unhealthy", "#ef4444", "🔴"

    if aqi <= 300:
        return "Very Unhealthy", "#a855f7", "🟣"

    return "Hazardous", "#7e0023", "☠️"


def health_message(aqi: float) -> str:
    """Return general AQI guidance."""

    if aqi <= 50:
        return (
            "Air quality is good. Outdoor activities are "
            "generally suitable."
        )

    if aqi <= 100:
        return (
            "Air quality is moderate. Unusually sensitive "
            "people may notice effects."
        )

    if aqi <= 150:
        return (
            "Sensitive groups should consider reducing "
            "prolonged outdoor activity."
        )

    if aqi <= 200:
        return (
            "Everyone may begin to experience health effects. "
            "Consider reducing outdoor exposure."
        )

    if aqi <= 300:
        return (
            "Health risk is increased for everyone. "
            "Reduce unnecessary outdoor exposure."
        )

    return (
        "Hazardous air quality. Avoid unnecessary outdoor "
        "exposure and follow local health guidance."
    )


def trend_text(
    current: float,
    predicted: float,
) -> tuple[str, str]:
    """Return trend label and CSS class."""

    delta = predicted - current

    if delta > 5:
        return f"▲ +{delta:.1f}", "trend-worse"

    if delta < -5:
        return f"▼ {delta:.1f}", "trend-better"

    return f"→ {delta:+.1f}", "trend-neutral"


def alert_message(category: str) -> str:
    """Return alert text based on AQI category."""

    messages = {
        "Good": (
            "Air quality is good. Outdoor activities are "
            "generally suitable."
        ),
        "Moderate": (
            "Air quality is moderate. Unusually sensitive "
            "people may want to limit prolonged exertion."
        ),
        "Unhealthy for Sensitive Groups": (
            "Sensitive groups should consider reducing "
            "prolonged outdoor activity."
        ),
        "Unhealthy": (
            "Unhealthy air quality is present or forecast. "
            "Consider limiting prolonged outdoor activity."
        ),
        "Very Unhealthy": (
            "Very unhealthy air quality is present or forecast. "
            "Avoid unnecessary outdoor exposure."
        ),
        "Hazardous": (
            "Hazardous air quality is present or forecast. "
            "Avoid unnecessary outdoor exposure."
        ),
    }

    return messages.get(
        category,
        "AQI information is currently unavailable.",
    )


# ============================================================
# CSS
# ============================================================

render_html(
    """
    <style>

        /* ====================================================
           Global
           ==================================================== */

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    circle at 0% 0%,
                    rgba(59,130,246,.12),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 100% 100%,
                    rgba(16,185,129,.08),
                    transparent 30%
                ),
                #07111f;
        }

        .main .block-container {
            max-width: 1250px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] {
            background: #091725;
            border-right: 1px solid rgba(255,255,255,.08);
        }

        /* ====================================================
           Sidebar
           ==================================================== */

        [data-testid="stSidebar"] * {
            color: #e7eef7;
        }

        .brand {
            font-size: 1.25rem;
            font-weight: 800;
            color: white;
            margin-bottom: 2px;
        }

        .brand-subtitle {
            color: rgba(255,255,255,.42);
            font-size: .75rem;
            line-height: 1.4;
        }

        .sidebar-label {
            color: rgba(255,255,255,.34);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin: 18px 0 8px;
        }

        /* ====================================================
           Hero
           ==================================================== */

        .hero {
            background:
                linear-gradient(
                    135deg,
                    #123f68 0%,
                    #0b243d 100%
                );

            border:
                1px solid rgba(255,255,255,.09);

            border-radius: 22px;

            padding: 28px;

            box-shadow:
                0 18px 45px rgba(0,0,0,.20);
        }

        .hero-label {
            color: rgba(255,255,255,.52);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .hero-city {
            color: white;
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1;
            margin-top: 8px;
        }

        .hero-subtitle {
            color: rgba(255,255,255,.60);
            font-size: .92rem;
            line-height: 1.5;
            margin-top: 8px;
            max-width: 620px;
        }

        .hero-badge {
            display: inline-block;
            margin-top: 14px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.12);
            font-size: .76rem;
            font-weight: 700;
        }

        .hero-aqi-label {
            color: rgba(255,255,255,.44);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
        }

        .hero-aqi {
            color: white;
            font-size: 4.5rem;
            font-weight: 900;
            line-height: .9;
            margin-top: 8px;
        }

        /* ====================================================
           Cards
           ==================================================== */

        .card {
            background: rgba(14,28,45,.88);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 18px;
            padding: 20px;
            min-height: 128px;
        }

        .card-label {
            color: rgba(255,255,255,.42);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .card-value {
            color: white;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 7px;
            line-height: 1.2;
        }

        .card-note {
            color: rgba(255,255,255,.38);
            font-size: .72rem;
            margin-top: 6px;
            line-height: 1.4;
        }

        /* ====================================================
           Forecast cards
           ==================================================== */

        .forecast-card {
            background: rgba(14,28,45,.88);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 18px;
            padding: 20px;
            min-height: 190px;
        }

        .forecast-day {
            color: rgba(255,255,255,.42);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .forecast-aqi {
            color: white;
            font-size: 2.8rem;
            font-weight: 900;
            margin-top: 10px;
            line-height: 1;
        }

        .forecast-category {
            color: rgba(255,255,255,.67);
            font-size: .80rem;
            margin-top: 6px;
        }

        .forecast-time {
            color: rgba(255,255,255,.31);
            font-size: .68rem;
            margin-top: 10px;
        }

        .trend-better {
            color: #86efac;
            font-size: .75rem;
            font-weight: 700;
        }

        .trend-worse {
            color: #fca5a5;
            font-size: .75rem;
            font-weight: 700;
        }

        .trend-neutral {
            color: #cbd5e1;
            font-size: .75rem;
            font-weight: 700;
        }

        /* ====================================================
           Alerts
           ==================================================== */

        .alert {
            background: rgba(255,255,255,.05);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 14px;
            padding: 15px 18px;
            color: rgba(255,255,255,.78);
            font-size: .83rem;
            line-height: 1.5;
        }

        /* ====================================================
           City status
           ==================================================== */

        .city-card {
            background: rgba(14,28,45,.88);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 16px;
            padding: 17px;
            min-height: 145px;
        }

        .city-name {
            color: rgba(255,255,255,.43);
            font-size: .67rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .city-aqi {
            color: white;
            font-size: 1.9rem;
            font-weight: 800;
            margin-top: 6px;
        }

        .live {
            color: #86efac;
            font-size: .72rem;
            font-weight: 700;
        }

        .offline {
            color: #fca5a5;
            font-size: .72rem;
            font-weight: 700;
        }

        .city-note {
            color: rgba(255,255,255,.35);
            font-size: .69rem;
            line-height: 1.4;
            margin-top: 7px;
        }

        /* ====================================================
           Footer
           ==================================================== */

        .footer {
            border-top: 1px solid rgba(255,255,255,.07);
            margin-top: 40px;
            padding-top: 18px;
            color: rgba(255,255,255,.25);
            text-align: center;
            font-size: .68rem;
            line-height: 1.6;
        }

    </style>
    """
)


# ============================================================
# Load predictions
# ============================================================

@st.cache_data(ttl=60)
def load_predictions() -> dict:
    """Load and validate the prediction file."""

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
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

    if not isinstance(
        data.get("forecasts"),
        list,
    ):
        raise ValueError(
            "Prediction file is missing 'forecasts'."
        )

    return data


try:
    data = load_predictions()

except FileNotFoundError:
    st.error(
        "No live prediction file is available yet."
    )
    st.info(
        "Run: python -m src.models.live_predictor"
    )
    st.stop()

except Exception as exc:
    st.error(
        f"Unable to load prediction data: {exc}"
    )
    st.stop()


forecasts = data.get(
    "forecasts",
    [],
)

generated_at = data.get(
    "generated_at",
)

model_name = data.get(
    "model",
    "Tuned Random Forest",
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="brand">
            🌍 Pearls AQI
        </div>

        <div class="brand-subtitle">
            Live air-quality forecasting for Pakistan
        </div>

        <div class="sidebar-label">
            City
        </div>
        """
    )

    selected_city = st.selectbox(
        "Select city",
        CITIES,
        label_visibility="collapsed",
    )

    st.divider()

    render_html(
        """
        <div class="sidebar-label">
            System
        </div>
        """
    )

    st.write(
        f"**Model:** {model_name}"
    )

    st.write(
        f"**Updated:** "
        f"{format_timestamp(generated_at)}"
    )

    st.write(
        "**Sources:** OpenAQ · Open-Meteo"
    )

    st.write("")

    if st.button(
        "↻ Refresh dashboard",
        width="stretch",
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


# ============================================================
# Unavailable city
# ============================================================

if (
    city_result is None
    or city_result.get("status") != "success"
):

    reason = (
        city_result.get(
            "reason",
            "Live AQI data is unavailable.",
        )
        if city_result
        else
        "No prediction record exists for this city."
    )

    render_html(
        f"""
        <div style="margin-top:18px;">

            <div class="hero">

                <div class="hero-label">
                    PEARLS AQI · LIVE MONITORING
                </div>

                <div class="hero-city">
                    {safe_text(selected_city)}
                </div>

                <div class="hero-subtitle">
                    Live AQI data is currently unavailable.
                </div>

                <div class="hero-badge">
                    ⚠️ Data unavailable
                </div>

            </div>

        </div>
        """
    )

    st.write("")

    render_html(
        f"""
        <div class="card">

            <div class="card-label">
                Why no forecast?
            </div>

            <div style="
                color:rgba(255,255,255,.70);
                font-size:.90rem;
                margin-top:8px;
                line-height:1.6;
            ">
                {safe_text(reason)}
            </div>

            <div class="card-note">
                Pearls AQI does not fabricate predictions when
                the required live sensor data is unavailable.
            </div>

        </div>
        """
    )

    st.stop()


# ============================================================
# Current city values
# ============================================================

try:
    current_aqi = float(
        city_result.get(
            "current_aqi",
            0,
        )
    )
except (TypeError, ValueError):
    current_aqi = 0.0


latest_observation = city_result.get(
    "latest_observation"
)

predictions = city_result.get(
    "predictions",
    {},
)


current_category, current_color, current_icon = (
    classify_aqi(current_aqi)
)


# ============================================================
# Hero
# ============================================================

hero_left, hero_right = st.columns(
    [2.2, 1],
    gap="large",
)

with hero_left:

    render_html(
        f"""
        <div class="hero">

            <div class="hero-label">
                PEARLS AQI · LIVE MONITORING
            </div>

            <div class="hero-city">
                {safe_text(selected_city)}
            </div>

            <div class="hero-subtitle">
                3-day AQI forecasting powered by the
                production {safe_text(model_name)} model.
            </div>

            <div class="hero-badge"
                style="color:{current_color};">

                {safe_text(current_icon)}
                {safe_text(current_category)}

            </div>

            <div style="
                color:rgba(255,255,255,.35);
                font-size:.69rem;
                margin-top:13px;
            ">
                Latest observation:
                {safe_text(
                    format_timestamp(latest_observation)
                )}
            </div>

        </div>
        """
    )


with hero_right:

    render_html(
        f"""
        <div class="hero"
             style="text-align:center;">

            <div class="hero-aqi-label">
                CURRENT AQI
            </div>

            <div class="hero-aqi">
                {current_aqi:.0f}
            </div>

            <div style="
                color:{current_color};
                font-size:.76rem;
                font-weight:700;
                margin-top:8px;
            ">
                {safe_text(current_category)}
            </div>

        </div>
        """
    )


# ============================================================
# Summary cards
# ============================================================

summary_columns = st.columns(4)

summary_items = [
    (
        "Status",
        "● LIVE",
        "Current source available",
    ),
    (
        "Category",
        f"{current_icon} {current_category}",
        "AQI classification",
    ),
    (
        "Model",
        "Random Forest",
        "Production model",
    ),
    (
        "Forecast",
        "72h",
        "24h / 48h / 72h",
    ),
]


for column, (
    label,
    value,
    note,
) in zip(
    summary_columns,
    summary_items,
):

    with column:

        render_html(
            f"""
            <div class="card">

                <div class="card-label">
                    {safe_text(label)}
                </div>

                <div class="card-value">
                    {safe_text(value)}
                </div>

                <div class="card-note">
                    {safe_text(note)}
                </div>

            </div>
            """
        )


# ============================================================
# Health guidance
# ============================================================

st.write("")

render_html(
    f"""
    <div class="alert">

        <strong>
            {safe_text(current_icon)}
            Health guidance:
        </strong>

        {safe_text(
            health_message(current_aqi)
        )}

    </div>
    """
)


# ============================================================
# Forecast cards
# ============================================================

st.markdown(
    "### 🔮 3-Day Forecast"
)

forecast_columns = st.columns(
    3,
    gap="medium",
)


for column, (
    key,
    horizon,
    day_label,
) in zip(
    forecast_columns,
    HORIZONS,
):

    prediction = predictions.get(
        key
    )

    with column:

        if not isinstance(
            prediction,
            dict,
        ):

            render_html(
                f"""
                <div class="forecast-card">

                    <div class="forecast-day">
                        {safe_text(day_label)}
                        ·
                        {safe_text(horizon)}
                    </div>

                    <div class="forecast-aqi">
                        —
                    </div>

                    <div class="forecast-category">
                        Forecast unavailable
                    </div>

                </div>
                """
            )

            continue

        try:

            predicted_aqi = float(
                prediction[
                    "predicted_aqi"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            render_html(
                f"""
                <div class="forecast-card">

                    <div class="forecast-day">
                        {safe_text(day_label)}
                        ·
                        {safe_text(horizon)}
                    </div>

                    <div class="forecast-aqi">
                        —
                    </div>

                    <div class="forecast-category">
                        Invalid forecast
                    </div>

                </div>
                """
            )

            continue

        forecast_category, forecast_color, forecast_icon = (
            classify_aqi(
                predicted_aqi
            )
        )

        delta, trend_class = trend_text(
            current_aqi,
            predicted_aqi,
        )

        render_html(
            f"""
            <div class="forecast-card">

                <div class="forecast-day">
                    {safe_text(day_label)}
                    ·
                    {safe_text(horizon)}
                </div>

                <div class="forecast-aqi">
                    {predicted_aqi:.1f}
                </div>

                <div class="forecast-category"
                     style="color:{forecast_color};">

                    {safe_text(forecast_icon)}
                    {safe_text(forecast_category)}

                </div>

                <div style="
                    margin-top:10px;
                ">
                    <span class="{trend_class}">
                        {safe_text(delta)}
                    </span>

                    <span style="
                        color:rgba(255,255,255,.30);
                        font-size:.70rem;
                        margin-left:4px;
                    ">
                        vs current
                    </span>
                </div>

                <div class="forecast-time">
                    {safe_text(
                        format_timestamp(
                            prediction.get(
                                "forecast_timestamp"
                            )
                        )
                    )}
                </div>

            </div>
            """
        )


# ============================================================
# Forecast chart
# ============================================================

st.markdown(
    "### 📈 AQI Forecast Trend"
)

labels = [
    "Now",
    "+24h",
    "+48h",
    "+72h",
]

values: list[float | None] = [
    current_aqi
]

for key in [
    "24",
    "48",
    "72",
]:

    prediction = predictions.get(
        key
    )

    if not isinstance(
        prediction,
        dict,
    ):

        values.append(None)
        continue

    try:
        values.append(
            float(
                prediction[
                    "predicted_aqi"
                ]
            )
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        values.append(None)


figure = go.Figure()

figure.add_trace(
    go.Scatter(
        x=labels,
        y=values,
        mode="lines+markers+text",
        text=[
            (
                f"{value:.1f}"
                if value is not None
                else "—"
            )
            for value in values
        ],
        textposition="top center",
        line=dict(
            color="#60a5fa",
            width=3,
        ),
        marker=dict(
            color="#dbeafe",
            size=9,
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "AQI: %{y:.1f}"
            "<extra></extra>"
        ),
    )
)

for level, label, color in [
    (
        50,
        "Good",
        "rgba(34,197,94,.45)",
    ),
    (
        100,
        "Moderate",
        "rgba(234,179,8,.45)",
    ),
    (
        150,
        "Unhealthy",
        "rgba(249,115,22,.45)",
    ),
]:

    figure.add_hline(
        y=level,
        line_dash="dash",
        line_color=color,
        annotation_text=label,
        annotation_position="right",
        annotation_font_size=9,
    )


valid_values = [
    value
    for value in values
    if value is not None
]

maximum_value = max(
    valid_values,
    default=100,
)


figure.update_layout(
    height=330,
    margin=dict(
        l=10,
        r=50,
        t=20,
        b=10,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="rgba(255,255,255,.70)",
    ),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
    ),
    yaxis=dict(
        title="AQI",
        range=[
            0,
            max(
                120,
                maximum_value * 1.25,
            ),
        ],
        gridcolor="rgba(255,255,255,.06)",
        zeroline=False,
    ),
)

st.plotly_chart(
    figure,
    width="stretch",
    config={
        "displayModeBar": False,
        "displaylogo": False,
        "responsive": True,
    },
)


# ============================================================
# Forecast details
# ============================================================

with st.expander(
    "📊 Forecast details",
):

    rows = []

    for (
        key,
        horizon,
        day_label,
    ) in HORIZONS:

        prediction = predictions.get(
            key
        )

        if not isinstance(
            prediction,
            dict,
        ):
            continue

        try:

            predicted_aqi = float(
                prediction[
                    "predicted_aqi"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

        forecast_category, _, forecast_icon = (
            classify_aqi(
                predicted_aqi
            )
        )

        rows.append(
            {
                "Forecast": day_label,
                "Horizon": horizon,
                "AQI": round(
                    predicted_aqi,
                    1,
                ),
                "Category": (
                    f"{forecast_icon} "
                    f"{forecast_category}"
                ),
                "Forecast Time (UTC)": (
                    format_timestamp(
                        prediction.get(
                            "forecast_timestamp"
                        )
                    )
                ),
            }
        )

    if rows:

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No forecast details are available."
        )


# ============================================================
# Forecast alert
# ============================================================

max_forecast_aqi = current_aqi

for prediction in predictions.values():

    if not isinstance(
        prediction,
        dict,
    ):
        continue

    try:

        max_forecast_aqi = max(
            max_forecast_aqi,
            float(
                prediction[
                    "predicted_aqi"
                ]
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        continue


max_category, max_color, max_icon = classify_aqi(
    max_forecast_aqi
)

st.markdown(
    "### 🚨 Air Quality Alert"
)

render_html(
    f"""
    <div class="alert">

        <strong style="
            color:{max_color};
        ">
            {safe_text(max_icon)}
            Maximum forecast AQI:
            {max_forecast_aqi:.1f}
        </strong>

        <br>

        {safe_text(
            alert_message(max_category)
        )}

    </div>
    """
)


# ============================================================
# City availability
# ============================================================

st.markdown(
    "### 🌐 City Availability"
)

city_columns = st.columns(
    len(CITIES),
    gap="medium",
)


for column, city in zip(
    city_columns,
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

            try:

                city_aqi = float(
                    result.get(
                        "current_aqi",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                city_aqi = 0.0

            city_category, city_color, city_icon = (
                classify_aqi(
                    city_aqi
                )
            )

            render_html(
                f"""
                <div class="city-card">

                    <div class="city-name">
                        {safe_text(city)}
                    </div>

                    <div class="city-aqi">
                        {safe_text(city_icon)}
                        {city_aqi:.0f}
                    </div>

                    <div class="live">
                        ● LIVE
                    </div>

                    <div class="city-note"
                         style="color:{city_color};">
                        {safe_text(city_category)}
                    </div>

                </div>
                """
            )

        else:

            reason = (
                result.get(
                    "reason",
                    "Data unavailable.",
                )
                if result
                else
                "No prediction record."
            )

            render_html(
                f"""
                <div class="city-card">

                    <div class="city-name">
                        {safe_text(city)}
                    </div>

                    <div class="city-aqi">
                        —
                    </div>

                    <div class="offline">
                        ● OFFLINE
                    </div>

                    <div class="city-note">
                        {safe_text(reason)}
                    </div>

                </div>
                """
            )


# ============================================================
# System information
# ============================================================

with st.expander(
    "⚙️ System information",
):

    info1, info2 = st.columns(2)

    with info1:

        st.write(
            f"**Production model:** {model_name}"
        )

        st.write(
            "**Forecast horizons:** +24h / +48h / +72h"
        )

        st.write(
            "**Feature store:** Hopsworks"
        )

    with info2:

        st.write(
            "**Automation:** GitHub Actions"
        )

        st.write(
            "**Inference source:** OpenAQ + weather features"
        )

        st.write(
            f"**Latest pipeline output:** "
            f"{format_timestamp(generated_at)}"
        )


# ============================================================
# Footer
# ============================================================

render_html(
    """
    <div class="footer">

        Pearls AQI Predictor ·
        10Pearls SHINE Internship Cohort 9

        <br>

        OpenAQ · Open-Meteo · Hopsworks ·
        GitHub Actions · Streamlit

    </div>
    """
)