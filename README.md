# 🌍 Pearls AQI Predictor

An end-to-end serverless MLOps pipeline for 3-day Air Quality Index (AQI) forecasting for Pakistani cities — built as the 10Pearls SHINE Internship Program (Cohort 9) capstone project.

**🚀 Live Demo:** [aqi-predictor-iris.streamlit.app](https://aqi-predictor-iris.streamlit.app/)
**Documentation:** [https://1drv.ms/w/c/274dc0bb586af6ed/IQBOzQYvlFBWQ6ZgWEXhl8s6AdT_VDxk89faNhJBHIXOsLM?e=jJ1dHH )

---

## Overview

The Pearls AQI Predictor forecasts AQI 24h, 48h, and 72h ahead for Lahore, Karachi, and Islamabad using real-time PM2.5 sensor readings and weather data. The system runs entirely on free-tier serverless infrastructure with no manual intervention required after deployment.

---

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│      OpenAQ         │     │     Open-Meteo       │
│  PM2.5 / PM10 data  │     │  Historical weather  │
└──────────┬──────────┘     └──────────┬───────────┘
           │                           │
           └─────────────┬─────────────┘
                         │
                         ▼  (every hour via GitHub Actions)
           ┌─────────────────────────┐
           │    Feature Pipeline     │
           │  AQI + lag + rolling +  │
           │  time + weather feats   │
           └─────────────┬───────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │  Hopsworks Feature      │
           │  Store  (aqi_features   │
           │  v6 · 17,047 rows)      │
           └──────────┬──────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼  (daily 02:00 UTC)      ▼  (every hour)
  ┌──────────────┐         ┌───────────────────┐
  │  Training    │         │   Live Inference   │
  │  Pipeline    │         │   Pipeline         │
  └──────┬───────┘         └────────┬───────────┘
         │                          │
         ▼                          ▼
  Candidate RF models       Production RF models
         │                          │
         ▼                          ▼
   MAE comparison          predictions/latest.json
    │          │                    │
  better?   keep                    ▼
    │       current         GitHub repository
    ▼                               │
 Hopsworks                          ▼
 Model Registry              Streamlit Cloud
                                    │
                                    ▼
                          Public AQI Dashboard
```

---

## Key Features

- **Real-time forecasting** — Live PM2.5 readings from OpenAQ sensors
- **3-day horizon** — Separate models for +24h, +48h, +72h
- **Automated feature store** — Hourly updates to Hopsworks Feature Store
- **Auto model promotion** — Daily retraining compares candidate vs production MAE
- **Graceful degradation** — Unavailable cities are reported, not fabricated
- **Hazardous AQI alerts** — Dashboard warns when AQI > 150/200/300
- **100% serverless** — GitHub Actions + Hopsworks Free Tier + Streamlit Cloud

---

## Data Sources

| Source | Data | Usage |
|---|---|---|
| [OpenAQ v3](https://openaq.org) | Hourly PM2.5, PM10 | AQI calculation, lag features |
| [Open-Meteo](https://open-meteo.com) | Temperature, humidity, pressure, wind, rain | Weather features |

**Coverage:** August 2024 → present · Cities: Lahore, Karachi, Islamabad

---

## Feature Engineering

37 features engineered from raw PM2.5 + weather:

| Category | Features |
|---|---|
| Raw | `pm25`, `aqi`, `temperature`, `humidity`, `pressure`, `wind_speed`, `wind_direction`, `rain_1h`, `rain_3h` |
| Time | `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend` |
| Cyclical | `hour_sin/cos`, `day_sin/cos`, `month_sin/cos` |
| AQI lags | `aqi_lag_1h`, `aqi_lag_3h`, `aqi_lag_6h`, `aqi_lag_12h`, `aqi_lag_24h` |
| Rolling | `aqi_roll_mean_3h/6h/24h`, `aqi_roll_std_24h` |
| Change | `aqi_change_1h`, `aqi_change_rate` |

All lag and rolling features are **timestamp-aware** — large data gaps do not silently corrupt features.

---

## ML Models

Six model families evaluated on a chronological 80/20 train-test split:

| Model | +24h MAE | +24h R² | +48h MAE | +48h R² | +72h MAE | +72h R² |
|---|---|---|---|---|---|---|
| Naive Baseline | 32.08 | 0.432 | — | — | 34.77 | 0.309 |
| Ridge Regression | 26.47 | 0.626 | 29.38 | 0.537 | 31.68 | 0.448 |
| Random Forest | 27.26 | 0.598 | 30.29 | 0.513 | 31.44 | 0.482 |
| **Tuned Random Forest** | **26.38** | **0.627** | **29.13** | **0.549** | **30.90** | **0.495** |
| Tuned XGBoost | 27.01 | 0.601 | 30.84 | 0.497 | 31.87 | 0.471 |
| LSTM | 29.72 | 0.288 | 32.03 | 0.194 | 35.26 | 0.038 |
| Ensemble (Ridge+RF+XGB) | 27.29 | 0.647 | 29.57 | 0.599 | 31.07 | 0.541 |

**Production model:** Tuned Random Forest (300 trees, trained on 100% of clean data)

SHAP feature importance confirms `pm25`, `aqi`, `aqi_lag_24h`, and `aqi_roll_mean_24h` are the top predictors across all horizons.

---

## Hopsworks Integration

| Component | Status | Details |
|---|---|---|
| Feature Group | ✅ | `aqi_features` v6 · 17,047 rows · 37 columns |
| Feature Pipeline | ✅ | Hourly via GitHub Actions |
| Model Registry | ✅ | `aqi_random_forest_24h/48h/72h` v1 |
| Auto Promotion | ✅ | Candidate replaces production if MAE improves |

---

## Automated Pipelines

### Hourly Feature Pipeline (`.github/workflows/feature_pipeline.yml`)
```
Fetch last 30 days OpenAQ + Open-Meteo
→ Merge by city + timestamp (30-min tolerance)
→ Compute all 37 features
→ Upload fresh rows to Hopsworks Feature Store
```

### Hourly Live Inference (`.github/workflows/live_inference.yml`)
```
Download production models from Hopsworks Registry
→ Fetch latest OpenAQ readings
→ Build inference features
→ Generate +24h/+48h/+72h forecasts
→ Save predictions/latest.json
→ Commit to GitHub (Streamlit reads this)
```

### Daily Training Pipeline (`.github/workflows/training_pipeline.yml`)
```
Read Feature Store → train candidate RF models
→ evaluate on chronological test set
→ compare candidate MAE vs production MAE
→ promote and register if candidate is better
```

---

## Project Structure

```
AQI-Predictor/
├── src/
│   ├── data/
│   │   └── historical_backfill.py      # 2-year PM2.5 backfill
│   ├── features/
│   │   ├── openaq_client.py            # OpenAQ v3 API client
│   │   ├── openmeteo_client.py         # Open-Meteo API client
│   │   ├── aqi_calculator.py           # PM2.5 → AQI (EPA formula)
│   │   ├── feature_engineering.py      # All 37 features
│   │   └── hopsworks_upload.py         # Bulk historical upload
│   ├── models/
│   │   ├── baseline.py                 # Naive baseline
│   │   ├── ridge.py                    # Ridge regression
│   │   ├── random_forest.py            # Random Forest
│   │   ├── tune_random_forest.py       # RF hyperparameter tuning
│   │   ├── xgboost.py                  # XGBoost
│   │   ├── tune_xgboost.py             # XGBoost tuning
│   │   ├── deep_learning.py            # LSTM (TensorFlow)
│   │   ├── improved_ensemble.py        # Weighted ensemble
│   │   ├── feature_importance.py       # SHAP analysis
│   │   ├── error_analysis.py           # City/regime error breakdown
│   │   ├── final_predictor.py          # Production RF pipeline
│   │   ├── live_predictor.py           # Live inference
│   │   ├── register_models.py          # Hopsworks registration
│   │   └── download_production_models.py
│   ├── pipelines/
│   │   ├── feature_pipeline.py         # Hourly feature automation
│   │   └── training_pipeline.py        # Daily retraining + promotion
│   └── utils/
│       ├── logger.py
│       └── config.py
├── app/
│   ├── streamlit_app.py                # Streamlit dashboard
│   └── flask_api.py                    # Flask REST API
├── predictions/
│   └── latest.json                     # Live predictions (auto-updated)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── tests/
│   ├── test_live_predictor.py
│   ├── test_feature_pipeline.py
│   ├── test_hopsworks_upload.py
│   └── test_models.py
├── .github/workflows/
│   ├── feature_pipeline.yml            # Hourly feature updates
│   ├── live_inference.yml              # Hourly predictions
│   ├── training_pipeline.yml           # Daily retraining
│   └── hopsworks_upload.yml            # One-time bulk upload
├── requirements.txt
├── requirements-hopsworks.txt
└── requirements-inference.txt
```

---

## Installation

```bash
git clone https://github.com/Shahrukh-aidev/AQI-Predictor.git
cd AQI-Predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
```
OPENAQ_API_KEY=your_openaq_api_key
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=AQI_Predicto
```

---

## Local Execution

```bash
# Run historical backfill (one-time)
python -m src.data.historical_backfill

# Run hourly feature pipeline
python -m src.pipelines.feature_pipeline

# Run daily training pipeline
python -m src.pipelines.training_pipeline

# Run live inference
python -m src.models.live_predictor

# Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## GitHub Actions Secrets

| Secret | Description |
|---|---|
| `HOPSWORKS_API_KEY` | Hopsworks project API key |
| `HOPSWORKS_PROJECT` | Hopsworks project name (`AQI_Predicto`) |
| `OPENAQ_API_KEY` | OpenAQ v3 API key |

---

## Tests

```bash
pytest -q --ignore=tests/test_openaq.py
```

Note: `test_openaq.py` calls the live OpenAQ location search API which occasionally returns HTTP 500. All production inference uses sensor-level endpoints which are unaffected.

---

## Known Limitations

- **Karachi and Islamabad** — OpenAQ PM2.5 sensors for these cities are intermittently offline. The system reports them as unavailable rather than fabricating forecasts.
- **LSTM underperforms** — With 17,047 training rows and CPU-only inference on Windows, LSTM (R²=0.038 at +72h) is significantly outperformed by tree-based models. Not used in production.
- **Lahore error gap** — Mean Absolute Error for Lahore (33.8) is significantly higher than Islamabad (22.1) due to extreme AQI spikes (400+) during winter smog season that the model cannot anticipate from lag features alone.

---

## Technology Stack

Python · scikit-learn · TensorFlow/Keras · XGBoost · pandas · NumPy · SHAP · Hopsworks · Streamlit · Flask · GitHub Actions · OpenAQ API · Open-Meteo API

---

## Author

**Shahrukh** — BS Computer Science (AI & Robotics), Sukkur IBA University  
10Pearls SHINE Internship Program · Cohort 9 · Data Science Track  
MLSA · Fiverr AI/ML Developer

---

## License

MIT License — see [LICENSE](LICENSE)
