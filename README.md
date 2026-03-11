## 👨‍💻 About the Author

**Pichikala Kali Bhargav Sai**
2023BCSE07AED278
CSE AIML - C

This project was built as part of a hands-on ML workshop covering the complete pipeline — from raw data all the way to a deployed, production-ready web application.

[![GitHub](https://img.shields.io/badge/GitHub-bhargav--pichikala-181717?style=flat-square&logo=github)](https://github.com/bhargav-pichikala)


# 🌫️ Air Quality Index (AQI) Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red?style=flat-square)

> A machine learning web application that predicts the **Air Quality Index (AQI)** of major Indian cities based on real-time pollutant measurements — trained on 29,500+ daily records, powered by XGBoost, and deployed live on Streamlit Community Cloud.

---

## 🔗 Important Links

| | Link |
|---|---|
| 🚀 **Live Streamlit App** | [AQI Predictor · Streamlit](https://aqi-predictor-6dceqbqcbi9bfewzuyxttc.streamlit.app/) |
| 📓 **Training Notebook** | [AQI_Prediction_Final.ipynb · Google Colab](https://colab.research.google.com/drive/1PKeHwSg4GG__EYvSB6QAHv7taum6YCUF#scrollTo=c-req) |

---

## 📌 Problem Statement

Air pollution is one of the most pressing environmental and public health challenges in India. Millions of people are exposed to hazardous air quality daily, yet real-time AQI prediction tools remain limited. This project addresses that gap by building a **regression-based machine learning pipeline** that:

- Takes pollutant readings (PM2.5, PM10, NO2, SO2, CO, O3) as input
- Predicts the AQI value for a given Indian city and date
- Classifies the predicted AQI into health categories (Good to Severe)
- Presents results in an interactive, easy-to-use web dashboard

This kind of system is directly applicable to:
- 🏙️ Smart city air monitoring infrastructure
- 🏥 Public health early warning systems
- 🌱 Environmental policy and research
- 📊 Government and civic pollution dashboards

---

## 🌐 Live Demo

Click the badge below to open the app:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bhargav-pichikala-aqi-predictor-streamlit-app.streamlit.app)

The app has 5 pages:

| Page | What it does |
|---|---|
| 🔮 **Predict AQI** | Enter pollutant values and get an instant AQI prediction with category |
| 📊 **EDA Dashboard** | Distribution plots, correlation heatmap, trend charts, city-level view |
| 🏙️ **City Rankings** | Compare all cities by average AQI with a color-coded table and chart |
| 🎛️ **What-If Simulator** | Drag sliders to simulate pollution scenarios and see live AQI updates |
| 📋 **About** | Project details, model info, and tech stack |

---

## 📓 Training Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhargav-pichikala/aqi-predictor/blob/main/AQI_Prediction_Final.ipynb)

The full ML pipeline was built in Google Colab. The notebook covers:

| Step | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Import all libraries |
| 3 | Load `city_day.csv` dataset |
| 4 | Data preprocessing — handle nulls, filter top 10 cities |
| 5 | Feature engineering — month, season, day of week, AQI lag feature |
| 6 | EDA — 4 visualizations (distribution, heatmap, trend, city bar chart) |
| 7 | Train 4 models — Linear Regression, Random Forest, Gradient Boosting, XGBoost |
| 8 | Evaluate and compare all models using RMSE, MAE and R² |
| 9 | Save model, scaler, clean data and config to disk |
| 10 | Test interactively with **Gradio** inside Colab (free 72hr public link) |
| 11 | Write production **Streamlit app** to disk using `%%writefile` |
| 12 | Write `requirements.txt` for deployment |
| 13 | Deploy to Streamlit Community Cloud |

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) |
| **Author** | Rohan Rao |
| **Records** | ~29,500 city-day observations |
| **Cities** | 26 Indian cities |
| **Date Range** | 2015 – 2020 |
| **Target Variable** | AQI (Air Quality Index) |

---

## 🔬 Features Used

| Feature | Type | Description |
|---|---|---|
| PM2.5 | Pollutant | Fine particulate matter (µg/m³) |
| PM10 | Pollutant | Coarse particulate matter (µg/m³) |
| NO2 | Pollutant | Nitrogen Dioxide (µg/m³) |
| SO2 | Pollutant | Sulphur Dioxide (µg/m³) |
| CO | Pollutant | Carbon Monoxide (mg/m³) |
| O3 | Pollutant | Ozone (µg/m³) |
| City_enc | Engineered | Label-encoded city name |
| month | Engineered | Extracted from Date |
| dayofweek | Engineered | Extracted from Date (0=Monday) |
| season | Engineered | Derived from month (1=Winter, 2=Spring, 3=Summer, 4=Autumn) |
| AQI_lag1 | Time-series | Previous day's AQI per city |

---

## 🧠 Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~61 | ~35 | ~0.86 |
| Random Forest | ~49 | ~26 | ~0.91 |
| Gradient Boosting | ~48 | ~26 | ~0.91 |
| **XGBoost (Deployed)** ✅ | **~49** | **~25** | **~0.91** |


## 🔢 AQI Categories (CPCB Standard)

| AQI Range | Category | Health Implication |
|---|---|---|
| 0 – 50 | 😊 Good | Minimal impact |
| 51 – 100 | 🙂 Satisfactory | Minor breathing discomfort for sensitive people |
| 101 – 200 | 😐 Moderate | Breathing discomfort for asthma patients |
| 201 – 300 | 😷 Poor | Breathing discomfort for most on prolonged exposure |
| 301 – 400 | 🤢 Very Poor | Respiratory illness on prolonged exposure |
| 401+ | ☠️ Severe | Affects healthy people; serious risk for sensitive groups |

---

## 🗂️ Repository Structure

```
bhargav-pichikala/aqi-predictor/
│
├── streamlit_app.py         # Main Streamlit web application
├── aqi_model.pkl            # Trained XGBoost model (serialized)
├── scaler.pkl               # StandardScaler for input features
├── aqi_clean.csv            # Cleaned and processed dataset
├── app_config.json          # Feature list + city label encoding map
├── requirements.txt         # All Python dependencies
├── AQI_Prediction_Final.ipynb  # Full training notebook (optional)
└── README.md                # Project documentation
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Web App | Streamlit |
| Colab Prototyping | Gradio |
| Model Serialization | Joblib |
| Training Environment | Google Colab |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---


