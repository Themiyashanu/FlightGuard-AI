from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from pipeline import MODELS_DIR, ROOT_DIR as PROJECT_ROOT, predict_risk  # noqa: E402


st.set_page_config(page_title="Aviation Risk Dashboard", layout="wide")
st.title("Aviation Risk Prediction Dashboard")
st.caption("Interactive risk assessment using trained ML model")

metrics_path = MODELS_DIR / "metrics.json"
model_path = MODELS_DIR / "aviation_risk_model.pkl"
processed_path = PROJECT_ROOT / "data" / "processed" / "processed_aviation_data.csv"

if not model_path.exists():
    st.error("Model not found. Please run: python src/train_model.py")
    st.stop()

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", metrics.get("best_model", "N/A"))
    col2.metric("Accuracy", f"{metrics.get('best_accuracy', 0):.3f}")
    col3.metric("AUC", f"{metrics.get('best_auc', 0):.3f}")

st.subheader("Predict Flight Risk")
left, right = st.columns(2)

with left:
    year = st.number_input("Year", min_value=1908, max_value=2035, value=2020)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    aboard = st.number_input("People Aboard", min_value=0, max_value=1000, value=120)
    summary_length = st.slider("Incident Summary Length", min_value=0, max_value=1000, value=120)
    aircraft_category = st.selectbox(
        "Aircraft Category", ["Commercial", "Military", "Helicopter", "General Aviation", "Other", "Unknown"]
    )

with right:
    country = st.text_input("Country", value="United States")
    broad_phase = st.text_input("Flight Phase", value="Unknown")
    is_military = st.selectbox("Military Operator", [0, 1], index=0)
    is_commercial = st.selectbox("Commercial Operator", [0, 1], index=1)
    weather_mentioned = st.selectbox("Adverse Weather Mentioned", [0, 1], index=0)
    mechanical_failure = st.selectbox("Mechanical Failure Mentioned", [0, 1], index=0)
    pilot_error = st.selectbox("Pilot Error Mentioned", [0, 1], index=0)

if st.button("Predict Risk", type="primary"):
    payload = {
        "year": year,
        "month": month,
        "aboard": aboard,
        "summary_length": summary_length,
        "is_military": is_military,
        "is_commercial": is_commercial,
        "weather_mentioned": weather_mentioned,
        "mechanical_failure": mechanical_failure,
        "pilot_error": pilot_error,
        "aircraft_category": aircraft_category,
        "country": country,
        "broad_phase": broad_phase,
    }
    pred = predict_risk(payload)
    st.success(f"Predicted Risk: {pred['risk_level']} ({pred['probability']:.2%})")

st.subheader("Historical Data Insights")
if processed_path.exists():
    df = pd.read_csv(processed_path)
    if {"year", "fatalities"}.issubset(df.columns):
        yearly = df.groupby("year", as_index=False)["fatalities"].sum()
        fig = px.line(yearly, x="year", y="fatalities", title="Total Fatalities by Year")
        st.plotly_chart(fig, use_container_width=True)

    if {"aircraft_category", "high_risk"}.issubset(df.columns):
        risk_rate = (
            df.groupby("aircraft_category", as_index=False)["high_risk"]
            .mean()
            .sort_values("high_risk", ascending=False)
        )
        fig2 = px.bar(risk_rate, x="aircraft_category", y="high_risk", title="High-Risk Rate by Aircraft Category")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Processed dataset not found yet. Run training to generate processed outputs.")
