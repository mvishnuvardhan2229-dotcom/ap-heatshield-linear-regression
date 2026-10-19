import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

st.set_page_config(page_title="AP HeatShield | Vijayawada Temp Estimator", layout="wide")

@st.cache_resource
def load_assets():
    bundle = joblib.load("heatshield_lr_model.pkl")
    hist_df = pd.read_csv("vijayawada_heatshield_clean.csv", parse_dates=["Date"])
    return bundle["model"], bundle["features"], hist_df

model, feature_names, df = load_assets()

st.title("☀️ AP HeatShield: Peak Temperature & Heatwave Risk Estimator")
st.caption("Station: Vijayawada (Gannavaram IMD/NOAA Archive) • Model: Multiple Linear Regression")

col_input, col_result = st.columns([1.1, 1.3], gap="large")

with col_input:
    st.subheader("Today's Station Observations")
    
    # Default values pulled from the latest row in our dataset
    latest = df.iloc[-1]
    
    c1, c2 = st.columns(2)
    with c1:
        tm_today = st.slider("Today's Max Temp - TM (°C)", 24.0, 48.0, float(latest["TM"]), 0.1)
        t_today = st.slider("Today's Mean Temp - T (°C)", 20.0, 42.0, float(latest["T"]), 0.1)
        tm_min = st.slider("Today's Min Temp - Tm (°C)", 15.0, 35.0, float(latest["Tm"]), 0.1)
        humidity = st.slider("Relative Humidity - H (%)", 15, 100, int(latest["H"]), 1)
        
    with c2:
        tm_lag1 = st.number_input("Yesterday's Max Temp (°C)", 24.0, 48.0, float(latest["tm_lag1"]), 0.1)
        tm_roll3 = st.number_input("Last 3-Day Avg Max Temp (°C)", 24.0, 48.0, float(latest["tm_roll3"]), 0.1)
        tm_roll7 = st.number_input("Last 7-Day Avg Max Temp (°C)", 24.0, 48.0, float(latest["tm_roll7"]), 0.1)
        wind = st.slider("Wind Speed - V (km/h)", 0.0, 35.0, float(latest["V"]), 0.5)
        rain = st.number_input("Rainfall - PP (mm)", 0.0, 200.0, 0.0, 0.5)
        selected_date = st.date_input("Observation Date", date.today())

with col_result:
    st.subheader("Next-Day Forecast & Heat Advisory")
    
    # Calculate seasonal cyclical features from selected date
    doy = selected_date.timetuple().tm_yday
    s_sin = np.sin(2 * np.pi * doy / 365.25)
    s_cos = np.cos(2 * np.pi * doy / 365.25)
    
    input_row = pd.DataFrame([{
        'TM': tm_today,
        'T': t_today,
        'Tm': tm_min,
        'H': humidity,
        'PP': rain,
        'VV': float(latest["VV"]),
        'V': wind,
        'tm_lag1': tm_lag1,
        'tm_roll3': tm_roll3,
        'tm_roll7': tm_roll7,
        'h_roll3': float(humidity),
        'season_sin': s_sin,
        'season_cos': s_cos
    }])[feature_names]
    
    pred_tm = float(model.predict(input_row)[0])
    
    # Determine IMD Heatwave Risk Category
    if pred_tm >= 43.0:
        risk_label = "🔴 SEVERE HEATWAVE ALERT (Extreme Outdoor Risk)"
        advice = "Avoid outdoor exposure between 11 AM and 4 PM. High risk of heatstroke for field workers."
    elif pred_tm >= 40.0:
        risk_label = "🟠 HEATWAVE WARNING (High Caution)"
        advice = "Temperatures crossing 40°C threshold. Increase crop irrigation & hydration breaks."
    elif pred_tm >= 37.0:
        risk_label = "🟡 WARM / MODERATE HEAT STRESS"
        advice = "Normal summer heat conditions. Maintain regular hydration."
    else:
        risk_label = "🟢 COMFORTABLE / NORMAL RANGE"
        advice = "No heatwave anomaly detected for tomorrow."

    st.metric(
        label="Predicted Tomorrow's Peak Temperature (TM)",
        value=f"{pred_tm:.1f} °C",
        delta=f"{pred_tm - tm_today:+.1f} °C vs Today"
    )
    
    st.info(f"**Status:** {risk_label}\n\n**Actionable Advisory:** {advice}")
    
    st.markdown("#### Recent Historical Peak Temperature Trend (Vijayawada)")
    chart_data = df.tail(180).set_index("Date")[["TM", "tm_roll7"]]
    chart_data.columns = ["Daily Max Temp (°C)", "7-Day Rolling Mean (°C)"]
    st.line_chart(chart_data)
