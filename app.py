import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="🔥 Wildfire Risk Predictor", layout="wide")
st.title("🔥 Wildfire Risk Predictor")
st.markdown("Predict fire-prone zones using weather and terrain data")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 1000
    X = np.column_stack([
        np.random.uniform(15, 45, n_samples),  # temp
        np.random.uniform(20, 100, n_samples),  # humidity
        np.random.uniform(0, 30, n_samples),    # wind speed
        np.random.uniform(0, 10, n_samples),    # precipitation
        np.random.uniform(0, 3000, n_samples),  # elevation
        np.random.randint(0, 5, n_samples),     # vegetation (0-4)
        np.random.uniform(0, 45, n_samples),    # slope
    ])
    y = ((X[:, 0] > 30) & (X[:, 1] < 40) & (X[:, 2] > 15) & (X[:, 4] < 2000)).astype(int)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📍 Input Weather")
    temp = st.slider("Temperature (°C)", 10, 50, 28)
    humidity = st.slider("Humidity (%)", 10, 100, 35)
    wind = st.slider("Wind Speed (km/h)", 0, 50, 20)
    precip = st.slider("Precipitation (mm)", 0, 50, 2)
    elevation = st.slider("Elevation (m)", 0, 3000, 1200)
    veg = st.selectbox("Vegetation", ["Grassland", "Shrub", "Pine", "Mixed", "Urban"])
    slope = st.slider("Slope (°)", 0, 45, 15)

with col2:
    if st.button("🔥 Predict Risk"):
        veg_map = {"Grassland": 0, "Shrub": 1, "Pine": 2, "Mixed": 3, "Urban": 4}
        features = np.array([[temp, humidity, wind, precip, elevation, veg_map[veg], slope]])
        risk_prob = model.predict_proba(features)[0][1] * 100
        
        st.markdown("### 📊 Risk Assessment")
        if risk_prob > 70:
            st.error(f"🚨 **CRITICAL RISK: {risk_prob:.1f}%**")
            st.write("Immediate fire danger. Alert communities. Deploy resources.")
        elif risk_prob > 40:
            st.warning(f"⚠️ **HIGH RISK: {risk_prob:.1f}%**")
            st.write("Elevated danger. Monitor closely. Prepare evacuation plans.")
        else:
            st.success(f"✅ **LOW RISK: {risk_prob:.1f}%**")
            st.write("Safe conditions. Routine monitoring sufficient.")
        
        st.progress(min(risk_prob / 100, 1.0))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{temp}°C", "High risk if >30")
        col2.metric("Humidity", f"{humidity}%", "Low risk if >50")
        col3.metric("Wind", f"{wind} km/h", "High risk if >15")

st.markdown("---")
st.subheader("🗺️ Risk Map (Simulated)")
fig, ax = plt.subplots(figsize=(10, 8))
risk_map = np.random.rand(20, 20) * 100
im = ax.imshow(risk_map, cmap="RdYlGn_r", vmin=0, vmax=100)
ax.set_title("Wildfire Risk Map (0-100)")
plt.colorbar(im, ax=ax, label="Risk %")
st.pyplot(fig)
