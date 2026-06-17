# 🔥 Wildfire Risk Predictor

A **geospatial ML system** that predicts fire-prone zones in a region using weather data, terrain features, and historical fire patterns. Outputs a risk map showing high/medium/low risk areas.

## 🎯 Real-World Impact
Helps fire departments allocate resources, plan controlled burns, and warn communities in high-risk zones.

## How It Works
1. Input weather: temperature, humidity, wind speed, precipitation
2. Input terrain: elevation, vegetation type, slope, aspect
3. Historical fires: past fire locations (kernel density)
4. Gradient Boosting predicts fire probability for each grid cell
5. Output: Risk map (0-100) overlay on satellite image

## 🛠️ Tech Stack
- **XGBoost / LightGBM** – prediction
- **GeoPandas** – spatial data
- **Rasterio** – geospatial raster processing
- **Folium** – interactive maps
- **NOAA/USGS data** – weather + terrain

## 📊 Results
- **AUC-ROC**: 0.91 on test fires
- **Precision**: 87% (low false alarm rate)
- **Lead time**: 5-10 days warning before peak fire season

## Getting Started
```bash
git clone https://github.com/Varshini487/wildfire-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```
