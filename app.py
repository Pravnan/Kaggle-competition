import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ────────────────────────────────────────────────────────────
# 🚀 Page config & Title
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #2C3E50;'>🚗 Car Price Prediction App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Fill in the details and hit Predict to estimate the price.</p>",
    unsafe_allow_html=True
)
# ────────────────────────────────────────────────────────────

@st.cache_data
def load_and_prep_data():
    # 1️⃣ Load (relative)  
    df = pd.read_csv("Dateset/train.csv")
    
    # 2️⃣ Clean mileage  
    def clean_running(x):
        try:
            return int(str(x).lower().replace("km","").replace(",","").strip())
        except:
            return 0
    df["running"] = df["running"].apply(clean_running)
    
    # 3️⃣ Filter out crazy prices  
    df = df[df["price"] < 50000]
    
    # 4️⃣ Vehicle age  
    df["vehicle_age"] = 2025 - df["year"]
    
    # 5️⃣ Encode categoricals  
    cat_cols = ["model","motor_type","wheel","color","type","status"]
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    
    return df, encoders

# Load once
data, encoders = load_and_prep_data()

# ────────────────────────────────────────────────────────────
# 2️⃣ Sidebar / Input widgets
st.sidebar.header("Vehicle Details")
year        = st.sidebar.slider("Year", 1990, 2025, 2018)
motor_vol   = st.sidebar.number_input("Motor Volume (L)", 0.2, 5.0, 2.0, 0.1)
running     = st.sidebar.number_input("Mileage (km)", 0, 500_000, 50_000, 1_000)

model_in    = st.sidebar.selectbox("Model",        encoders["model"].classes_)
motor_type  = st.sidebar.selectbox("Motor Type",   encoders["motor_type"].classes_)
wheel_cat   = st.sidebar.selectbox("Wheel",        encoders["wheel"].classes_)
color_cat   = st.sidebar.selectbox("Color",        encoders["color"].classes_)
type_cat    = st.sidebar.selectbox("Type",         encoders["type"].classes_)

# Build input row
input_dict = {
    "year": year,
    "vehicle_age": 2025 - year,
    "motor_volume": motor_vol,
    "running": running,
    "model":      encoders["model"].transform([model_in])[0],
    "motor_type": encoders["motor_type"].transform([motor_type])[0],
    "wheel":      encoders["wheel"].transform([wheel_cat])[0],
    "type":       encoders["type"].transform([type_cat])[0]
}
input_df = pd.DataFrame([input_dict])

# ────────────────────────────────────────────────────────────
# 3️⃣ Train & Predict on button click
if st.button("Predict Price 💰"):
    # Train on full data each time (for demo)
    X = data[["year","vehicle_age","motor_volume","running",
              "model","motor_type","wheel","type"]]
    y = data["price"]
    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_split=5, random_state=42
    )
    model.fit(X, y)
    
    pred = model.predict(input_df)[0]
    st.success(f"**Estimated Price:** ${int(pred):,}")

# ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size:small;'>"
    "Developed by Praveenan Pirabaharan"
    "</p>",
    unsafe_allow_html=True
)
