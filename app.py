import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ────────────────────────────────────────────────────────────
# 🚀 Page Configuration
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>Car Price Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Enter the details below and click <b style='color:#27ae60;'>Predict</b> to estimate your car's price.</p>",
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("Dataset/Car_prediciton_update_final.csv")

    def clean_running(x):
        try:
            return int(str(x).lower().replace("km","").replace(",","").strip())
        except:
            return 0
    df["running"] = df["running"].apply(clean_running)
    df = df[df["price"] < 50000]
    df["vehicle_age"] = 2025 - df["year"]

    cat_cols = ["model","motor_type","wheel","color","type","status"]
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    return df, encoders

# Load Data
data, encoders = load_and_prep_data()

# ────────────────────────────────────────────────────────────
st.markdown("### 🚘 Vehicle Details")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        year = st.slider("Year of Manufacture", 1990, 2025, 2018)
        motor_vol = st.number_input("Motor Volume (L)", 0.2, 5.0, 2.0, step=0.1)
        running = st.number_input("Mileage (in km)", 0, 500_000, 50_000, step=1000)
    with col2:
        model_in = st.selectbox("Model", encoders["model"].classes_)
        motor_type = st.selectbox("Motor Type", encoders["motor_type"].classes_)
        wheel_cat = st.selectbox("Wheel Drive", encoders["wheel"].classes_)
        color_cat = st.selectbox("Color", encoders["color"].classes_)
        type_cat = st.selectbox("Vehicle Type", encoders["type"].classes_)

# Prepare Input Data
input_dict = {
    "year": year,
    "vehicle_age": 2025 - year,
    "motor_volume": motor_vol,
    "running": running,
    "model": encoders["model"].transform([model_in])[0],
    "motor_type": encoders["motor_type"].transform([motor_type])[0],
    "wheel": encoders["wheel"].transform([wheel_cat])[0],
    "type": encoders["type"].transform([type_cat])[0],
}
input_df = pd.DataFrame([input_dict])

# ────────────────────────────────────────────────────────────
# 🔮 Prediction
st.markdown("---")
if st.button("💰 Predict Price", type="primary"):
    X = data[["year","vehicle_age","motor_volume","running",
              "model","motor_type","wheel","type"]]
    y = data["price"]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_split=5, random_state=42
    )
    model.fit(X, y)

    pred = model.predict(input_df)[0]
    st.markdown(
        f"<div style='text-align: center; background-color: #dff0d8; padding: 1rem; border-radius: 10px;'>"
        f"<h2 style='color: #2e7d32;'>Estimated Price: ${int(pred):,}</h2>"
        f"</div>",
        unsafe_allow_html=True
    )

# ────────────────────────────────────────────────────────────
# 📦 Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size:small; color: gray;'>"
    "Crafted with ❤️ by Praveenan Pirabaharan"
    "</p>",
    unsafe_allow_html=True
)
