import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 🚀 Page Configuration
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter vehicle details below to estimate the market price.</p>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        data = pd.read_csv('/Users/praveenan/Desktop/Kaggle competition/Dateset/train.csv')
    except FileNotFoundError:
        st.error("🚨 Dataset not found! Check the path to train.csv.")
        return None, None

    # Check required columns exist
    required_cols = ['model', 'motor_type', 'wheel', 'color', 'type', 'status', 'year', 'motor_volume', 'running', 'price']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"🚨 Missing required columns in dataset: {missing_cols}")
        return None, None

    # Clean running column
    def clean_running(x):
        try:
            return int(str(x).replace('km', '').replace('KM', '').replace(' ', '').replace(',', '').strip())
        except:
            return 0

    data['running'] = data['running'].apply(clean_running)

    # Filter out outliers for price
    data = data[data['price'] < 50000]

    data['vehicle_age'] = 2025 - data['year']

    cat_cols = ['model', 'motor_type', 'wheel', 'color', 'type', 'status']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    features = ['year', 'vehicle_age', 'motor_volume', 'model', 'motor_type', 'running', 'wheel', 'type']
    X = data[features]
    y = data['price']

    model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X, y)

    return model, encoders

model, encoders = load_model()

if model is None:
    st.stop()

# 2️⃣ UI Inputs
year = st.slider('Year', 1990, 2025, 2018)
motor_volume = st.number_input('Motor Volume (L)', min_value=0.2, max_value=5.0, value=2.0, step=0.1)
running = st.number_input('Mileage (km)', min_value=0, max_value=500000, value=50000, step=1000)

model_input = st.selectbox('Model', list(encoders['model'].classes_))
motor_type_input = st.selectbox('Motor Type', list(encoders['motor_type'].classes_))
wheel_input = st.selectbox('Wheel', list(encoders['wheel'].classes_))
color_input = st.selectbox('Color', list(encoders['color'].classes_))
type_input = st.selectbox('Type', list(encoders['type'].classes_))

# 3️⃣ Prepare Input for Model
input_data = {
    'year': year,
    'vehicle_age': 2025 - year,
    'motor_volume': motor_volume,
    'model': encoders['model'].transform([model_input])[0],
    'motor_type': encoders['motor_type'].transform([motor_type_input])[0],
    'running': running,
    'wheel': encoders['wheel'].transform([wheel_input])[0],
    'type': encoders['type'].transform([type_input])[0]
}

input_df = pd.DataFrame([input_data])

# 4️⃣ Prediction
if st.button('Predict Price 💰'):
    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: **${int(predicted_price):,}**")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: small;'>Developed for CIS6005 Project | Praveenan Pirabaharan </p>", unsafe_allow_html=True)
