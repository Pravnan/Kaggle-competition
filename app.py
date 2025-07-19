import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Title
st.title("Kaggle Competition Model Trainer")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('Dateset/train.csv')  # Relative path (make sure train.csv is inside Dateset/)

data = load_data()

st.subheader("Dataset Preview")
st.write(data.head())

# Preprocessing (you can customize this based on your dataset)
st.subheader("Preprocessing")

# Drop rows with missing values
data_clean = data.dropna()
st.write(f"Cleaned Data Shape: {data_clean.shape}")

# Select features and target
features = st.multiselect("Select feature columns:", data_clean.columns[:-1], default=data_clean.columns[:-1])
target = st.selectbox("Select target column:", data_clean.columns)

if features and target:
    X = data_clean[features]
    y = data_clean[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Show feature importances
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=features)
    st.bar_chart(importances.sort_values(ascending=False))

else:
    st.warning("Please select both features and target column to train the model.")
