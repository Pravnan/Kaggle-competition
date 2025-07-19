import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────
# 🎨 Page Configuration & Custom CSS
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 15px 0;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: bold !important;
        color: #2c3e50 !important;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #7f8c8d;
        font-style: italic;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# 📊 Data Loading & Preparation
# ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prep_data():
    """Load and preprocess the car dataset"""
    try:
        # Load dataset
        df = pd.read_csv("Dateset/train.csv")
        
        # Clean mileage column
        def clean_running(x):
            try:
                return int(str(x).lower().replace("km","").replace(",","").strip())
            except:
                return 0
        
        df["running"] = df["running"].apply(clean_running)
        
        # Filter outliers
        df = df[df["price"] < 50000]
        df = df[df["price"] > 500]  # Remove unrealistically low prices
        
        # Calculate vehicle age
        df["vehicle_age"] = 2025 - df["year"]
        
        # Encode categorical variables
        cat_cols = ["model", "motor_type", "wheel", "color", "type", "status"]
        encoders = {}
        
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        return df, encoders
        
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Dateset/train.csv' exists.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# ────────────────────────────────────────────────────────────
# 🎯 Main Application
# ────────────────────────────────────────────────────────────

# Header Section
st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get instant price estimates for your vehicle using advanced machine learning</p>', unsafe_allow_html=True)

# Load data
with st.spinner("Loading data..."):
    data, encoders = load_and_prep_data()

if data is None or encoders is None:
    st.stop()

# ────────────────────────────────────────────────────────────
# 📋 Input Section
# ────────────────────────────────────────────────────────────

st.markdown("## 📝 Vehicle Information")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Basic Details")
    
    year = st.slider(
        "📅 Manufacturing Year",
        min_value=1990,
        max_value=2025,
        value=2018,
        help="Select the year your car was manufactured"
    )
    
    motor_vol = st.number_input(
        "🔧 Engine Volume (Liters)",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.1,
        help="Engine displacement in liters"
    )
    
    running = st.number_input(
        "📏 Mileage (KM)",
        min_value=0,
        max_value=500000,
        value=50000,
        step=5000,
        help="Total distance traveled by the vehicle"
    )

with col2:
    st.markdown("### Vehicle Specifications")
    
    model_options = list(encoders["model"].classes_) if "model" in encoders else ["Unknown"]
    model_in = st.selectbox(
        "🚙 Car Model",
        options=model_options,
        help="Select your car model"
    )
    
    motor_type_options = list(encoders["motor_type"].classes_) if "motor_type" in encoders else ["Unknown"]
    motor_type = st.selectbox(
        "⚡ Engine Type",
        options=motor_type_options,
        help="Type of engine (Petrol, Diesel, etc.)"
    )
    
    wheel_options = list(encoders["wheel"].classes_) if "wheel" in encoders else ["Unknown"]
    wheel_cat = st.selectbox(
        "🎯 Drive Type",
        options=wheel_options,
        help="Wheel drive configuration"
    )

# Additional specifications in a third row
col3, col4, col5 = st.columns(3)

with col3:
    color_options = list(encoders["color"].classes_) if "color" in encoders else ["Unknown"]
    color_cat = st.selectbox(
        "🎨 Color",
        options=color_options,
        help="Vehicle color"
    )

with col4:
    type_options = list(encoders["type"].classes_) if "type" in encoders else ["Unknown"]
    type_cat = st.selectbox(
        "🚗 Vehicle Type",
        options=type_options,
        help="Category of vehicle"
    )

with col5:
    # Calculate and display vehicle age
    vehicle_age = 2025 - year
    st.metric("📊 Vehicle Age", f"{vehicle_age} years")

# ────────────────────────────────────────────────────────────
# 🔮 Prediction Section
# ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 💰 Price Prediction")

# Create prediction button with better styling
col_center = st.columns([1, 2, 1])
with col_center[1]:
    predict_button = st.button(
        "🚀 Predict Price",
        use_container_width=True,
        type="primary"
    )

if predict_button:
    with st.spinner("🤖 Analyzing vehicle data..."):
        try:
            # Prepare input data
            input_dict = {
                "year": year,
                "vehicle_age": vehicle_age,
                "motor_volume": motor_vol,
                "running": running,
            }
            
            # Add encoded categorical variables
            if "model" in encoders:
                input_dict["model"] = encoders["model"].transform([model_in])[0]
            if "motor_type" in encoders:
                input_dict["motor_type"] = encoders["motor_type"].transform([motor_type])[0]
            if "wheel" in encoders:
                input_dict["wheel"] = encoders["wheel"].transform([wheel_cat])[0]
            if "type" in encoders:
                input_dict["type"] = encoders["type"].transform([type_cat])[0]
            
            input_df = pd.DataFrame([input_dict])
            
            # Prepare features for training
            feature_cols = ["year", "vehicle_age", "motor_volume", "running"]
            if "model" in encoders:
                feature_cols.append("model")
            if "motor_type" in encoders:
                feature_cols.append("motor_type")
            if "wheel" in encoders:
                feature_cols.append("wheel")
            if "type" in encoders:
                feature_cols.append("type")
            
            # Train model
            X = data[feature_cols]
            y = data["price"]
            
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display results
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2 style="color: white; margin: 0;">💰 Estimated Price</h2>
                    <h1 style="color: white; margin: 10px 0; font-size: 3rem;">${int(prediction):,}</h1>
                    <p style="color: white; opacity: 0.9; margin: 0;">Based on current market analysis</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Additional insights
            col_insight1, col_insight2, col_insight3 = st.columns(3)
            
            with col_insight1:
                price_per_year = prediction / max(vehicle_age, 1)
                st.metric(
                    "💡 Price per Year",
                    f"${int(price_per_year):,}",
                    help="Average value retained per year of age"
                )
            
            with col_insight2:
                avg_price = data["price"].mean()
                diff_from_avg = ((prediction - avg_price) / avg_price) * 100
                st.metric(
                    "📊 vs Market Average",
                    f"{diff_from_avg:+.1f}%",
                    help="Comparison with average market price"
                )
            
            with col_insight3:
                depreciation_rate = (1 - (prediction / (prediction + vehicle_age * 2000))) * 100
                st.metric(
                    "📉 Depreciation Factor",
                    f"{depreciation_rate:.1f}%",
                    help="Estimated depreciation consideration"
                )
            
            # Price range estimation
            st.markdown("### 📈 Price Range Analysis")
            
            # Generate multiple predictions for confidence interval
            predictions = []
            for i in range(10):
                model_temp = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=i*42
                )
                model_temp.fit(X, y)
                pred_temp = model_temp.predict(input_df)[0]
                predictions.append(pred_temp)
            
            min_pred = min(predictions)
            max_pred = max(predictions)
            
            col_range1, col_range2 = st.columns(2)
            with col_range1:
                st.info(f"**Minimum Estimate:** ${int(min_pred):,}")
            with col_range2:
                st.info(f"**Maximum Estimate:** ${int(max_pred):,}")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check your input values and try again.")

# ────────────────────────────────────────────────────────────
# 📊 Dataset Overview
# ────────────────────────────────────────────────────────────

with st.expander("📊 Dataset Overview", expanded=False):
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("📋 Total Records", f"{len(data):,}")
    
    with col_stats2:
        st.metric("💰 Average Price", f"${int(data['price'].mean()):,}")
    
    with col_stats3:
        st.metric("🔝 Max Price", f"${int(data['price'].max()):,}")
    
    with col_stats4:
        st.metric("🔻 Min Price", f"${int(data['price'].min()):,}")
    
    # Show sample data
    st.markdown("#### Sample Data")
    sample_data = data.head(5)[['year', 'motor_volume', 'running', 'price']]
    st.dataframe(sample_data, use_container_width=True)

# ────────────────────────────────────────────────────────────
# 📞 Footer
# ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p>🔬 <strong>Developed by Praveenan Pirabaharan</strong></p>
        <p>Powered by Machine Learning • Built with Streamlit • Data-driven insights</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────
# 💡 Help Section
# ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
    1. **Fill Vehicle Details**: Enter your car's specifications in the form
    2. **Click Predict**: Hit the prediction button to get instant estimates
    3. **View Results**: Get detailed price analysis and market insights
    
    **Tips for Better Accuracy:**
    - Ensure all details are accurate
    - Select the closest matching model
    - Double-check mileage information
    """)
    
    st.markdown("### 📊 Model Info")
    st.info("""
    This predictor uses **Random Forest Regression** with:
    - 300+ decision trees
    - Advanced feature engineering
    - Real market data training
    """)
    
    st.markdown("### 🔄 Data Updates")
    st.success("Model trained on latest market data")