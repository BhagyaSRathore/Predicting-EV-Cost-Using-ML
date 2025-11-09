# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="EV Cost Predictor",
    page_icon="🚗",
    layout="wide"
)

# ------------------ CUSTOM STYLES ------------------
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fb;
        }
        .stButton>button {
            background-color: #0072B5;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 24px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #005a8d;
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER SECTION ------------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Electric Vehicle Price Predictor")
    st.subheader("Predict EV prices using Machine Learning")
    st.write(
        "Get an estimated market price for an electric vehicle by entering key specifications. "
        "Built using a trained ML model."
    )
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2885/2885844.png", width=120)

st.markdown("---")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("EV_price_predictor.pkl")
        return model
    except FileNotFoundError:
        st.error(" Model file not found. Please upload 'EV_price_predictor.pkl' to this repo.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# ------------------ SIDEBAR INPUT SECTION ------------------
st.sidebar.header("Enter Vehicle Specifications")

battery_capacity = st.sidebar.number_input(
    "Battery Capacity (kWh)", min_value=10.0, max_value=200.0, value=50.0, step=1.0
)
range_km = st.sidebar.number_input("Range (km)", min_value=50, max_value=1000, value=300, step=10)
power = st.sidebar.number_input("Power (bhp)", min_value=20, max_value=800, value=150)
torque = st.sidebar.number_input("Torque (Nm)", min_value=50, max_value=1200, value=300)
fast_charging = st.sidebar.selectbox("Fast Charging Capability", ["Yes", "No"])
seating_capacity = st.sidebar.number_input("Seating Capacity", min_value=2, max_value=8, value=5)
brand = st.sidebar.selectbox(
    "Brand",
    ["Tata", "MG", "Hyundai", "BYD", "Mahindra", "Kia", "BMW", "Audi", "Mercedes", "Porsche"],
)

fast_charging_val = 1 if fast_charging == "Yes" else 0

# ------------------ INPUT DATAFRAME ------------------
input_data = pd.DataFrame(
    {
        "Battery_Capacity(kWh)": [battery_capacity],
        "Range(km)": [range_km],
        "Power(bhp)": [power],
        "Torque(Nm)": [torque],
        "Fast_Charging": [fast_charging_val],
        "Seating_Capacity": [seating_capacity],
        "Brand": [brand],
    }
)

st.markdown("### Input Summary")
st.dataframe(input_data, use_container_width=True)

# ------------------ PREDICTION ------------------
if st.button(" Predict EV Price"):
    if model is not None:
        try:
            prediction = model.predict(input_data)
            predicted_price = float(prediction[0])
            st.success(f" **Estimated EV Price: ₹{predicted_price:,.2f} Lakh**")
            st.balloons()

            st.markdown(
                """
                ###  Insights:
                - Higher battery capacity and range increase cost.
                - Premium brands (Audi, BMW, Mercedes) have higher base prices.
                - Fast-charging support improves resale value.
                """
            )
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.warning(" Please ensure your model file is available.")

# ------------------ INDUSTRY INSIGHTS ------------------
st.markdown("---")
st.markdown("###  EV Industry Overview")
st.write(
    """
    Electric Vehicles are revolutionizing transportation.  
    Battery efficiency, range, and brand reputation are key factors driving EV prices.  
    This predictor helps visualize pricing trends and assist decision-making.
    """
)

# ------------------ CHATBOT PLACEHOLDER ------------------
st.markdown("---")
st.markdown("###  Chat with EVBot (Coming Soon)")
st.info("AI-powered EV assistant under development. Future integration planned with OpenAI API.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    """
    <center>
    👩‍💻 Built by <b>Bhagya Singh Rathore</b><br>
    🔗 <a href="https://github.com/BhagyaSRathore/Predicting-EV-Cost-Using-ML" target="_blank">
    GitHub Repository</a><br>
    ⚙️ Powered by Streamlit & Machine Learning
    </center>
    """,
    unsafe_allow_html=True,
)
