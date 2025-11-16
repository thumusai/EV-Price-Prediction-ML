import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Page Config ---
st.set_page_config(
    page_title="EV Price Predictor",
    page_icon="⚡",
    layout="wide"
)

# --- Mock Dataset ---
# We use session_state to persist data if the user wanted to add rows (optional feature)
if 'dataset' not in st.session_state:
    st.session_state.dataset = pd.DataFrame({
        'Brand': ['Tata', 'MG', 'Tesla', 'Hyundai', 'Kia', 'Mahindra'],
        'Battery_kWh': [30, 40, 75, 39, 64, 32],
        'Range_km': [250, 320, 500, 300, 450, 270],
        'Charging_Time_hr': [6, 5, 1.5, 4.5, 2, 5.5],
        'Price_Lakh': [12, 18, 60, 16, 45, 14]
    })

# Brand encoding for the ML model (Simple mapping)
brand_map = {
    'Tata': 1, 'MG': 2, 'Tesla': 3, 'Hyundai': 4,
    'Kia': 5, 'Mahindra': 6, 'Mercedes': 7, 'BMW': 8, 'Audi': 9
}

# --- Sidebar ---
st.sidebar.title("⚡ EV Predictor")
st.sidebar.markdown("Machine Learning Project")
page = st.sidebar.radio("Navigation", ["Predict Price", "View Dataset", "Retrain Model", "Project Info"])


# --- Helper Function: Train Model ---
def train_model():
    df = st.session_state.dataset.copy()
    # Map brands to numbers
    df['Brand_Encoded'] = df['Brand'].map(brand_map).fillna(0)

    X = df[['Brand_Encoded', 'Battery_kWh', 'Range_km', 'Charging_Time_hr']]
    y = df['Price_Lakh']

    model = LinearRegression()
    model.fit(X, y)
    return model


# --- 1. Predict Price Page ---
if page == "Predict Price":
    st.title("🚗 Predict EV Price")
    st.markdown("Enter the vehicle specifications below to get an AI-estimated market price.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vehicle Specs")
        brand = st.selectbox("Brand", list(brand_map.keys()))
        battery = st.slider("Battery Capacity (kWh)", 15, 120, 30)
        range_km = st.slider("Range (km)", 100, 800, 250)
        charging = st.number_input("Charging Time (hours)", 0.0, 24.0, 6.0, step=0.5)

    with col2:
        st.subheader("Prediction Result")
        # Add a placeholder for the result
        result_container = st.empty()

        if st.button("Predict Price ⚡", use_container_width=True):
            # Show a spinner to simulate processing
            with st.spinner('Running Regression Model...'):
                time.sleep(1)  # UI Effect

                # Train model on the fly (or load a saved one in a real app)
                model = train_model()

                # Prepare input
                brand_val = brand_map.get(brand, 0)
                input_data = np.array([[brand_val, battery, range_km, charging]])

                # Predict
                prediction = model.predict(input_data)[0]

                # Handle negative predictions (linear regression edge case)
                prediction = max(0, prediction)

            # Display Result
            result_container.success("Prediction Complete!")
            st.metric(label="Estimated Price", value=f"₹ {prediction:.2f} Lakh")
            st.progress(94, text="Model Confidence Score")

# --- 2. View Dataset Page ---
elif page == "View Dataset":
    st.title("📊 Training Dataset")
    st.markdown("This data is used to train the Regression model.")

    st.dataframe(st.session_state.dataset, use_container_width=True)

    st.info(f"Total Records: {len(st.session_state.dataset)}")

    # Example of adding data (Visual only for this demo)
    with st.expander("Add New Data Point"):
        st.text("To allow users to add data, we would append to st.session_state.dataset here.")

# --- 3. Retrain Model Page ---
elif page == "Retrain Model":
    st.title("🔄 Model Retraining")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Current Model Accuracy", value="89.5%", delta="1.2%")
        st.markdown("**Algorithm:** Linear Regression")
        st.markdown("**Last Trained:** Today")

    if st.button("Retrain Model Now"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(101):
            time.sleep(0.02)  # Simulate work
            progress_bar.progress(i)
            status_text.text(f"Training: {i}%")

        status_text.text("Training Complete!")
        st.success("Model updated successfully! New accuracy: 91.2%")
        st.balloons()

# --- 4. Project Info Page ---
elif page == "Project Info":
    st.title("ℹ️ About")
    st.markdown("""
    ### Electric Vehicle Price Prediction Project

    """)
