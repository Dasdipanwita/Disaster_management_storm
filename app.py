import streamlit as st
import numpy as np
import pickle

# Load the trained model, imputer, scaler, and encoder
try:
    with open('storm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('storm_imputer.pkl', 'rb') as imputer_file:
        imputer = pickle.load(imputer_file)
    with open('storm_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('storm_encoder.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Please run training first.")
    st.stop()

# Streamlit UI
st.title("🌪️ Storm Status Predictor")
st.markdown("Enter the storm's details to predict its status.")

# Input fields
lat = st.number_input("Latitude", value=32.5)
long = st.number_input("Longitude", value=-52.0)
wind = st.number_input("Wind Speed (knots)", value=20)
pressure = st.number_input("Pressure (mb)", value=1005.0, step=0.1)

# Prediction
if st.button("Predict Status"):
    try:
        # Prepare features
        features = np.array([[lat, long, wind, pressure]])

        # Impute missing values
        features_imputed = imputer.transform(features)

        # Scale
        features_scaled = scaler.transform(features_imputed)

        # Predict
        prediction_encoded = model.predict(features_scaled)
        prediction = encoder.inverse_transform(prediction_encoded)[0]

        st.success(f"✅ Predicted Storm Status: **{prediction}**")
    except Exception as e:
        st.error(f"Error: {e}")
