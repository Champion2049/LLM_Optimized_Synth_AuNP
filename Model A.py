import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import model_from_json
import os

# Set the path to the model and scaler files (in the parent directory)
current_dir = os.path.dirname(__file__)
PARENT_DIR = os.path.join(os.path.dirname(current_dir), "saved_models")

# Paths to the model files
json_model_path = os.path.join(PARENT_DIR, "config.json")
weights_model_path = os.path.join(PARENT_DIR, "model.weights.h5")

# Load XGBoost model and scalers
xgb_model = joblib.load(os.path.join(PARENT_DIR, "aunp_xgboost_model_20250502_201740.joblib"))
scaler_X = joblib.load(os.path.join(PARENT_DIR, "scaler_X_20250502_201740.joblib"))
scaler_y = joblib.load(os.path.join(PARENT_DIR, "scaler_y_20250502_201740.joblib"))

# Load the deep learning model from JSON and weights
with open(json_model_path, 'r') as json_file:
    model_json = json_file.read()

dl_model = model_from_json(model_json)
dl_model.load_weights(weights_model_path)

# Feature names
features = [
    "Precursor_Conc_mM", "Reducing_Agent", "Stabilizer", "pH", "Temperature_C", 
    "Reaction_Time_min", "Mixing_Speed_RPM", "Polydispersity", "Particle_Size_nm", 
    "Zeta_Potential_mV", "Drug_Loading_Efficiency_%", "Targeting_Efficiency_%", 
    "Cytotoxicity_%"
]

# Sliders and inputs for parameters
Precursor_Conc_mM = st.slider('Precursor Concentration (mM)', min_value=0.0, max_value=10.0, step=0.1)
Temperature_C = st.slider('Temperature (°C)', min_value=20, max_value=100, step=1)
Reaction_Time_min = st.slider('Reaction Time (minutes)', min_value=1, max_value=60, step=1)
Mixing_Speed_RPM = st.slider('Mixing Speed (RPM)', min_value=0.01, max_value=1.5, step=0.01)

# Dropdown for reducing agent
Reducing_Agent = st.selectbox("Select Reducing Agent", 
    options=["NaBH4", "ascorbic_acid", "citrate"])

# Dropdown for stabilizer
Stabilizer = st.selectbox("Select Stabilizer", 
    options=["CTAB", "PEG", "PVP", "citrate"])

# Inputs for other continuous features
pH = st.number_input('pH', min_value=0.0, max_value=14.0, step=0.1)
Polydispersity = st.number_input('Polydispersity', min_value=0.0, max_value=1.0, step=0.01)

# Prepare the input array with the right order of features
input_data = [
    Precursor_Conc_mM,
    1 if Reducing_Agent == "NaBH4" else 0,
    1 if Reducing_Agent == "ascorbic_acid" else 0,
    1 if Reducing_Agent == "citrate" else 0,
    1 if Stabilizer == "CTAB" else 0,
    1 if Stabilizer == "PEG" else 0,
    1 if Stabilizer == "PVP" else 0,
    1 if Stabilizer == "citrate" else 0,
    pH,
    Temperature_C,
    Reaction_Time_min,
    Mixing_Speed_RPM,
    Polydispersity
]

# Scale the input data
scaled_input = scaler_X.transform([input_data])

# Add the predict button
predict_button = st.button('Predict')

if predict_button:
    # Make predictions for all five target outputs
    xgb_pred = xgb_model.predict(scaled_input)  # Use scaled_input for prediction
    dl_pred = dl_model.predict(scaled_input)  # Use scaled_input for prediction

    # Reshape prediction to match the scaler_y's expected shape
    xgb_pred = xgb_pred.reshape(-1, 5)  # Ensure it's a 2D array with 5 columns (one for each target)
    dl_pred = dl_pred.reshape(-1, 5)    # Ensure it's a 2D array with 5 columns (one for each target)

    # Inverse the scaling of the predictions
    xgb_pred = scaler_y.inverse_transform(xgb_pred)
    dl_pred = scaler_y.inverse_transform(dl_pred)

    # Display the separate predictions for all five targets
    targets = [
        "Cytotoxicity_%", "Targeting_Efficiency_%", "Drug_Loading_Efficiency_%", 
        "Zeta_Potential_mV", "Particle_Size_nm"
    ]

    st.write("### XGBoost Predictions")
    for i, target in enumerate(targets):
        st.write(f"{target}: {xgb_pred[0][i]:.2f}")

    st.write("### Deep Learning Predictions")
    for i, target in enumerate(targets):
        st.write(f"{target}: {dl_pred[0][i]:.2f}")