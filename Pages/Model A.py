import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import model_from_json
import os
import torch
import torch.nn as nn # PyTorch models use torch.nn

# --- Configuration ---
# Set the path to the model and scaler files (assuming they are in the parent directory relative to the script)
# Adjust this path if your file structure is different
try:
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    # Go up one level to the parent directory
    PARENT_DIR = os.path.dirname(current_dir)
except NameError:
    # __file__ is not defined, e.g., when running in an interactive environment
    # Fallback to current working directory or specify the path directly
    PARENT_DIR = "./saved_models" # Or replace with the actual path to the parent directory
    st.warning(f"Could not determine script directory automatically. Assuming models are relative to: {os.path.abspath(PARENT_DIR)}")


# --- Model and Scaler Paths ---
# Keras DL Model
json_model_path = os.path.join(PARENT_DIR, "config.json")
weights_model_path = os.path.join(PARENT_DIR, "model.weights.h5")
# XGBoost Model
xgb_model_path = os.path.join(PARENT_DIR, "aunp_xgboost_model_20250502_201740.joblib")
# Scalers
scaler_X_path = os.path.join(PARENT_DIR, "scaler_X_20250502_201740.joblib")
scaler_y_path = os.path.join(PARENT_DIR, "scaler_y_20250502_201740.joblib")
# PyTorch Models (Update filenames to match your saved models)
# This should point to the model saved by DCN.py (e.g., final_mlp_model_with_scalers_5targets.pth)
dcn_style_mlp_model_path = os.path.join(PARENT_DIR, "final_mlp_model_with_scalers_5targets.pth")
# This should point to the model saved by MLP_gpu.py (e.g., aunp_model_YYYYMMDD_HHMMSS.pth)
mlp_gpu_style_model_path = os.path.join(PARENT_DIR, "aunp_model_YYYYMMDD_HHMMSS.pth") # <-- UPDATE FILENAME (replace YYYYMMDD_HHMMSS)

# --- Feature Definitions ---
# Define the expected input features *after* one-hot encoding
# This must match the order used during training ALL models
encoded_features = [
    "Precursor_Conc_mM",
    "Reducing_Agent_NaBH4",        # One-hot encoded features
    "Reducing_Agent_ascorbic_acid",
    "Reducing_Agent_citrate",
    "Stabilizer_CTAB",             # One-hot encoded features
    "Stabilizer_PEG",
    "Stabilizer_PVP",
    "Stabilizer_citrate",
    "pH",
    "Temperature_C",
    "Reaction_Time_min",
    "Mixing_Speed_RPM",
    "Polydispersity"
]
# Define the target variables in the order the models predict them
targets = [
    "Cytotoxicity_%",
    "Targeting_Efficiency_%",
    "Drug_Loading_Efficiency_%",
    "Zeta_Potential_mV",
    "Particle_Size_nm"
]
# Ensure the order matches scaler_y and model outputs
# Note: The order in MLP_gpu.py was ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%']
# Note: The order in DCN.py was ['Cytotoxicity_%', 'Targeting_Efficiency_%', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%', 'Particle_Size_nm']
# You *MUST* ensure the 'targets' list here matches the order your scaler_y expects and the order your models actually output.
# Let's assume the order defined above is correct for scaler_y. Models might need adjustment if they output in a different order.

num_input_features = len(encoded_features)
num_output_features = len(targets)

# --- Helper Functions ---
def load_keras_model(json_path, weights_path):
    """Loads a Keras model from JSON and H5 weights files."""
    if not os.path.exists(json_path):
        st.error(f"Keras model JSON file not found at: {json_path}")
        return None
    if not os.path.exists(weights_path):
        st.error(f"Keras model weights file not found at: {weights_path}")
        return None
    try:
        with open(json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        st.success("Keras model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None

def load_joblib_model(path):
    """Loads a model or scaler from a joblib file."""
    if not os.path.exists(path):
        st.error(f"Joblib file not found at: {path}")
        return None
    try:
        model = joblib.load(path)
        st.success(f"Joblib file loaded successfully from: {os.path.basename(path)}")
        return model
    except Exception as e:
        st.error(f"Error loading joblib file {os.path.basename(path)}: {e}")
        return None

# --- PyTorch Model Definitions (Derived from your files) ---

# Model based on the structure in DCN.py (SimpleMLP / FinalMLP)
class MLPWithBatchNormModel(nn.Module):
    """
    MLP with BatchNorm and Dropout, based on the structure found in DCN.py.
    Uses a fixed architecture inspired by the Optuna search space.
    """
    def __init__(self, num_inputs, num_outputs, hidden_size=256, num_layers=4, dropout_rate=0.2):
        super(MLPWithBatchNormModel, self).__init__()
        layers = []
        in_dim = num_inputs
        for i in range(num_layers):
            out_dim_layer = hidden_size
            layers.append(nn.Linear(in_dim, out_dim_layer))
            # BatchNorm1d expects input shape (N, C) or (N, L), where C=num_features or L=length
            # It normalizes over the batch dimension.
            layers.append(nn.BatchNorm1d(out_dim_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim_layer

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, num_outputs)
        print(f"Initialized MLPWithBatchNormModel: {num_layers} hidden layers, hidden_size={hidden_size}, dropout={dropout_rate}")


    def forward(self, x):
        # Ensure input is at least 2D for BatchNorm1d: (batch_size, num_features)
        if x.dim() == 1:
            x = x.unsqueeze(0) # Add batch dimension if missing
        mlp_out = self.mlp(x)
        outputs = self.output_layer(mlp_out)
        return outputs

# Model based on the structure in MLP_gpu.py (FinalNet)
class SimpleMLPModel(nn.Module):
    """
    Simple MLP with Dropout, based on the structure found in MLP_gpu.py.
    Uses a fixed architecture inspired by the Optuna search space.
    """
    def __init__(self, num_inputs, num_outputs, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super(SimpleMLPModel, self).__init__()
        layers = []
        in_dim = num_inputs
        for i in range(num_layers):
            out_dim_layer = hidden_size
            layers.append(nn.Linear(in_dim, out_dim_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim_layer

        self.model = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, num_outputs) # Add final output layer
        print(f"Initialized SimpleMLPModel: {num_layers} hidden layers, hidden_size={hidden_size}, dropout={dropout_rate}")


    def forward(self, x):
        hidden_output = self.model(x)
        output = self.output_layer(hidden_output) # Pass through final layer
        return output


# --- Load Models and Scalers ---
st.title("Nanoparticle Synthesis Prediction")

# Load scalers first
scaler_X = load_joblib_model(scaler_X_path)
scaler_y = load_joblib_model(scaler_y_path)

# Load non-PyTorch models
xgb_model = load_joblib_model(xgb_model_path)
dl_model = load_keras_model(json_model_path, weights_model_path)

# Load PyTorch models
dcn_style_mlp_model = None
mlp_gpu_style_model = None

if scaler_X and scaler_y: # Only proceed if scalers loaded
    # Instantiate PyTorch models with correct dimensions
    try:
        # Model from DCN.py structure
        dcn_model_instance = MLPWithBatchNormModel(num_inputs=num_input_features, num_outputs=num_output_features)
        # Model from MLP_gpu.py structure
        mlp_model_instance = SimpleMLPModel(num_inputs=num_input_features, num_outputs=num_output_features)

        # Load DCN-style MLP state dictionary
        if os.path.exists(dcn_style_mlp_model_path):
            try:
                # Load the bundle which contains state_dict and potentially other info
                checkpoint_dcn = torch.load(dcn_style_mlp_model_path, map_location=torch.device('cpu')) # Load to CPU
                # Check if the bundle contains the state dict directly or nested
                if 'model_state_dict' in checkpoint_dcn:
                    dcn_model_instance.load_state_dict(checkpoint_dcn['model_state_dict'])
                else:
                    # Assume the loaded object *is* the state_dict (older saving method)
                    dcn_model_instance.load_state_dict(checkpoint_dcn)
                dcn_model_instance.eval() # Set to evaluation mode
                dcn_style_mlp_model = dcn_model_instance # Assign if loaded successfully
                st.success("DCN-style MLP model loaded successfully.")
            except Exception as e:
                st.error(f"Error loading DCN-style MLP model state_dict from {os.path.basename(dcn_style_mlp_model_path)}: {e}")
                st.info("Ensure the model architecture defined in the script matches the saved checkpoint.")
        else:
            st.error(f"DCN-style MLP model file not found at: {dcn_style_mlp_model_path}")

        # Load MLP GPU style state dictionary
        if os.path.exists(mlp_gpu_style_model_path):
            try:
                 # Load the bundle which contains state_dict and potentially other info
                checkpoint_mlp = torch.load(mlp_gpu_style_model_path, map_location=torch.device('cpu')) # Load to CPU
                # Check if the bundle contains the state dict directly or nested
                if 'model_state_dict' in checkpoint_mlp:
                    mlp_model_instance.load_state_dict(checkpoint_mlp['model_state_dict'])
                else:
                     # Assume the loaded object *is* the state_dict (older saving method)
                    mlp_model_instance.load_state_dict(checkpoint_mlp)
                mlp_model_instance.eval() # Set to evaluation mode
                mlp_gpu_style_model = mlp_model_instance # Assign if loaded successfully
                st.success("MLP (GPU-style) model loaded successfully.")
            except Exception as e:
                st.error(f"Error loading MLP (GPU-style) model state_dict from {os.path.basename(mlp_gpu_style_model_path)}: {e}")
                st.info("Ensure the model architecture defined in the script matches the saved checkpoint.")
        else:
            st.error(f"MLP (GPU-style) model file not found at: {mlp_gpu_style_model_path}")

    except Exception as e:
        st.error(f"Error instantiating PyTorch models: {e}")

else:
    st.error("Scalers could not be loaded. Cannot proceed with model loading or prediction.")
    st.stop() # Stop execution if scalers failed to load

# --- User Inputs ---
st.sidebar.header("Input Parameters")

# Sliders and inputs for parameters
Precursor_Conc_mM = st.sidebar.slider('Precursor Concentration (mM)', min_value=0.0, max_value=10.0, step=0.1, value=5.0)
Temperature_C = st.sidebar.slider('Temperature (°C)', min_value=20, max_value=100, step=1, value=60)
Reaction_Time_min = st.sidebar.slider('Reaction Time (minutes)', min_value=1, max_value=60, step=1, value=30)
Mixing_Speed_RPM = st.sidebar.slider('Mixing Speed (RPM)', min_value=100.0, max_value=1500.0, step=10.0, value=500.0) # Adjusted range

# Dropdown for reducing agent
reducing_agent_options = ["NaBH4", "ascorbic_acid", "citrate"]
Reducing_Agent = st.sidebar.selectbox("Select Reducing Agent", options=reducing_agent_options, index=0)

# Dropdown for stabilizer
stabilizer_options = ["CTAB", "PEG", "PVP", "citrate"]
Stabilizer = st.sidebar.selectbox("Select Stabilizer", options=stabilizer_options, index=1)

# Inputs for other continuous features
pH = st.sidebar.number_input('pH', min_value=0.0, max_value=14.0, step=0.1, value=7.0)
Polydispersity = st.sidebar.number_input('Polydispersity Index (PDI)', min_value=0.0, max_value=1.0, step=0.01, value=0.2)

# --- Data Preparation ---
# Create the input array with one-hot encoding, matching 'encoded_features' order
input_data = np.array([
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
]).reshape(1, -1) # Reshape to 2D array for scaler

# --- Prediction ---
predict_button = st.button('Predict Nanoparticle Properties')

if predict_button:
    # Check if all models required for prediction are loaded
    models_loaded = scaler_X and scaler_y and xgb_model and dl_model and dcn_style_mlp_model and mlp_gpu_style_model
    if models_loaded:
        try:
            # Scale the input data
            scaled_input_np = scaler_X.transform(input_data)

            # --- XGBoost Prediction ---
            xgb_pred_scaled = xgb_model.predict(scaled_input_np)
            if xgb_pred_scaled.ndim == 1:
                 xgb_pred_scaled = xgb_pred_scaled.reshape(1, -1)
            if xgb_pred_scaled.shape[1] != num_output_features:
                 st.error(f"XGBoost prediction has {xgb_pred_scaled.shape[1]} columns, expected {num_output_features}.")
                 xgb_pred_final = np.full((1, num_output_features), np.nan)
            else:
                xgb_pred_final = scaler_y.inverse_transform(xgb_pred_scaled)

            # --- Keras DL Prediction ---
            dl_pred_scaled = dl_model.predict(scaled_input_np)
            dl_pred_scaled = dl_pred_scaled.reshape(1, -1)
            if dl_pred_scaled.shape[1] != num_output_features:
                st.error(f"Keras prediction has {dl_pred_scaled.shape[1]} columns, expected {num_output_features}.")
                dl_pred_final = np.full((1, num_output_features), np.nan)
            else:
                dl_pred_final = scaler_y.inverse_transform(dl_pred_scaled)

            # --- PyTorch Predictions ---
            scaled_input_torch = torch.tensor(scaled_input_np, dtype=torch.float32)

            with torch.no_grad(): # Disable gradient calculation for inference
                # DCN-style MLP Prediction
                dcn_pred_scaled_torch = dcn_style_mlp_model(scaled_input_torch)
                dcn_pred_scaled_np = dcn_pred_scaled_torch.cpu().numpy().reshape(1, -1) # Convert to numpy, ensure 2D
                if dcn_pred_scaled_np.shape[1] != num_output_features:
                     st.error(f"DCN-style MLP prediction has {dcn_pred_scaled_np.shape[1]} columns, expected {num_output_features}.")
                     dcn_pred_final = np.full((1, num_output_features), np.nan)
                else:
                    dcn_pred_final = scaler_y.inverse_transform(dcn_pred_scaled_np)

                # MLP (GPU-style) Prediction
                mlp_pred_scaled_torch = mlp_gpu_style_model(scaled_input_torch)
                mlp_pred_scaled_np = mlp_pred_scaled_torch.cpu().numpy().reshape(1, -1) # Convert to numpy, ensure 2D
                if mlp_pred_scaled_np.shape[1] != num_output_features:
                     st.error(f"MLP (GPU-style) prediction has {mlp_pred_scaled_np.shape[1]} columns, expected {num_output_features}.")
                     mlp_pred_final = np.full((1, num_output_features), np.nan)
                else:
                    mlp_pred_final = scaler_y.inverse_transform(mlp_pred_scaled_np)

            # --- Display Results ---
            st.write("---")
            st.header("Prediction Results")

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.subheader("XGBoost")
                if not np.isnan(xgb_pred_final).any():
                    for i, target in enumerate(targets):
                        st.metric(label=target, value=f"{xgb_pred_final[0, i]:.2f}")
                else:
                     st.warning("Could not generate XGBoost predictions.")

            with col2:
                st.subheader("Keras DL")
                if not np.isnan(dl_pred_final).any():
                    for i, target in enumerate(targets):
                        st.metric(label=target, value=f"{dl_pred_final[0, i]:.2f}")
                else:
                     st.warning("Could not generate Keras predictions.")

            with col3:
                st.subheader("MLP (BatchNorm)") # Renamed from DCN
                if not np.isnan(dcn_pred_final).any():
                    for i, target in enumerate(targets):
                        st.metric(label=target, value=f"{dcn_pred_final[0, i]:.2f}")
                else:
                    st.warning("Could not generate MLP (BatchNorm) predictions.")

            with col4:
                st.subheader("MLP (Simple)") # Renamed from MLP (PyTorch)
                if not np.isnan(mlp_pred_final).any():
                    for i, target in enumerate(targets):
                        st.metric(label=target, value=f"{mlp_pred_final[0, i]:.2f}")
                else:
                    st.warning("Could not generate MLP (Simple) predictions.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            import traceback
            st.error(traceback.format_exc()) # Print detailed traceback for debugging

    else:
        st.error("One or more models/scalers failed to load. Cannot make predictions. Please check file paths and model definitions.")
        # Provide more specific feedback if possible
        if not scaler_X: st.warning("Scaler X not loaded.")
        if not scaler_y: st.warning("Scaler Y not loaded.")
        if not xgb_model: st.warning("XGBoost model not loaded.")
        if not dl_model: st.warning("Keras model not loaded.")
        if not dcn_style_mlp_model: st.warning("MLP (BatchNorm) model not loaded.")
        if not mlp_gpu_style_model: st.warning("MLP (Simple) model not loaded.")


# Add some information about the app
st.sidebar.info(
    "This app predicts nanoparticle properties based on synthesis parameters "
    "using multiple machine learning models."
)
