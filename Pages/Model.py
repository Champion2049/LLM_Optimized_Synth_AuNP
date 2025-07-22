import streamlit as st
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import time
from dotenv import load_dotenv
import google.generativeai as genai

# --- Page Configuration (Set this FIRST) ---
st.set_page_config(
    page_title="AuNP Synthesis Predictor & Optimizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load environment variables and Configure Gemini API ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Configuration ---
DATA_FILE_PATH = "./aunp_synthesis_realistic_v1.csv"
MODEL_DIR = "./saved_hybrid_models"

# Define the model's output targets
REGRESSION_TARGETS = [
    'Particle_Size_nm', 'Particle_Width_nm', 'Drug_Loading_Efficiency', 
    'Targeting_Efficiency', 'Cytotoxicity'
]
BINARY_CLASS_TARGET = 'Successful_Treatment'

# --- Use-Case Specific Suitability Criteria ---
SUITABILITY_CRITERIA = {
    "Cancer Treatment (General)": {
        'Particle_Size_nm': {'ideal_range': (10, 150), 'description': "Optimal size to leverage the Enhanced Permeability and Retention (EPR) effect for passive tumor accumulation, while being large enough to avoid rapid renal clearance."},
        'Particle_Width_nm': {'ideal_range': (5, 40), 'description': "For non-spherical particles, ensures an appropriate aspect ratio for navigating the tumor microenvironment and cellular uptake."},
        'Drug_Loading_Efficiency': {'ideal_range': (30, 60), 'description': "A balanced loading capacity to ensure a therapeutic dose without causing premature drug release or nanoparticle instability."},
        'Targeting_Efficiency': {'ideal_range': (40, 60), 'description': "Effective targeting to maximize drug concentration at the tumor site while minimizing systemic toxicity."},
        'Cytotoxicity': {'ideal_range': (10, 30), 'description': "Indicates potent cancer cell-killing ability within a controlled therapeutic window to avoid excessive off-target damage."}
    },
    "Targeted Drug Delivery (Systemic)": {
        'Particle_Size_nm': {'ideal_range': (20, 80), 'description': "Balances circulation time with the ability to extravasate into target tissues."},
        'Particle_Width_nm': {'ideal_range': (10, 40), 'description': "Ensures stability and appropriate aspect ratio for circulation."},
        'Drug_Loading_Efficiency': {'ideal_range': (80, 100), 'description': "Maximizes the therapeutic payload to reduce the required dose."},
        'Targeting_Efficiency': {'ideal_range': (85, 100), 'description': "Crucial for minimizing systemic toxicity and engaging specific cell receptors."},
        'Cytotoxicity': {'ideal_range': (5, 20), 'description': "The nanoparticle itself should be non-toxic; toxicity should come from the drug payload."}
    },
    "Bio-imaging Contrast Agent": {
        'Particle_Size_nm': {'ideal_range': (5, 40), 'description': "Small size for good distribution and renal clearance, while providing a strong optical signal."},
        'Particle_Width_nm': {'ideal_range': (2, 20), 'description': "Appropriate dimensions for plasmon resonance and stability."},
        'Drug_Loading_Efficiency': {'ideal_range': (0, 100), 'description': "Not a primary concern unless used for theranostics (therapy + diagnostics)."},
        'Targeting_Efficiency': {'ideal_range': (60, 100), 'description': "Important for highlighting specific tissues or cell types."},
        'Cytotoxicity': {'ideal_range': (0, 10), 'description': "Must be extremely low to be considered a safe diagnostic agent."}
    }
}


# ==============================================================================
# --- Custom Loss Functions (MUST match the training script) ---
# ==============================================================================
def weighted_mse_loss(y_true, y_pred):
    weights = tf.constant([1.0, 1.0, 25.0, 25.0, 2.0]) 
    squared_errors = tf.square(y_true - y_pred)
    weighted_squared_errors = squared_errors * weights
    return tf.reduce_mean(weighted_squared_errors)

def create_weighted_binary_crossentropy(weights_dict):
    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weights = y_true * weights_dict.get(1, 1.0) + (1.0 - y_true) * weights_dict.get(0, 1.0)
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce
    return weighted_binary_crossentropy

# ==============================================================================
# --- Resource Loading ---
# ==============================================================================
def find_file(directory, prefix, extension):
    try:
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
        if not files: return None
        files.sort(reverse=True)
        return os.path.join(directory, files[0])
    except FileNotFoundError:
        st.error(f"Error: Directory not found: {directory}")
        return None
    return None

@st.cache_resource
def load_resources(data_file_path, model_dir):
    resources = {"df": None, "input_features": [], "model": None, "feature_preprocessor": None, "regression_scaler": None, "load_errors": []}
    try:
        df = pd.read_csv(data_file_path)
        resources["df"] = df
        all_targets = REGRESSION_TARGETS + [BINARY_CLASS_TARGET]
        resources["input_features"] = [col for col in df.columns if col not in all_targets and col != 'Precursor']
    except Exception as e:
        resources["load_errors"].append(f"Error loading data: {e}")
        return resources

    preprocessor_file = find_file(model_dir, "feature_preprocessor", ".joblib")
    if preprocessor_file: resources["feature_preprocessor"] = joblib.load(preprocessor_file)
    else: resources["load_errors"].append("Feature preprocessor not found.")

    scaler_file = find_file(model_dir, "regression_target_scaler", ".joblib")
    if scaler_file: resources["regression_scaler"] = joblib.load(scaler_file)
    else: resources["load_errors"].append("Regression target scaler not found.")

    model_file = find_file(model_dir, "best_hybrid_model", ".keras")
    if model_file:
        try:
            dummy_weights = {0: 1.0, 1: 1.0} 
            custom_objects = {'weighted_mse_loss': weighted_mse_loss, 'weighted_binary_crossentropy': create_weighted_binary_crossentropy(dummy_weights)}
            resources["model"] = load_model(model_file, custom_objects=custom_objects)
        except Exception as e:
            resources["load_errors"].append(f"Error loading Keras model: {e}")
    else:
        resources["load_errors"].append("Keras model file not found.")
    return resources

# ==============================================================================
# --- AI Analysis Functions (Now using Gemini) ---
# ==============================================================================
def call_gemini_api(prompt, max_retries=3, initial_delay=1.0):
    """Calls the Gemini API with retry logic."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured. Please set it in your .env file."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                st.warning(f"Gemini API call failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (2 ** attempt))
                else:
                    return f"Error: Gemini API call failed after {max_retries} attempts."
    except Exception as e:
        return f"An unexpected error occurred while configuring the Gemini model: {e}"
    return "Error: Max retries reached."


def create_ai_analysis_prompt(user_input_df, predictions_df, success_prediction, use_case, criteria):
    prompt = f"Task: Analyze the results of a gold nanoparticle (AuNP) synthesis prediction for the specific use-case of **{use_case}** and provide an explanation and optimization suggestions.\n\n"
    prompt += "--- Input Synthesis Parameters ---\n"
    for col, val in user_input_df.iloc[0].items():
        prompt += f"- {col.replace('_', ' ').title()}: {val}\n"
    
    prompt += "\n--- Predicted Outcomes ---\n"
    for col, val in predictions_df.iloc[0].items():
        prompt += f"- {col.replace('_', ' ').title()}: {val:.2f}\n"
    
    success_text = "SUCCESSFUL" if success_prediction else "UNSUCCESSFUL"
    prompt += f"- Predicted Treatment/Application Success: {success_text}\n"

    prompt += f"\n--- Ideal Criteria for {use_case} ---\n"
    for prop, details in criteria.items():
        prompt += f"- {prop.replace('_', ' ').title()}: Ideal Range {details['ideal_range']} ({details['description']})\n"

    prompt += """
--- Instructions ---
1.  **Explanation:** Based on the ideal criteria for the specified use-case, briefly explain the significance of the predicted properties. Why are these values good or bad for this application?
2.  **Optimization:** Based on the inputs and predicted outputs, suggest 1-2 specific, actionable changes to the input parameters that could better align the outcomes with the ideal criteria for the chosen use-case. Provide a scientific rationale for each suggestion.
"""
    return prompt

# --- Load Resources ---
resources = load_resources(DATA_FILE_PATH, MODEL_DIR)
df = resources["df"]
INPUT_FEATURES = resources["input_features"]
ml_model = resources["model"]
feature_preprocessor = resources["feature_preprocessor"]
regression_scaler = resources["regression_scaler"]

if resources["load_errors"]:
    st.error("Issues during resource loading:")
    for err in resources["load_errors"]: st.warning(err)

# ==============================================================================
# --- UI Layout ---
# ==============================================================================
st.title("🔬 Gold Nanoparticle Synthesis Predictor & Optimizer")
st.markdown("An advanced tool to predict AuNP properties and treatment success, with AI-driven analysis for specific use-cases.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("🧬 Synthesis Parameters")
    
    if not INPUT_FEATURES or df is None:
        st.error("Input features could not be loaded.")
        st.stop()
    
    # Use-case selection
    use_case = st.selectbox(
        "Select Use-Case",
        options=list(SUITABILITY_CRITERIA.keys()),
        help="The chosen use-case will determine the ideal property ranges and tailor the AI's analysis."
    )
    
    st.divider()

    user_input = {}
    for feature in INPUT_FEATURES:
        if df[feature].dtype == 'object':
            options = [''] + sorted(df[feature].unique().tolist())
            user_input[feature] = st.selectbox(label=feature.replace("_", " ").title(), options=options, key=feature)
        else:
            min_val, max_val, median_val = float(df[feature].min()), float(df[feature].max()), float(df[feature].median())
            user_input[feature] = st.number_input(label=feature.replace("_", " ").title(), min_value=min_val, max_value=max_val, value=median_val, key=feature)

# --- Main Panel for Results ---
st.header(f"🎯 Analysis for: {use_case}")

# Display the dynamic suitability criteria
with st.expander("View Ideal Property Ranges for this Use-Case"):
    criteria_df = pd.DataFrame([
        {"Property": k.replace('_', ' ').title(), "Ideal Range": str(v['ideal_range']), "Description": v['description']}
        for k, v in SUITABILITY_CRITERIA[use_case].items()
    ])
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)


# --- Prediction and Analysis ---
if st.button("🚀 Predict & Analyze", type="primary", use_container_width=True):
    if any(val == '' or val is None for val in user_input.values()):
        st.warning("⚠️ Please fill in all input fields in the sidebar.")
    elif ml_model is None or feature_preprocessor is None or regression_scaler is None:
        st.error("Critical resources failed to load. Prediction cannot proceed.")
    else:
        with st.spinner("🔬 Running Keras model and Gemini AI analysis..."):
            try:
                # 1. Keras Model Prediction
                input_df = pd.DataFrame([user_input])
                input_processed = feature_preprocessor.transform(input_df)
                pred_reg_scaled, pred_binary_proba = ml_model.predict(input_processed)
                pred_reg_unscaled = regression_scaler.inverse_transform(pred_reg_scaled)
                pred_binary_class = (pred_binary_proba[0][0] > 0.5)
                
                predictions_df = pd.DataFrame(pred_reg_unscaled, columns=REGRESSION_TARGETS)

                # 2. Display Keras Predictions in a structured layout
                st.subheader("🧪 Predicted Results (from Keras Model)")
                
                with st.container(border=True):
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.metric(label="Application Success Prediction", value="✅ Successful" if pred_binary_class else "❌ Unsuccessful")
                    with res_col2:
                        st.markdown("**Predicted Physical Properties:**")
                        prop_cols = st.columns(3)
                        for i, target in enumerate(REGRESSION_TARGETS):
                            prop_cols[i % 3].metric(
                                label=target.replace('_', ' ').title(), 
                                value=f"{predictions_df[target].iloc[0]:.2f}"
                            )
                
                st.divider()

                # 3. External AI Analysis
                st.subheader("🤖 AI Analysis & Optimization (from Gemini)")
                ai_prompt = create_ai_analysis_prompt(input_df, predictions_df, pred_binary_class, use_case, SUITABILITY_CRITERIA[use_case])
                ai_response = call_gemini_api(ai_prompt)
                
                with st.container(border=True):
                    if "Error:" not in ai_response:
                        st.markdown(ai_response)
                    else:
                        st.error(f"Failed to get analysis from AI. Response: {ai_response}")

            except Exception as e:
                st.error("An error occurred during prediction or analysis:")
                st.exception(e)

st.divider()
st.caption("Powered by TensorFlow/Keras, Google Gemini, and Streamlit.")