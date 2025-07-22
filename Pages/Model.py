import streamlit as st
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import requests
import re
import time
import json
from dotenv import load_dotenv

# --- Page Configuration (Set this FIRST) ---
st.set_page_config(
    page_title="AuNP Property Predictor & Optimizer",
    page_icon="✨", # You can use emojis or a path to an icon file
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="auto" # Or 'expanded' or 'collapsed'
)

# --- Load environment variables from .env file ---
load_dotenv(dotenv_path='./GROQ_API_KEY.env')
# Groq API Configuration (Direct Call)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama3-8b-8192" # Or another model if needed

# Configuration - Update this path to your actual data file location
# Use a raw string (r"...") or forward slashes for paths
# Ensure this path is correct for your environment
DATA_FILE_PATH = "./aunp_synthesis_cancer_treatment_v3_transformed.csv"
MODEL_DIR = "./saved_models" # Assuming model/scaler/explanation files are in this directory

# Define your target columns
TARGET_COLS = ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%',
               'Targeting_Efficiency_%', 'Cytotoxicity_%']
OUTPUT_FEATURES = [col for col in TARGET_COLS if col != 'Cytotoxicity_%'] # Features to display as primary outputs in UI

# Hardcoded suitability criteria - Particle Size range changed to (10, 100)
# Suitability is now determined programmatically based on meeting a COUNT threshold
suitability_criteria = {
    'Particle_Size_nm': {'ideal_range': (10, 100), 'description': "A range including smaller sizes, potentially relevant for certain applications, up to the optimal size for tumor accumulation and cellular uptake via EPR effect."},
    'Zeta_Potential_mV': {'ideal_range': (-30, -5), 'description': "Slightly negative charge indicates stability and can aid cell interaction without excessive non-specific binding."},
    'Drug_Loading_Efficiency_%': {'ideal_range': (70, 100), 'description': "High efficiency ensures sufficient drug delivery per nanoparticle."},
    'Targeting_Efficiency_%': {'ideal_range': (75, 100), 'description': "High targeting ensures nanoparticles preferentially accumulate in tumor tissue."},
    'Cytotoxicity_%': {'ideal_range': (70, 90), 'description': "Indicates effective cancer cell killing within a therapeutic window (higher is better, up to a point to avoid excessive off-target toxicity)."}
}

# --- Helper function to find a file ---
def find_file(directory, prefix, extension):
    """Finds a file starting with prefix and ending with extension in a directory."""
    try:
        for f_name in os.listdir(directory):
            if f_name.startswith(prefix) and f_name.endswith(extension):
                return os.path.join(directory, f_name)
    except FileNotFoundError:
        st.error(f"Error: Directory not found: {directory}")
        return None
    except Exception as e:
        st.error(f"Error listing files in directory {directory}: {e}")
        return None
    return None

# --- Load Data, Model, and Scalers (using Streamlit caching) ---
@st.cache_resource # Caches the loaded resources
def load_resources(data_file_path, model_dir, target_cols):
    """Loads data, model, scalers, explanation, and insights."""
    df = pd.DataFrame()
    input_features = []
    ml_model, scaler_X, scaler_y = None, None, None
    loaded_explanation_text = "Default explanation: AuNP properties are critical for cancer treatment effectiveness."
    loaded_feature_insights = {}
    load_errors = [] # Collect errors during loading

    # Load Data and determine input features
    try:
        df = pd.read_csv(data_file_path)
        all_cols = df.columns.tolist()
        excluded_cols = target_cols + ['Suitable_for_Cancer_Treatment'] # Columns to exclude from inputs
        input_features = [col for col in all_cols if col not in excluded_cols]
        if not input_features:
            load_errors.append("Warning: No input features identified from data columns after excluding targets.")
        # st.success(f"Data loaded. Identified {len(input_features)} input features.") # Optional debug
    except FileNotFoundError:
        load_errors.append(f"Error: Data file not found at {data_file_path}. Cannot determine input features.")
    except Exception as e:
        load_errors.append(f"Error loading data from {data_file_path}: {e}")

    # Load Model
    model_file = find_file(model_dir, "best_dl_model", ".keras")
    if model_file:
        try:
            ml_model = load_model(model_file)
            # st.success("ML model loaded.") # Optional debug
        except Exception as e:
            load_errors.append(f"Error loading ML model from {model_file}: {e}")
    else:
        load_errors.append(f"ML model file (best_dl_model*.keras) not found in {model_dir}.")

    # Load Scaler X
    scaler_x_file = find_file(model_dir, "scaler_X", ".joblib")
    if scaler_x_file:
        try:
            scaler_X = joblib.load(scaler_x_file)
            # st.success("Scaler X loaded.") # Optional debug
            # Feature number check
            if scaler_X is not None and hasattr(scaler_X, 'n_features_in_') and input_features:
                 if scaler_X.n_features_in_ != len(input_features):
                      load_errors.append(f"Warning: Scaler X expects {scaler_X.n_features_in_} features, but {len(input_features)} were identified from data. Mismatch may cause errors.")
        except Exception as e:
            load_errors.append(f"Error loading Scaler X from {scaler_x_file}: {e}")
    else:
        load_errors.append(f"Scaler X file (scaler_X*.joblib) not found in {model_dir}.")

    # Load Scaler Y
    scaler_y_file = find_file(model_dir, "scaler_y", ".joblib")
    if scaler_y_file:
        try:
            scaler_y = joblib.load(scaler_y_file)
            # st.success("Scaler y loaded.") # Optional debug
            # Feature number check
            if scaler_y is not None and hasattr(scaler_y, 'n_features_in_'):
                 if scaler_y.n_features_in_ != len(TARGET_COLS):
                      load_errors.append(f"Warning: Scaler y expects {scaler_y.n_features_in_} features, but {len(TARGET_COLS)} target columns defined. Mismatch may cause errors.")
        except Exception as e:
            load_errors.append(f"Error loading Scaler y from {scaler_y_file}: {e}")
    else:
        load_errors.append(f"Scaler y file (scaler_y*.joblib) not found in {model_dir}.")

    # Load explanation and insights
    explanation_file = find_file(model_dir, "aunp_synthesis_explanation", ".md")
    if explanation_file:
        try:
            with open(explanation_file, 'r', encoding='utf-8') as f: # Added encoding
                loaded_explanation_text = f.read()
            # st.success("Explanation text loaded.") # Optional debug
        except Exception as e:
            load_errors.append(f"Warning: Could not load explanation text from {explanation_file}: {e}")
    else:
        load_errors.append(f"Warning: Explanation file (aunp_synthesis_explanation*.md) not found in {model_dir}.")


    insights_file = find_file(model_dir, "feature_insights", ".json")
    if insights_file:
        try:
            with open(insights_file, 'r', encoding='utf-8') as f: # Added encoding
                loaded_feature_insights = json.load(f)
            # st.success("Feature insights loaded.") # Optional debug
        except Exception as e:
            load_errors.append(f"Warning: Could not load feature insights from {insights_file}: {e}")
    else:
         load_errors.append(f"Warning: Feature insights file (feature_insights*.json) not found in {model_dir}.")

    # Display loading errors/warnings at the top if any
    if load_errors:
        st.error("Issues during resource loading:")
        for err in load_errors:
            st.warning(err)

    return df, input_features, ml_model, scaler_X, scaler_y, loaded_explanation_text, loaded_feature_insights

# Load all resources using the cached function
df, INPUT_FEATURES_DYNAMIC, ml_model, scaler_X, scaler_y, loaded_explanation_text, loaded_feature_insights = load_resources(DATA_FILE_PATH, MODEL_DIR, TARGET_COLS)


# --- Function to create prompt for LLM explanation (MODIFIED for explicit check) ---
def create_llm_explanation_prompt(sample, feature_explanation, criteria, input_features_list, met_criteria_count, criteria_check_results):
    """
    Create a prompt for an LLM to generate an explanation of suitability based on
    explicit criteria check results and count provided by the Python code.
    """

    prompt = f"""Task: Generate an explanation of the suitability of the gold nanoparticle (AuNP) synthesis parameters and resulting properties for cancer treatment, based on the provided criteria checks and the count of properties meeting the criteria.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{feature_explanation}

Ideal Criteria for Cancer Treatment Suitability:
"""
    # Add criteria to the prompt for context
    for prop, details in criteria.items():
        prompt += f"- {prop}: Ideal Range {details['ideal_range']} ({details['description']})\n"

    prompt += "\nSample to Evaluate:\n"

    # Add input features using the provided input_features_list
    prompt += "## Input Synthesis Parameters:\n"
    input_cols_in_sample = [col for col in sample.index if col in input_features_list]
    if input_cols_in_sample:
        for col in input_cols_in_sample:
            value = sample[col]
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                prompt += f"- {col}: {value:.4f}\n"
            else:
                prompt += f"- {col}: {value}\n" # Display non-numeric or NaN as is
    else:
        prompt += "No input parameters provided in sample.\n"

    # Add output properties and the EXPLICIT criteria check results
    prompt += "\n## Predicted Resulting Properties and Criteria Check:\n"
    if criteria_check_results:
        for result in criteria_check_results:
            prompt += f"- {result}\n"
    else:
        prompt += "Criteria check results are not available.\n"


    # Add the count and instructions for generating explanation
    prompt += f"""
Number of properties meeting criteria: {met_criteria_count} out of {len(TARGET_COLS)}

Instructions:
1. Based on the "Predicted Resulting Properties and Criteria Check" section, summarize which properties met their ideal range and which did not.
2. Clearly state the total "Number of properties meeting criteria".
3. Provide a detailed explanation of *why* the sample is considered Suitable or Not Suitable based *only* on the fact that {met_criteria_count} out of {len(TARGET_COLS)} properties met the criteria.
4. Discuss the implications of the properties that did *not* meet the criteria, even if the overall conclusion is "Suitable".
5. Structure your response clearly, starting with a summary of the checks, followed by the overall conclusion based on the count, and then the detailed reasoning.
"""
    return prompt

# --- Function to create prompt for LLM optimization (No changes needed) ---
def generate_synthesis_optimization_prompt(sample, feature_explanation, criteria, insights, input_features_list):
    """
    Generate a prompt for the LLM to suggest optimizations to the synthesis method
    based on the current parameters and desired outputs. Includes explicit check results.
    """
    unmet_criteria = []
    met_criteria = []

    # Perform criteria check in Python
    for col, spec in criteria.items():
        if col in sample.index:
            low, high = spec['ideal_range']
            value = sample[col]
            # Check if the value is numeric before comparing or formatting
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                # Adjust value for comparison if it's a percentage property (assuming criteria are 0-100)
                compare_value = value * 100 if col in ['Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%'] else value

                if low <= compare_value <= high:
                    met_criteria.append(f"{col} ({compare_value:.2f}) is within the ideal range ({low}-{high}).")
                elif compare_value < low:
                    unmet_criteria.append(f"{col} ({compare_value:.2f}) is below the ideal range ({low}-{high}). Needs increase.")
                else: # compare_value > high
                    unmet_criteria.append(f"{col} ({compare_value:.2f}) is above the ideal range ({low}-{high}). Needs decrease.")
            else:
                unmet_criteria.append(f"Property '{col}' has a non-numeric or missing value ({value}). Cannot assess.")
        else:
             unmet_criteria.append(f"Property '{col}' is missing from the predicted results.")

    prompt = f"""Task: Suggest specific, actionable optimizations to the gold nanoparticle (AuNP) synthesis method to improve its suitability for cancer treatment, focusing on the properties that are outside the ideal range.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{feature_explanation}

Ideal Criteria for Cancer Treatment Suitability:
"""
    for prop, details in criteria.items():
        prompt += f"- {prop}: Ideal Range {details['ideal_range']} ({details['description']})\n"

    prompt += "\nCurrent Synthesis Parameters and Predicted Results:\n"

    # Add input parameters
    prompt += "## Input Synthesis Parameters:\n"
    input_cols_in_sample = [col for col in sample.index if col in input_features_list]
    if input_cols_in_sample:
        for col in input_cols_in_sample:
            value = sample[col]
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                prompt += f"- {col}: {value:.4f}\n"
            else:
                prompt += f"- {col}: {value}\n"
    else:
        prompt += "No input parameters provided.\n"

    prompt += "\n## Predicted Resulting Properties:\n"
    output_cols_in_sample = [col for col in TARGET_COLS if col in sample.index]
    if output_cols_in_sample:
        for col in output_cols_in_sample:
            value = sample[col]
            # Multiply by 100 for percentage values in the prompt for clarity to the LLM
            if col in ['Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%']:
                 if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                      prompt += f"- {col}: {value * 100:.2f}%\n" # Display as percentage in prompt
                 else:
                      prompt += f"- {col}: {value}\n" # Display non-numeric or NaN as is
            else:
                 if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                      prompt += f"- {col}: {value:.4f}\n"
                 else:
                      prompt += f"- {col}: {value}\n" # Display non-numeric or NaN as is
    else:
        prompt += "No resulting properties predicted.\n"


    # Summarize criteria assessment based on Python check results
    prompt += "\n## Assessment Against Ideal Criteria:\n"
    if unmet_criteria:
        prompt += "Properties needing optimization:\n"
        for issue in unmet_criteria:
            prompt += f"- {issue}\n"
    else:
        prompt += "All predicted properties meet the ideal criteria.\n"

    if met_criteria:
        prompt += "\nProperties already within ideal range:\n"
        for met in met_criteria:
            prompt += f"- {met}\n"

    # Add feature insights if available and relevant
    if insights:
        prompt += "\n## Potential Relationships (from data analysis - use with caution):\n"
        # Only show insights related to input features or target features for brevity?
        # Example: Filter insights if needed
        # relevant_insights = {k: v for k, v in insights.items() if k in input_features_list or k in TARGET_COLS}
        for feature, insight_list in insights.items(): # Using all insights for now
             prompt += f"### {feature}:\n"
             for insight in insight_list:
                 prompt += f"- {insight}\n"

    prompt += """
Instructions for Optimization Suggestions:
1. Focus ONLY on the properties listed under "Properties needing optimization".
2. For each property needing optimization, suggest ADJUSTMENTS to specific INPUT Synthesis Parameters (e.g., "Increase Temperature to X", "Decrease Stabilizer_Concentration_mM to Y").
3. Explain the SCIENTIFIC RATIONALE for why adjusting that specific input parameter is expected to affect the target property in the desired direction (increase/decrease). Reference the background knowledge or general chemistry principles.
4. If possible, suggest a magnitude or direction for the change (e.g., "slightly increase", "significantly decrease").
5. If multiple properties need optimization, consider potential trade-offs (adjusting one input might affect multiple outputs).
6. Present the suggestions clearly, perhaps as a list of recommended actions.
7. If all properties already meet the criteria, state that no optimization is needed based on the current predictions.
"""
    return prompt

# --- Function to call the Groq API directly (No suitability label parsing) ---
def call_groq_api(prompt, model=GROQ_MODEL_NAME, max_tokens=1000, temperature=0.7, max_retries=3, initial_delay=1.0):
    """
    Calls the Groq API with retry logic. Returns the generated text.
    """
    if not GROQ_API_KEY:
        # st.error("Error: GROQ_API_KEY environment variable not set.") # Display error in main UI if needed
        return "Error: GROQ_API_KEY not configured."

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=60) # Added timeout
            response.raise_for_status()
            response_data = response.json()

            if 'choices' in response_data and response_data['choices']:
                generated_text = response_data['choices'][0].get('message', {}).get('content', '').strip()
                return generated_text
            else:
                # Handle cases where API returns success but no choices (should be rare)
                 error_msg = f"Groq API returned success but no 'choices' in response: {response_data}"
                 # st.warning(error_msg) # Show warning in UI
                 return error_msg # Return error message

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429: # Rate limit
                retries += 1
                wait_time = initial_delay * (2 ** (retries - 1))
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    try: wait_time = max(wait_time, int(retry_after)) # Use suggested wait time if available
                    except ValueError: pass
                st.warning(f"Rate limit hit (429). Retrying in {wait_time:.1f}s... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
            elif e.response.status_code >= 500: # Server error
                 retries += 1
                 wait_time = initial_delay * (2 ** (retries - 1))
                 st.warning(f"Groq API server error ({e.response.status_code}). Retrying in {wait_time:.1f}s... (Attempt {retries}/{max_retries})")
                 time.sleep(wait_time)
            else: # Other HTTP errors (e.g., 400 Bad Request, 401 Unauthorized)
                error_msg = f"HTTP Error calling Groq API: {e}"
                st.error(error_msg)
                return f"API Error: {e.response.status_code} - {e.response.text}"
        except requests.exceptions.Timeout:
            retries += 1
            wait_time = initial_delay * (2 ** (retries - 1))
            st.warning(f"Request timed out. Retrying in {wait_time:.1f}s... (Attempt {retries}/{max_retries})")
            time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            # Catch other potential network errors
            error_msg = f"Network Error calling Groq API: {e}"
            st.error(error_msg)
            return error_msg
        except json.JSONDecodeError as e:
            # Handle cases where response isn't valid JSON
            error_msg = f"Error decoding JSON response from Groq API: {e}. Response: {response.text}"
            st.error(error_msg)
            return error_msg
        except Exception as e:
             # Catch any other unexpected errors
             error_msg = f"An unexpected error occurred during Groq API call: {e}"
             st.error(error_msg)
             import traceback
             st.error(traceback.format_exc()) # Log full traceback for debugging
             return error_msg

    # If loop finishes without returning, max retries were reached
    final_error_msg = f"Error: Max retries ({max_retries}) reached. Could not get response from Groq API."
    st.error(final_error_msg)
    return final_error_msg

# --- Streamlit UI ---

st.title("✨ Gold Nanoparticle Synthesis Predictor & Optimizer")
st.markdown("Predict AuNP properties from synthesis parameters, analyze suitability for cancer treatment, and get AI-driven optimization suggestions.")

# Display Suitability Criteria in an Expander
with st.expander("View Suitability Criteria for Cancer Treatment)"):
    st.markdown("These are the target ranges for the predicted properties.")
    criteria_df = pd.DataFrame([
        {"Property": k, "Ideal Range": str(v['ideal_range']), "Description": v['description']}
        for k, v in suitability_criteria.items()
    ])
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

st.divider()

# --- Input Section ---
st.header("🧬 Input Synthesis Parameters")

if not INPUT_FEATURES_DYNAMIC:
    st.error("Input features could not be loaded or determined. Please check the `DATA_FILE_PATH` and ensure the CSV file is correctly formatted and accessible.")
    st.stop() # Stop execution if inputs aren't available

if ml_model is None or scaler_X is None or scaler_y is None:
     st.error("Critical resources (Model or Scalers) failed to load. Prediction cannot proceed. Check errors at the top.")
     st.stop()

# Create input fields based on dynamically loaded features
user_input = {}
num_columns = 3 # Adjust number of columns for inputs
cols = st.columns(num_columns)

for i, feature in enumerate(INPUT_FEATURES_DYNAMIC):
    col_index = i % num_columns
    target_col = cols[col_index]

    # Heuristic to determine input type (selectbox for binary-like, number_input otherwise)
    is_binary_like = False
    default_value = 0.0 # Default for number input
    min_val = 0.0 # Sensible default min
    max_val = None # No default max unless known

    if feature in df.columns:
        unique_values = df[feature].dropna().unique()
        # Check if looks like binary (0/1) or one-hot encoded
        if len(unique_values) <= 2 and all(uv in [0, 1, 0.0, 1.0] for uv in unique_values):
             is_binary_like = True
             # Provide more context for binary features if name suggests category
             label = feature.replace("_", " ").title()
             if "Agent" in label or "Stabilizer" in label or "Method" in label:
                 label = f"{label}"
             user_input[feature] = target_col.selectbox(label, [0, 1], key=feature, index=0) # Default to 0
        else:
             # For numerical, use number_input with calculated step
             if pd.api.types.is_numeric_dtype(df[feature]):
                 min_val = float(df[feature].min()) if not pd.isna(df[feature].min()) else 0.0
                 max_val = float(df[feature].max()) if not pd.isna(df[feature].max()) else None
                 # Calculate a reasonable step based on range, avoid zero step
                 range_val = (max_val - min_val) if max_val is not None else 1.0
                 step = 10**np.floor(np.log10(range_val / 100)) if range_val > 0 else 0.01
                 step = max(step, 1e-6) # Ensure step is not too small or zero
                 default_value = float(df[feature].median()) if not pd.isna(df[feature].median()) else min_val
                 user_input[feature] = target_col.number_input(
                     feature.replace("_", " ").title(),
                     min_value=min_val,
                     max_value=max_val,
                     value=default_value,
                     step=step,
                     format="%.4f", # More precision allowed
                     key=feature,
                     help=f"Range in data: [{min_val:.3f} - {max_val:.3f}]" if max_val is not None else f"Min in data: {min_val:.3f}"
                 )
             else:
                 # Fallback for non-numeric if not binary (should ideally be preprocessed)
                 user_input[feature] = target_col.text_input(f"{feature.replace('_', ' ').title()} (text)", key=feature)
                 st.warning(f"Feature '{feature}' is non-numeric and not binary. Input as text.")

    else:
        # Fallback if feature name not in DataFrame columns (should not happen with current logic)
         user_input[feature] = target_col.number_input(f"{feature.replace('_', ' ').title()}", value=0.0, format="%.4f", key=feature)


# Convert user input dictionary to pandas Series (important for processing)
# Ensure the series has the same index order as INPUT_FEATURES_DYNAMIC
user_sample_input_series = pd.Series(user_input)[INPUT_FEATURES_DYNAMIC]

st.divider()

# --- Prediction and Analysis Button ---
if st.button("🚀 Predict Properties & Analyze Suitability", type="primary", use_container_width=True):
    # --- Input Validation ---
    if user_sample_input_series.isnull().any():
        st.warning("⚠️ Please ensure all input fields have values.")
    elif len(user_sample_input_series) != len(INPUT_FEATURES_DYNAMIC):
         st.error(f"Input feature mismatch. Expected {len(INPUT_FEATURES_DYNAMIC)}, got {len(user_sample_input_series)}. Check UI/data loading.")
    else:
        with st.spinner("🔬 Predicting properties and running AI analysis... Please wait."):
            try:
                # --- 1. Prediction ---
                st.subheader("🧪 Predicted AuNP Properties")
                input_array = user_sample_input_series.values.reshape(1, -1)
                input_scaled = scaler_X.transform(input_array)
                predicted_outputs_scaled = ml_model.predict(input_scaled)
                predicted_outputs_unscaled = scaler_y.inverse_transform(predicted_outputs_scaled)

                predicted_output_values = predicted_outputs_unscaled[0]
                predicted_output_series = pd.Series(predicted_output_values, index=TARGET_COLS)

                # Display predictions using columns and metrics
                pred_cols = st.columns(len(TARGET_COLS)) # Display all targets including Cytotoxicity
                for i, output_name in enumerate(TARGET_COLS):
                     value = predicted_output_series.get(output_name, np.nan) # Use .get for safety
                     if pd.notna(value):
                         # Multiply by 100 for percentage values when displaying
                         display_value = value
                         if output_name in ['Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%']:
                             display_value = value * 100
                             # Also update the label to show it's a percentage explicitly if needed, though % sign is added below
                             label = output_name.replace("_"," ").replace(" Percent","%")
                         else:
                             label = output_name.replace("_"," ")

                         pred_cols[i].metric(label=label, value=f"{display_value:.2f}")
                     else:
                         pred_cols[i].metric(label=output_name.replace("_"," ").replace(" Percent","%"), value="N/A")

                st.divider()

                # --- 2. Suitability Analysis (Programmatic Check + LLM Explanation) ---
                st.subheader("✅ Suitability Analysis")

                # Perform the criteria check and count in Python
                met_criteria_count = 0
                criteria_check_results = []
                for col in TARGET_COLS:
                    value = predicted_output_series.get(col, np.nan)
                    status = "Value N/A (Missing or non-numeric)"
                    if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                        low, high = suitability_criteria.get(col, {}).get('ideal_range', (None, None))
                        if low is not None and high is not None:
                            compare_value = value * 100 if col in ['Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%'] else value
                            if low <= compare_value <= high:
                                status = "Meets criteria"
                                met_criteria_count += 1
                            elif compare_value < low:
                                status = f"Does not meet criteria (too low, ideal range: {low}-{high})"
                            else: # compare_value > high
                                status = f"Does not meet criteria (too high, ideal range: {low}-{high})"
                        else:
                             status = "No ideal range defined for this property"

                        display_value = value * 100 if col in ['Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%'] else value
                        criteria_check_results.append(f"{col}: {display_value:.2f} ({status})")
                    else:
                         criteria_check_results.append(f"{col}: {value} ({status})")

                # Determine suitability based on the count
                predicted_suitability_label = 1 if met_criteria_count >= 3 else 0

                # Create prompt for LLM to generate explanation (no longer asks for conclusion)
                combined_sample = pd.concat([user_sample_input_series, predicted_output_series]) # Still pass combined data for context
                explanation_prompt = create_llm_explanation_prompt(
                    combined_sample,
                    loaded_explanation_text,
                    suitability_criteria,
                    INPUT_FEATURES_DYNAMIC,
                    met_criteria_count, # Pass the calculated count
                    criteria_check_results # Pass the explicit check results
                )

                # st.text_area("DEBUG: Explanation Prompt", explanation_prompt, height=300) # Uncomment for prompt debugging

                # Call LLM to get explanation text only
                explanation_response = call_groq_api(
                    explanation_prompt, max_tokens=1000, temperature=0.5 # Use a slightly higher temp for more detailed explanation
                )

                # Display Suitability Result and LLM Explanation
                if "Error:" not in explanation_response:
                    if predicted_suitability_label == 1:
                        st.success("**Conclusion: Suitable for Cancer Treatment**")
                        with st.container(border=True): # Use border for visual grouping
                             st.markdown("**AI Explanation:**")
                             st.markdown(explanation_response) # Display the full explanation from LLM
                    else: # predicted_suitability_label == 0
                        st.error("**Conclusion: Not Suitable for Cancer Treatment**")
                        with st.container(border=True):
                            st.markdown("**AI Explanation:**")
                            st.markdown(explanation_response) # Display the full explanation from LLM
                else:
                    st.error(f"Failed to get suitability explanation from Groq API. Response: {explanation_response}")


                st.divider()

                # --- 3. Optimization Suggestions (LLM) ---
                st.subheader("💡 Optimization Suggestions (LLM based)")
                # Generate optimization suggestions regardless of overall suitability, focusing on unmet criteria
                optimization_prompt = generate_synthesis_optimization_prompt(
                    combined_sample,
                    loaded_explanation_text,
                    suitability_criteria,
                    loaded_feature_insights,
                    INPUT_FEATURES_DYNAMIC
                )
                # st.text_area("DEBUG: Optimization Prompt", optimization_prompt, height=200) # Uncomment for prompt debugging

                optimization_suggestion = call_groq_api(
                    optimization_prompt, max_tokens=1500, temperature=0.6 # Slightly higher temp for more creative suggestions
                )

                if "Error:" not in optimization_suggestion:
                     if predicted_suitability_label == 1:
                          st.info("✅ The predicted properties meet the suitability criteria (3/5 or more). Optimization suggestions below can help potentially improve the remaining properties.")
                     else:
                          st.info("💡 Optimization suggestions are provided to help improve the properties that did not meet the criteria.")

                     with st.container(border=True):
                         st.markdown("**AI Optimization Suggestions:**")
                         st.markdown(optimization_suggestion) # Use markdown
                else:
                     st.error(f"Failed to get optimization suggestions from Groq API. Response: {optimization_suggestion}")


            except Exception as e:
                st.error(f"An error occurred during the prediction or analysis pipeline:")
                st.exception(e) # Display full traceback for debugging

# --- Footer or additional info ---
st.divider()
st.caption("Powered by TensorFlow/Keras, Groq Llama3, and Streamlit.")
st.markdown("[Code/ Implementation at Github](https://github.com/Champion2049/LLM_Transformer_Model)")