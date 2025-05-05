# integrated_app_final.py

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

# --- Load environment variables from .env file ---
load_dotenv(dotenv_path='GROQ_API_KEY.env')

# Groq API Configuration (Direct Call)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama3-8b-8192"

# Configuration - Update this path to your actual data file location
DATA_FILE_PATH = r"C:\Users\mechi\Documents\GitHub\LLM_Transformer_Model\aunp_synthesis_cancer_treatment_v3_transformed.csv"
MODEL_DIR = r"C:\Users\mechi\Documents\GitHub\LLM_Transformer_Model\saved_models" # Assuming model/scaler/explanation files are in this directory

# Define your target columns (from LLM.py)
TARGET_COLS = ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%',
               'Targeting_Efficiency_%', 'Cytotoxicity_%']
OUTPUT_FEATURES = [col for col in TARGET_COLS if col != 'Cytotoxicity_%'] # Features to display as primary outputs in UI

# Hardcoded suitability criteria (from LLM.py)
suitability_criteria = {
    'Particle_Size_nm': {'ideal_range': (40, 100), 'description': "Optimal particle size range..."},
    'Zeta_Potential_mV': {'ideal_range': (-30, -5), 'description': "Zeta potential indicates..."},
    'Drug_Loading_Efficiency_%': {'ideal_range': (70, 100), 'description': "Indicates how efficiently..."},
    'Targeting_Efficiency_%': {'ideal_range': (75, 100), 'description': "Measures how well..."},
    'Cytotoxicity_%': {'ideal_range': (70, 90), 'description': "Indicates toxicity to..."}
}

# --- Helper function to find a file ---
def find_file(directory, prefix, extension):
    """Finds a file starting with prefix and ending with extension in a directory."""
    for f_name in os.listdir(directory):
        if f_name.startswith(prefix) and f_name.endswith(extension):
            return os.path.join(directory, f_name)
    return None

# --- Load Data, Model, and Scalers (modified for Streamlit caching) ---
@st.cache_resource
def load_resources(data_file_path, model_dir, target_cols):
    """Loads data, model, scalers, explanation, and insights."""
    df = pd.DataFrame()
    input_features = []

    # Load Data and determine input features
    try:
        df = pd.read_csv(data_file_path)
        # Determine input features dynamically from data columns
        input_features = [col for col in df.columns if col not in target_cols and col != 'Suitable_for_Cancer_Treatment']
        st.success(f"Data loaded successfully from {data_file_path}. Identified {len(input_features)} input features.")
        if not input_features:
             st.warning("No input features identified from data columns.")
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {data_file_path}. Cannot determine input features dynamically.")
    except Exception as e:
        st.error(f"Error loading data from {data_file_path}: {e}")


    # Load Model
    model_file = find_file(model_dir, "best_dl_model", ".keras")
    ml_model = None
    if model_file:
        try:
            ml_model = tf.keras.models.load_model(model_file)
            st.success(f"ML model loaded successfully from {model_file}")
        except Exception as e:
            st.error(f"Error loading ML model from {model_file}: {e}")
    else:
        st.error(f"ML model file not found in {model_dir}. Cannot make predictions.")

    # Load Scaler X
    scaler_x_file = find_file(model_dir, "scaler_X", ".joblib")
    scaler_X = None
    if scaler_x_file:
        try:
            scaler_X = joblib.load(scaler_x_file)
            st.success(f"Scaler X loaded successfully from {scaler_x_file}")
             # Optional: Add a check here to see if scaler_X expects the number of features found in input_features
            if scaler_X is not None and hasattr(scaler_X, 'n_features_in_') and len(input_features) > 0:
                 if scaler_X.n_features_in_ != len(input_features):
                      st.warning(f"Scaler X was fitted on {scaler_X.n_features_in_} features, but {len(input_features)} were identified from the data file. This may cause errors.")

        except Exception as e:
            st.error(f"Error loading Scaler X from {scaler_x_file}: {e}")
    else:
        st.error(f"Scaler X file not found in {model_dir}. Cannot preprocess input.")


    # Load Scaler Y
    scaler_y_file = find_file(model_dir, "scaler_y", ".joblib")
    scaler_y = None
    if scaler_y_file:
        try:
            scaler_y = joblib.load(scaler_y_file)
            st.success(f"Scaler y loaded successfully from {scaler_y_file}")
             # Optional: Add a check here to see if scaler_y expects the number of target features
            if scaler_y is not None and hasattr(scaler_y, 'n_features_in_') and len(TARGET_COLS) > 0:
                 # Note: scaler_y should be fitted on the *target* columns, which is TARGET_COLS
                 if scaler_y.n_features_in_ != len(TARGET_COLS):
                      st.warning(f"Scaler y was fitted on {scaler_y.n_features_in_} features, but {len(TARGET_COLS)} target features were expected. This may cause errors.")
        except Exception as e:
            st.error(f"Error loading Scaler y from {scaler_y_file}: {e}")
    else:
        st.error(f"Scaler y file not found in {model_dir}. Cannot inverse transform output.")

    # Load explanation and insights
    explanation_file = find_file(model_dir, "aunp_synthesis_explanation", ".md")
    loaded_explanation_text = "Explanation text not loaded."
    if explanation_file:
        try:
            with open(explanation_file, 'r') as f:
                loaded_explanation_text = f.read()
            st.success(f"Explanation text loaded from {explanation_file}")
        except Exception as e:
            st.warning(f"Could not load explanation text from {explanation_file}: {e}")

    insights_file = find_file(model_dir, "feature_insights", ".json")
    loaded_feature_insights = {}
    if insights_file:
        try:
            with open(insights_file, 'r') as f:
                loaded_feature_insights = json.load(f)
            st.success(f"Feature insights loaded from {insights_file}")
        except Exception as e:
            st.warning(f"Could not load feature insights from {insights_file}: {e}")


    return df, input_features, ml_model, scaler_X, scaler_y, loaded_explanation_text, loaded_feature_insights


# Load all resources using the cached function
df, INPUT_FEATURES_DYNAMIC, ml_model, scaler_X, scaler_y, loaded_explanation_text, loaded_feature_insights = load_resources(DATA_FILE_PATH, MODEL_DIR, TARGET_COLS)


# --- Function to create prompt for LLM classification ---
def create_llm_classification_prompt(sample, feature_explanation, criteria, input_features_list):
    """
    Create a prompt for an LLM to classify if a sample is suitable for cancer treatment.
    Includes checks for numeric values before formatting.
    """

    prompt = f"""Task: Determine if the gold nanoparticle (AuNP) synthesis parameters and resulting properties are suitable for cancer treatment.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{feature_explanation}

Sample to Evaluate:
"""

    # Add input features using the provided input_features_list
    prompt += "## Input Parameters:\n"
    input_cols_in_sample = [col for col in sample.index if col in input_features_list]
    if input_cols_in_sample:
        for col in input_cols_in_sample:
            value = sample[col]
            # Check if the value is numeric before formatting
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                 prompt += f"- {col}: {value:.4f}\n"
            else:
                 prompt += f"- {col}: {value}\n" # Display non-numeric or NaN as is
    else:
         prompt += "No input parameters provided in sample.\n"


    # Add output properties (using TARGET_COLS as defined in this script)
    prompt += "\n## Resulting Properties:\n"
    output_cols_in_sample = [col for col in TARGET_COLS if col in sample.index]
    if output_cols_in_sample:
        for col in output_cols_in_sample:
            value = sample[col]
            # Check if the value is numeric before formatting
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                 prompt += f"- {col}: {value:.4f}\n"
            else:
                 prompt += f"- {col}: {value}\n" # Display non-numeric or NaN as is
    else:
         prompt += "No resulting properties provided in sample.\n"


    # Add the classification question
    prompt += """\nBased on the criteria for optimal cancer treatment, are these gold nanoparticle properties suitable?
Please analyze each property against the ideal ranges and explain your reasoning.
Conclude with a clear YES or NO classification."""

    return prompt

# --- Function to create prompt for LLM optimization ---
def generate_synthesis_optimization_prompt(sample, feature_explanation, criteria, insights, input_features_list):
    """
    Generate a prompt for the LLM to suggest optimizations to the synthesis method
    based on the current parameters and desired outputs.
    Includes checks for numeric values before formatting.
    """
    unmet_criteria = []
    for col, spec in criteria.items():
        if col in sample.index:
            low, high = spec['ideal_range']
            value = sample[col]
            # Check if the value is numeric before comparing or formatting
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                if value < low:
                    unmet_criteria.append(f"{col} is too low ({value:.2f} vs. ideal minimum {low})")
                elif value > high:
                    unmet_criteria.append(f"{col} is too high ({value:.2f} vs. ideal maximum {high})")
            else:
                 unmet_criteria.append(f"Property '{col}' has a non-numeric or missing value ({value}).")
        else:
             unmet_criteria.append(f"Property '{col}' is missing from the sample data.")


    prompt = f"""Task: Suggest optimizations to the gold nanoparticle (AuNP) synthesis method to improve its suitability for cancer treatment.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{feature_explanation}

Current Synthesis Parameters and Results:
"""

    # Add input parameters using the provided input_features_list
    prompt += "## Input Parameters:\n"
    input_cols_in_sample = [col for col in sample.index if col in input_features_list]
    if input_cols_in_sample:
         for col in input_cols_in_sample:
            value = sample[col]
            # Check if the value is numeric before formatting
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                 prompt += f"- {col}: {value:.4f}\n"
            else:
                 prompt += f"- {col}: {value}\n" # Display non-numeric or NaN as is
    else:
         prompt += "No input parameters provided in sample.\n"

    prompt += "\n## Resulting Properties:\n"
    output_cols_in_sample = [col for col in TARGET_COLS if col in sample.index]
    if output_cols_in_sample:
         for col in output_cols_in_sample:
            value = sample[col]
            # Check if the value is numeric before formatting
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                 prompt += f"- {col}: {value:.4f}\n"
            else:
                 prompt += f"- {col}: {value}\n" # Display non-numeric or NaN as is
    else:
         prompt += "No resulting properties provided in sample.\n"


    if unmet_criteria:
        prompt += "\n## Criteria Not Met:\n"
        for issue in unmet_criteria:
            prompt += f"- {issue}\n"
    else:
         prompt += "\n## All Criteria Met:\n\nThe current synthesis parameters appear suitable.\n"

    # Add feature insights if available
    if insights:
        prompt += "\n## Feature Relationship Insights (from data analysis):\n"
        for feature, insight_list in insights.items():
             prompt += f"### {feature}:\n"
             for insight in insight_list:
                  prompt += f"- {insight}\n"

    prompt += "\nBased on the feature relationships and current parameters, please suggest specific modifications to the synthesis method that would improve the properties to meet all criteria for cancer treatment.\nProvide a step-by-step explanation of:\n1. Which parameters should be adjusted and by how much\n2. Expected impact on each target property\n3. Scientific rationale for each suggested change\n4. A revised synthesis protocol incorporating these changes"

    return prompt


# --- Function to call the Groq API directly (with refined parsing and debug messages removed) ---
def call_groq_api(prompt, model=GROQ_MODEL_NAME, max_tokens=1000, temperature=0.7, max_retries=3, initial_delay=1.0):
    """
    Calls the Groq API with the given prompt, implementing retry with exponential backoff.
    Modified to return generated text and a parsed suitability label (1 for Yes, 0 for No, None otherwise).
    Refined parsing logic to prioritize explicit conclusion and 'NOT SUITABLE' keyword.
    Debug messages removed.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY environment variable not set. Please check your .env file.", None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            response.raise_for_status()

            response_data = response.json()

            if 'choices' in response_data and response_data['choices']:
                generated_text = response_data['choices'][0].get('message', {}).get('content', '')

                # --- Refined Parsing Logic for suitability label ---
                predicted_label = None
                upper_text = generated_text.upper()

                # 1. Prioritize explicit CONCLUSION: YES/NO match at the end
                # Try matching variations of the conclusion statement at the end of the text
                conclusion_match = re.search(r"CLASSIFY THE AU NP PROPERTIES AS:\s*(YES|NO)\s*$", upper_text)
                if not conclusion_match:
                     conclusion_match = re.search(r"CLASSIFY AS:\s*(YES|NO)\s*$", upper_text)
                if not conclusion_match:
                     conclusion_match = re.search(r"CONCLUSION:\s*(YES|NO)\s*$", upper_text)

                if conclusion_match:
                    # Use the matched conclusion
                    predicted_label = 1 if conclusion_match.group(1) == "YES" else 0
                else:
                    # 2. Fallback to keyword check if explicit conclusion not found
                    # Prioritize "NOT SUITABLE" if mentioned anywhere
                    if "NOT SUITABLE" in upper_text:
                        predicted_label = 0 # If "NOT SUITABLE" is mentioned, classify as Not Suitable
                    elif "SUITABLE" in upper_text:
                         # Only classify as Suitable if "NOT SUITABLE" was NOT found
                        predicted_label = 1
                    else:
                        predicted_label = None # Still couldn't determine


                return generated_text.strip(), predicted_label

            else:
                return "Could not generate response from Groq API.", None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retries += 1
                wait_time = initial_delay * (2 ** (retries - 1))
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        pass
                st.warning(f"Rate limit hit (429). Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
            else:
                st.error(f"HTTP Error during Groq API request: {e}")
                return f"Failed to get response from Groq API. Error: {e}", None
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error during Groq API call: {e}")
            return f"Failed to get response from Groq API. Error: {e}", None
        except json.JSONDecodeError:
            st.error("Error: Could not decode JSON response from Groq API.")
            return "Failed to parse Groq API response.", None
        except Exception as e:
            st.error(f"An unexpected error occurred during Groq API call: {e}")
            return "An unexpected error occurred during API call.", None

    st.error(f"Error: Max retries ({max_retries}) reached after hitting rate limit.")
    return f"Error: Max retries ({max_retries}) reached after hitting rate limit.", None


# --- Streamlit UI ---
st.title("Gold Nanocluster Property Predictor and Optimizer 🌟")

st.write("Enter the synthesis parameters below to predict the properties of Gold Nanoclusters and get suitability analysis and optimization suggestions.")

# Use the dynamically loaded input features list
if not INPUT_FEATURES_DYNAMIC:
    st.error("Input features could not be loaded from the data file. Please check the DATA_FILE_PATH and the file content.")
else:
    # Create input fields based on dynamically loaded features
    user_input = {}
    st.subheader("Input Synthesis Parameters")
    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(INPUT_FEATURES_DYNAMIC):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3

        # Heuristic to determine input type (selectbox or number_input)
        # This is based on column names and checking unique values in loaded data
        is_binary_like = False
        if feature in df.columns:
             unique_values = df[feature].dropna().unique()
             if len(unique_values) <= 2 and all(uv in [0, 1, 0.0, 1.0] for uv in unique_values):
                 is_binary_like = True

        if feature.startswith("Reducing_Agent_") or feature.startswith("Stabilizer_") or "Binary" in feature or is_binary_like:
             # For binary/categorical, use selectbox
             user_input[feature] = col.selectbox(f"{feature}", [0, 1], key=feature)
        else:
             # For numerical, use number_input
             user_input[feature] = col.number_input(f"{feature}", value=0.0, format="%.4f", key=feature)


    # Convert user input dictionary to pandas Series
    user_sample_input_series = pd.Series(user_input)

    # Predict and Analyze on button click
    if st.button("Predict and Analyze"):
        if ml_model is None or scaler_X is None or scaler_y is None:
            st.error("Model or scalers not loaded. Please check the file paths and ensure necessary files exist.")
        elif user_sample_input_series.isnull().any():
             st.warning("Please provide values for all input features.")
        elif len(user_sample_input_series) != len(INPUT_FEATURES_DYNAMIC):
             st.error(f"Mismatch in input features. Expected {len(INPUT_FEATURES_DYNAMIC)} but received {len(user_sample_input_series)}. Check data loading and UI generation.")
        else:
            with st.spinner("Predicting properties and analyzing suitability..."):
                try:
                    # Ensure the order of input features matches the scaler's expected order
                    # This requires knowing the order scaler_X was fitted with, which is usually
                    # the column order of the training data. We assume INPUT_FEATURES_DYNAMIC
                    # preserves this order as it's derived from the data columns.
                    input_array = user_sample_input_series[INPUT_FEATURES_DYNAMIC].values.reshape(1, -1)
                    input_scaled = scaler_X.transform(input_array)

                    # Make prediction using the ML model
                    predicted_outputs_scaled = ml_model.predict(input_scaled)
                    predicted_outputs_unscaled = scaler_y.inverse_transform(predicted_outputs_scaled)

                    # Extract and format predicted output
                    predicted_output_values = predicted_outputs_unscaled[0]
                    predicted_output_series = pd.Series(predicted_output_values, index=TARGET_COLS)

                    st.subheader("🧪 Predicted AuNP Properties:")
                    output_cols = st.columns(len(OUTPUT_FEATURES))
                    for i, output in enumerate(OUTPUT_FEATURES):
                         # Ensure the output exists in the predicted_output_series and is numeric before formatting
                         if output in predicted_output_series.index:
                            value = predicted_output_series[output]
                            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                                metric_value = f"{value:.4f}"
                            else:
                                metric_value = "N/A" # Handle non-numeric or NaN
                            output_cols[i].metric(label=output, value=metric_value)
                         else:
                            output_cols[i].metric(label=output, value="N/A")

                    # Display Cytotoxicity separately if it's in TARGET_COLS but not in OUTPUT_FEATURES
                    if 'Cytotoxicity_%' in TARGET_COLS and 'Cytotoxicity_%' not in OUTPUT_FEATURES:
                         if 'Cytotoxicity_%' in predicted_output_series.index:
                            value = predicted_output_series['Cytotoxicity_%']
                            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                                metric_value = f"{value:.4f}"
                            else:
                                metric_value = "N/A" # Handle non-numeric or NaN
                            st.metric(label='Cytotoxicity_% (for Suitability Check)', value=metric_value)
                         else:
                             st.metric(label='Cytotoxicity_% (for Suitability Check)', value="N/A")


                    # Combine input and predicted output for LLM
                    combined_sample = pd.concat([user_sample_input_series, predicted_output_series])

                    # --- Call Groq API for Classification ---
                    st.subheader("🤔 Suitability Analysis")
                    classification_prompt = create_llm_classification_prompt(combined_sample, loaded_explanation_text, suitability_criteria, INPUT_FEATURES_DYNAMIC)
                    classification_response, predicted_suitability_label = call_groq_api(
                        classification_prompt, max_tokens=700, temperature=0.5
                    )

                    if classification_response:
                        suitability_status = 'Yes' if predicted_suitability_label == 1 else ('No' if predicted_suitability_label == 0 else 'Could not determine')
                        st.write(f"**Suitable for Cancer Treatment:** {suitability_status}")
                        st.write("---")
                        st.write("**Groq API Explanation:**")
                        st.write(classification_response)
                    else:
                        st.error("Failed to get suitability analysis from Groq API.")

                    time.sleep(1) # Small delay

                    # --- Call Groq API for Optimization Suggestions ---
                    st.subheader("💡 Optimization Suggestions")
                    optimization_prompt = generate_synthesis_optimization_prompt(
                        combined_sample, loaded_explanation_text, suitability_criteria, loaded_feature_insights, INPUT_FEATURES_DYNAMIC
                    )
                    optimization_suggestion, _ = call_groq_api(
                        optimization_prompt, max_tokens=1500, temperature=0.7
                    )

                    if optimization_suggestion:
                        st.write("**Groq API Optimization Suggestions:**")
                        st.write(optimization_suggestion)
                    else:
                        st.error("Failed to get optimization suggestions from Groq API.")


                except Exception as e:
                    st.error(f"An error occurred during prediction or analysis: {e}")
                    st.exception(e) # Display full traceback for debugging