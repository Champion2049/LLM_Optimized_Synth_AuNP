import pandas as pd
import numpy as np
import joblib # For loading the pre-trained scaler
import os
import json
from sklearn.model_selection import train_test_split # Added back for potential future use or if needed for analysis
from sklearn.preprocessing import MinMaxScaler
# Added imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

import time
from datetime import timedelta

# For loading environment variables from a .env file
from dotenv import load_dotenv # Import load_dotenv

# For LLM integration
import requests # Used for making HTTP requests directly to the Groq API
from typing import List, Dict, Tuple, Union, Optional
import re
import time # Ensure time is imported for sleep

# For loading Keras model
import tensorflow as tf # Import tensorflow


# --- Load environment variables from .env file ---
# Assumes your .env file is named GROQ_API_KEY.env and is in the same directory
# as the script, or you can provide a specific path.
load_dotenv(dotenv_path='GROQ_API_KEY.env') # Load variables from the specified .env file

# Configuration
# NOTE: Update this path to your actual file location
DATA_FILE_PATH = "./aunp_synthesis_cancer_treatment_v3_transformed.csv"
MODEL_DIR = "saved_models"
RANDOM_STATE = 42
TEST_SIZE = 0.2 # Used for splitting data if needed for evaluation

# --- Model and Scaler File Prefixes/Extensions ---
# We will find the latest files based on these prefixes and extensions
MODEL_PREFIX = "best_dl_model"
MODEL_EXTENSION = ".keras"
SCALER_X_PREFIX = "scaler_X"
SCALER_Y_PREFIX = "scaler_y"
SCALER_EXTENSION = ".joblib"
EXPLANATION_PREFIX = "aunp_synthesis_explanation"
EXPLANATION_EXTENSION = ".md"
INSIGHTS_PREFIX = "feature_insights"
INSIGHTS_EXTENSION = ".json"
CORRELATION_PLOT_PREFIX = "correlation_matrix"
CORRELATION_PLOT_EXTENSION = ".png"


# Groq API Configuration (Direct Call)
# IMPORTANT: The API key is now loaded from the .env file via load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # This will now get the key loaded by load_dotenv
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Choose a suitable Groq model. Check Groq's documentation for available models.
GROQ_MODEL_NAME = "llama3-8b-8192" # Or "llama3-70b-8192" if you have access and credits

# Small delay between API calls for different samples to help with rate limits
API_CALL_DELAY = 1.0 # seconds

# Target columns (the outputs your ML model predicts)
TARGET_COLS = ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%',
               'Targeting_Efficiency_%', 'Cytotoxicity_%']

# Create output directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# Generate a timestamp for saving files
timestamp = time.strftime("%Y%m%d_%H%M%S")

# --- Helper function to find the latest file with a given prefix and extension ---
def find_latest_file(directory, prefix, extension):
    latest_file = None
    latest_time_struct = None # Initialize with None for comparison
    try:
        for f_name in os.listdir(directory):
            if f_name.startswith(prefix) and f_name.endswith(extension):
                # Extract timestamp (assumingYYYYMMDD_HHMMSS format after prefix)
                # Adjusted slicing to handle prefix length and extension length
                # Find the part of the string between the prefix and the extension
                start_index = len(prefix)
                end_index = f_name.rfind(extension) # Use rfind to handle cases where prefix/extension might appear in timestamp
                if end_index == -1: continue # Skip if extension not found

                timestamp_str_candidate = f_name[start_index:end_index]

                # Check if the part between prefix and extension looks like a timestamp
                # It should start with '_' and then the timestamp format
                if timestamp_str_candidate.startswith('_') and len(timestamp_str_candidate) == len("_YYYYMMDD_HHMMSS"):
                    timestamp_str = timestamp_str_candidate[1:] # Remove the leading underscore
                    try:
                        # Parse the timestamp string into a time.struct_time object
                        file_time_struct = time.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        # Compare time.struct_time objects directly
                        if latest_time_struct is None or file_time_struct > latest_time_struct:
                            latest_time_struct = file_time_struct
                            latest_file = f_name
                    except ValueError:
                        continue # Skip files with incorrect timestamp format
    except FileNotFoundError:
        print(f"Warning: Directory '{directory}' not found.")
        return None
    except Exception as e:
        print(f"Error finding latest file in '{directory}': {e}")
        return None # Return None on error
    return latest_file


# --- Load Data ---
# Load the full dataset for analysis and potential future use (like evaluation)
df = pd.DataFrame() # Initialize df before loading
print(f"Loading data from: {DATA_FILE_PATH}")
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE_PATH}")
    print("Proceeding with empty DataFrame. Some features may not work.")
except Exception as e:
    print(f"Error loading data from {DATA_FILE_PATH}: {e}")
    print("Proceeding with empty DataFrame. Some features may not work.")


# --- Load Explanation and Criteria ---
# The script needs the explanation text and suitability criteria to build prompts.
# It should load these from the files saved by a previous run.

latest_explanation_file = find_latest_file(MODEL_DIR, EXPLANATION_PREFIX, EXPLANATION_EXTENSION)
latest_insights_file = find_latest_file(MODEL_DIR, INSIGHTS_PREFIX, INSIGHTS_EXTENSION)

explanation_text = "Explanation text not loaded. Please run the script once with data loading enabled to generate this file."
feature_insights = {} # Needed for optimization prompt generation
suitability_criteria = { # Default criteria if insights not loaded
    'Particle_Size_nm': {'ideal_range': (40, 100), 'description': "Optimal particle size range..."},
    'Zeta_Potential_mV': {'ideal_range': (-30, -5), 'description': "Zeta potential indicates..."},
    'Drug_Loading_Efficiency_%': {'ideal_range': (70, 100), 'description': "Indicates how efficiently..."},
    'Targeting_Efficiency_%': {'ideal_range': (75, 100), 'description': "Measures how well..."},
    'Cytotoxicity_%': {'ideal_range': (70, 90), 'description': "Indicates toxicity to..."}
}


if latest_explanation_file:
    explanation_path = os.path.join(MODEL_DIR, latest_explanation_file)
    try:
        with open(explanation_path, 'r') as f:
            explanation_text = f.read()
        print(f"Loaded explanation text from {explanation_path}")
    except FileNotFoundError:
        print(f"Error: Explanation text file not found at {explanation_path}")
    except Exception as e:
        print(f"Error loading explanation text: {e}")


if latest_insights_file:
    insights_path = os.path.join(MODEL_DIR, latest_insights_file)
    try:
        with open(insights_path, 'r') as f:
            feature_insights = json.load(f)
            # Update suitability criteria from insights if available and structured correctly
            # Assuming criteria might be stored in insights or you have a separate config
            # For robustness, explicitly define criteria here or load from a known config file.
            # Using the hardcoded criteria from the original script for consistency.
            suitability_criteria = {
                'Particle_Size_nm': {'ideal_range': (40, 100), 'description': "Optimal particle size range for gold nanoparticles in cancer treatment. Particles should be large enough to carry drug payload but small enough to penetrate tumor tissue via the EPR effect."},
                'Zeta_Potential_mV': {'ideal_range': (-30, -5), 'description': "Zeta potential indicates surface charge and stability. Slightly negative values promote stability while facilitating cellular uptake."},
                'Drug_Loading_Efficiency_%': {'ideal_range': (70, 100), 'description': "Indicates how efficiently the drug is loaded onto nanoparticles. Higher values mean more effective drug delivery."},
                'Targeting_Efficiency_%': {'ideal_range': (75, 100), 'description': "Measures how well nanoparticles target cancer cells. Higher values indicate better specificity for cancer cells."},
                'Cytotoxicity_%': {'ideal_range': (70, 90), 'description': "Indicates toxicity to cancer cells. Should be high enough to effectively kill cancer cells but not excessively toxic."}
            }
            print(f"Loaded feature insights from {insights_path}")
            print("Using hardcoded suitability criteria (consider loading from config).")
    except FileNotFoundError:
        print(f"Error: Feature insights file not found at {insights_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {insights_path}")
    except Exception as e:
        print(f"Error loading feature insights: {e}")


# --- Load the pre-trained ML Model and Scalers ---
ml_model = None
scaler_X = None # Scaler for input features
scaler_y = None # Scaler for output targets

# Find the latest model and scaler files
latest_model_file = find_latest_file(MODEL_DIR, MODEL_PREFIX, MODEL_EXTENSION)
latest_scaler_X_file = find_latest_file(MODEL_DIR, SCALER_X_PREFIX, SCALER_EXTENSION)
latest_scaler_Y_file = find_latest_file(MODEL_DIR, SCALER_Y_PREFIX, SCALER_EXTENSION)


if latest_model_file:
    model_path = os.path.join(MODEL_DIR, latest_model_file)
    try:
        ml_model = tf.keras.models.load_model(model_path)
        print(f"ML model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading ML model from {model_path}: {e}")
else:
    print(f"Error: No latest model file found with prefix '{MODEL_PREFIX}' in {MODEL_DIR}. Cannot make predictions.")


if latest_scaler_X_file:
    scaler_x_path = os.path.join(MODEL_DIR, latest_scaler_X_file)
    try:
        scaler_X = joblib.load(scaler_x_path)
        print(f"Scaler X loaded successfully from {scaler_x_path}")
    except Exception as e:
        print(f"Error loading Scaler X from {scaler_x_path}: {e}")
else:
    print(f"Error: No latest Scaler X file found with prefix '{SCALER_X_PREFIX}' in {MODEL_DIR}. Cannot preprocess input.")


if latest_scaler_Y_file:
    scaler_y_path = os.path.join(MODEL_DIR, latest_scaler_Y_file)
    try:
        scaler_y = joblib.load(scaler_y_path)
        print(f"Scaler y loaded successfully from {scaler_y_path}")
    except Exception as e:
        print(f"Error loading Scaler y from {scaler_y_path}: {e}")
else:
     print(f"Error: No latest Scaler y file found with prefix '{SCALER_Y_PREFIX}' in {MODEL_DIR}. Cannot inverse transform output.")


# --- Determine Input Features (excluding targets and suitability) ---
# This requires loading the data schema or inferring from loaded data/insights.
# Since we load the data initially to generate insights, we can use its columns.
input_features = []
# Use the loaded df to identify input features if available
if not df.empty:
    input_features = [col for col in df.columns if col not in TARGET_COLS and col != 'Suitable_for_Cancer_Treatment']
    print(f"Identified input features from loaded data: {input_features}")
else:
    print("Warning: Data not loaded, cannot identify input features dynamically.")


# --- Function to plot Correlation Matrix ---
def plot_correlation_matrix(df, save_path):
    """
    Calculates and plots the correlation matrix for the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        save_path (str): The path to save the correlation matrix plot.
    """
    if df.empty:
        print("Cannot plot correlation matrix: DataFrame is empty.")
        return

    print("\nGenerating correlation matrix plot...")
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        print("Cannot plot correlation matrix: No numeric columns found in DataFrame.")
        return

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 10)) # Adjust figure size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of AuNP Synthesis Data')
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Correlation matrix plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving correlation matrix plot to {save_path}: {e}")

    # plt.show() # Uncomment to display the plot immediately


# --- Generate and save Correlation Matrix Plot ---
if not df.empty:
    correlation_plot_path = os.path.join(MODEL_DIR, f"{CORRELATION_PLOT_PREFIX}_{timestamp}{CORRELATION_PLOT_EXTENSION}")
    plot_correlation_matrix(df, correlation_plot_path)
else:
    print("Skipping correlation matrix plot generation due to empty DataFrame.")


# --- Function to get user input for a single sample ---
def get_user_input(input_features):
    """Prompts the user to enter values for each input feature."""
    print("\nPlease enter the synthesis parameters for a single sample:")
    sample_data = {}
    for feature in input_features:
        while True:
            try:
                value = float(input(f"Enter value for '{feature}': "))
                sample_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value.")
    return pd.Series(sample_data) # Return as a pandas Series


# --- Function to create prompt for LLM classification ---
def create_llm_classification_prompt(sample, feature_explanation, criteria):
    """Create a prompt for an LLM to classify if a sample is suitable for cancer treatment"""

    prompt = f"""Task: Determine if the gold nanoparticle (AuNP) synthesis parameters and resulting properties are suitable for cancer treatment.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{explanation_text} # Use the loaded explanation text

Sample to Evaluate:
"""

    # Add input features
    prompt += "## Input Parameters:\n"
    # Assuming sample contains all input features and target properties
    input_cols_in_sample = [col for col in sample.index if col in input_features]
    if input_cols_in_sample:
        for col in input_cols_in_sample:
            prompt += f"- {col}: {sample[col]:.4f}\n"
    else:
         prompt += "No input parameters provided in sample.\n"


    # Add output properties
    prompt += "\n## Resulting Properties:\n"
    output_cols_in_sample = [col for col in criteria.keys() if col in sample.index]
    if output_cols_in_sample:
        for col in output_cols_in_sample:
            prompt += f"- {col}: {sample[col]:.4f}\n"
    else:
         prompt += "No resulting properties provided in sample.\n"


    # Add the classification question
    prompt += """\nBased on the criteria for optimal cancer treatment, are these gold nanoparticle properties suitable?
Please analyze each property against the ideal ranges and explain your reasoning.
Conclude with a clear YES or NO classification."""

    return prompt

# --- Function to create prompt for LLM optimization ---
def generate_synthesis_optimization_prompt(sample, feature_explanation, criteria):
    """
    Generate a prompt for the LLM to suggest optimizations to the synthesis method
    based on the current parameters and desired outputs
    """
    unmet_criteria = []
    for col, spec in criteria.items():
        if col in sample.index:
            low, high = spec['ideal_range']
            value = sample[col]
            if value < low:
                unmet_criteria.append(f"{col} is too low ({value:.2f} vs. ideal minimum {low})")
            elif value > high:
                unmet_criteria.append(f"{col} is too high ({value:.2f} vs. ideal maximum {high})")
        else:
             unmet_criteria.append(f"Property '{col}' is missing from the sample data.")


    prompt = f"""Task: Suggest optimizations to the gold nanoparticle (AuNP) synthesis method to improve its suitability for cancer treatment.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{explanation_text} # Use the loaded explanation text

Current Synthesis Parameters and Results:
"""

    prompt += "## Input Parameters:\n"
    # Assuming sample contains all input features and target properties
    input_cols_in_sample = [col for col in sample.index if col in input_features]
    if input_cols_in_sample:
         for col in input_cols_in_sample:
            prompt += f"- {col}: {sample[col]:.4f}\\n"
    else:
         prompt += "No input parameters provided in sample.\\n"

    prompt += "\\n## Resulting Properties:\\n"
    output_cols_in_sample = [col for col in criteria.keys() if col in sample.index]
    if output_cols_in_sample:
         for col in output_cols_in_sample:
            prompt += f"- {{col}}: {{sample[col]:.4f}}\\n"
    else:
         prompt += "No resulting properties provided in sample.\\n"


    if unmet_criteria:
        prompt += "\\n## Criteria Not Met:\\n"
        for issue in unmet_criteria:
            prompt += f"- {{issue}}\\n"
    else:
         prompt += "\\n## All Criteria Met:\\n\\nThe current synthesis parameters appear suitable.\\n"


    prompt += "\\nBased on the feature relationships and current parameters, please suggest specific modifications to the synthesis method that would improve the properties to meet all criteria for cancer treatment.\\nProvide a step-by-step explanation of:\\n1. Which parameters should be adjusted and by how much\\n2. Expected impact on each target property\\n3. Scientific rationale for each suggested change\\n4. A revised synthesis protocol incorporating these changes"

    return prompt


# --- Function to call the Groq API directly ---
def call_groq_api(prompt, model=GROQ_MODEL_NAME, max_tokens=500, temperature=0.7, max_retries=5, initial_delay=1.0):
    """
    Calls the Groq API with the given prompt, implementing retry with exponential backoff.

    Args:
        prompt (str): The text prompt to send to the LLM.
        model (str): The Groq model name to use.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness.
        max_retries (int): Maximum number of times to retry on rate limit errors.
        initial_delay (float): Initial delay in seconds before the first retry.

    Returns:
        tuple: (generated_text, predicted_label) or (error_message, None)
       predicted_label is 1 for Suitable, 0 for Not Suitable, None on API error or ambiguous parsing.
    """
    if not GROQ_API_KEY:
        # Return a specific error message if the key is not loaded
        return "Error: GROQ_API_KEY environment variable not set. Please check your .env file and environment.", None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [
            # Optional system message to guide the model's persona
            # {"role": "system", "content": "You are an expert in gold nanoparticle synthesis and characterization."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Add other parameters here based on Groq documentation (e.g., top_p, stop)
    }

    retries = 0
    while retries < max_retries:
        try:
            print(f"Calling Groq API with model: {model} (Attempt {retries + 1}/{max_retries})...") # Indicate API call is happening
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()

            if 'choices' in response_data and response_data['choices']:
                generated_text = response_data['choices'][0].get('message', {}).get('content', '')

                # --- IMPROVED PARSING LOGIC ---
                predicted_label = None # Default to None (ambiguous/not parsed)
                # Convert response to uppercase for case-insensitive matching
                upper_text = generated_text.upper()

                # Check for clear YES/NO conclusion line
                conclusion_match = re.search(r"CONCLUSION:\s*(YES|NO)", upper_text)
                if conclusion_match:
                    if conclusion_match.group(1) == "YES":
                        predicted_label = 1
                    else: # Must be NO
                        predicted_label = 0
                else:
                    # If no clear CONCLUSION line, look for "SUITABLE" or "NOT SUITABLE"
                    # Prioritize "NOT SUITABLE" if both are present or if the overall tone is negative
                    if "NOT SUITABLE" in upper_text:
                        predicted_label = 0 # Not Suitable
                    elif "SUITABLE" in upper_text:
                        predicted_label = 1 # Suitable
                    # If neither is found, predicted_label remains None


                return generated_text.strip(), predicted_label

            else:
                print("Warning: Groq API call successful but received no 'choices' in the response.")
                return "Could not generate response from Groq API.", None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retries += 1
                wait_time = initial_delay * (2 ** (retries - 1)) # Exponential backoff
                # Optionally, check 'Retry-After' header if available
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        pass # Stick with exponential backoff if header is not a simple integer

                print(f"Rate limit hit (429). Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # Re-raise other HTTP errors
                print(f"Error making Groq API request: {e}")
                if response is not None:
                    print(f"Status Code: {response.status_code}")
                    try:
                        print(f"Response Body: {response.json()}")
                    except json.JSONDecodeError:
                        print(f"Response Body: {response.text}")
                return f"Failed to get response from Groq API. Error: {e}", None

        except requests.exceptions.RequestException as e:
            # Handle other request errors (connection issues, etc.)
            print(f"Error making Groq API request: {e}")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                try:
                    print(f"Response Body: {response.json()}")
                except json.JSONDecodeError:
                    print(f"Response Body: {response.text}")
            return f"Failed to get response from Groq API. Error: {e}", None

        except json.JSONDecodeError:
            # Handle errors if the response is not valid JSON
            print("Error: Could not decode JSON response from Groq API.")
            if response is not None:
                 print(f"Received text: {response.text}")
            return "Failed to parse Groq API response.", None

        except Exception as e:
            # Handle any other unexpected errors
            print(f"An unexpected error occurred during Groq API call: {e}")
            return "An unexpected error occurred during API call.", None

    # If loop finishes, max retries were reached
    return f"Error: Max retries ({max_retries}) reached after hitting rate limit.", None


# --- Main execution flow ---
if __name__ == '__main__':
    # Check if model and scalers are loaded and input features are identified
    if ml_model is None or scaler_X is None or scaler_y is None:
        print("\nCannot proceed with prediction or LLM analysis without loaded ML model and scalers.")
    elif not input_features:
         print("\nCannot proceed with prediction or LLM analysis without identified input features.")
    else:
        # Get user input for a single sample
        user_sample_input = get_user_input(input_features)

        # Preprocess the input (scale X)
        # The scaler expects a 2D array (number of samples, number of features)
        user_sample_scaled = scaler_X.transform(user_sample_input.values.reshape(1, -1))

        # Make prediction using the ML model
        try:
            # Keras model predict returns a numpy array
            predicted_outputs_scaled = ml_model.predict(user_sample_scaled)

            # Inverse transform the predicted outputs using scaler_y
            predicted_outputs_unscaled = scaler_y.inverse_transform(predicted_outputs_scaled)

            # Extract the first (and only) sample's predictions
            predicted_output_values = predicted_outputs_unscaled[0]

            # Create a Series for the predicted outputs
            predicted_output_series = pd.Series(predicted_output_values, index=TARGET_COLS)

            print("\n--- Predicted AuNP Properties ---")
            for prop, value in predicted_output_series.items():
                print(f"{prop}: {value:.4f}")
            print("-" * 30)

            # Combine user input parameters and predicted outputs into a single sample Series
            # This combined sample will be used for LLM prompts
            # Ensure the order of columns matches what the LLM prompt functions expect
            combined_sample = pd.concat([user_sample_input, predicted_output_series])

            # --- Call Groq API for Classification ---
            print("\n--- Requesting Suitability Classification from Groq API ---")
            classification_prompt = create_llm_classification_prompt(combined_sample, explanation_text, suitability_criteria)
            classification_response, predicted_suitability_label = call_groq_api(
                classification_prompt, model=GROQ_MODEL_NAME, max_tokens=500
            )

            print("\n--- Suitability Classification Result ---")
            print(f"Predicted Suitable: {'Yes' if predicted_suitability_label == 1 else ('No' if predicted_suitability_label == 0 else 'Could not determine')}")
            print("\nGroq API Explanation:")
            print(classification_response)
            print("-" * 30)

            # Add a delay before the next API call
            if API_CALL_DELAY > 0:
                print(f"Waiting for {API_CALL_DELAY} seconds before requesting optimization...")
                time.sleep(API_CALL_DELAY)


            # --- Call Groq API for Optimization Suggestions ---
            print("\n--- Requesting Optimization Suggestions from Groq API ---")
            # Need feature_insights for generating the optimization prompt
            if not feature_insights:
                 print("Warning: Feature insights not loaded, optimization prompt may be less specific.")

            optimization_prompt = generate_synthesis_optimization_prompt(
                combined_sample, explanation_text, suitability_criteria
            )
            optimization_suggestion, _ = call_groq_api(
                optimization_prompt, model=GROQ_MODEL_NAME, max_tokens=1000 # More tokens for suggestions
            )

            print("\n--- Optimization Suggestion from Groq API ---")
            print(optimization_suggestion)
            print("-" * 30)


        except Exception as e:
            print(f"\nAn error occurred during prediction or LLM calls: {e}")
            # Provide more specific error messages if possible
            if "shape" in str(e) and ("scaler" in str(e) or "model" in str(e)):
                 print("Hint: Check if the input features provided match the number of features the scalers/model were fitted/trained on.")
                 print("Also ensure the scaler_X and scaler_y files match the model they are used with.")
            else:
                 print("Please check your model file, scaler files, and input data format.")


else:
    print("Skipping prediction and LLM analysis because input features could not be identified or data loading failed.")