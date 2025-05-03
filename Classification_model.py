import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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

# --- Load environment variables from .env file ---
# Assumes your .env file is named GROQ_API_KEY.env and is in the same directory
# as the script, or you can provide a specific path.
load_dotenv(dotenv_path='GROQ_API_KEY.env') # Load variables from the specified .env file

# Configuration
# NOTE: Update this path to your actual file location
DATA_FILE_PATH = r"C:\Users\mechi\Documents\GitHub\LLM_Transformer_Model\aunp_synthesis_cancer_treatment_v3_transformed.csv"
MODEL_DIR = "saved_models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Groq API Configuration (Direct Call)
# IMPORTANT: The API key is now loaded from the .env file via load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # This will now get the key loaded by load_dotenv
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Choose a suitable Groq model. Check Groq's documentation for available models.
GROQ_MODEL_NAME = "llama3-8b-8192" # Or "llama3-70b-8192" if you have access and credits

# Output file for results
RESULTS_FILE_PATH = os.path.join(MODEL_DIR, f"llm_classification_results_{time.strftime('%Y%m%d_%H%M%S')}.txt")

# Small delay between API calls for different samples to help with rate limits
API_CALL_DELAY = 1.0 # seconds


# Target columns from the original model (these are the properties we evaluate suitability on)
TARGET_COLS = ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%',
               'Targeting_Efficiency_%', 'Cytotoxicity_%']

# Create output directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# Generate a timestamp for saving files (used for saving explanation/insights)
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Load Data
print(f"Loading data from: {DATA_FILE_PATH}")
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE_PATH}")
    df = pd.DataFrame()
    print("Proceeding with empty DataFrame. Some features may not work.")

# Proceed only if DataFrame is not empty
if not df.empty:
    # Feature Analysis - Generate feature descriptions and relationships
    def analyze_features(df, target_cols):
        """Analyze feature relationships and generate descriptions for LLM context"""
        feature_cols = [col for col in df.columns if col not in target_cols]

        # Calculate correlations
        numeric_df = df.select_dtypes(include=np.number)
        correlation_matrix = numeric_df.corr()

        # Feature statistics
        feature_stats = {}
        for col in numeric_df.columns:
             feature_stats[col] = {
                 "mean": float(numeric_df[col].mean()),
                 "median": float(numeric_df[col].median()),
                 "min": float(numeric_df[col].min()),
                 "max": float(numeric_df[col].max()),
                 "std": float(numeric_df[col].std())
             }

        # Find top correlations for each target
        target_correlations = {}
        for target in target_cols:
            if target in correlation_matrix.columns:
                target_corr = correlation_matrix[target].drop([t for t in target_cols if t in correlation_matrix.index], errors='ignore')
                top_pos = target_corr.nlargest(5)
                top_neg = target_corr.nsmallest(5)

                target_correlations[target] = {
                    "positive": {feature: float(corr) for feature, corr in top_pos.items()},
                    "negative": {feature: float(corr) for feature, corr in top_neg.items()}
                }
            else:
                 target_correlations[target] = {"positive": {}, "negative": {}}
                 print(f"Warning: Target column '{target}' not found in numeric data for correlation analysis.")

        # Convert correlation matrix to a simple dictionary
        corr_dict = {}
        for col in correlation_matrix.columns:
            corr_dict[col] = {other_col: float(correlation_matrix.loc[col, other_col])
                              for other_col in correlation_matrix.columns}

        feature_info = {
            "feature_stats": feature_stats,
            "target_correlations": target_correlations,
            "correlation_matrix": corr_dict
        }

        return feature_info

    # Define cancer treatment suitability criteria (kept here as it's used for label generation and prompts)
    suitability_criteria = {
        'Particle_Size_nm': {
            'ideal_range': (40, 100),
            'description': "Optimal particle size range for gold nanoparticles in cancer treatment. Particles should be large enough to carry drug payload but small enough to penetrate tumor tissue via the EPR effect."
        },
        'Zeta_Potential_mV': {
            'ideal_range': (-30, -5),
            'description': "Zeta potential indicates surface charge and stability. Slightly negative values promote stability while facilitating cellular uptake."
        },
        'Drug_Loading_Efficiency_%': {
            'ideal_range': (70, 100),
            'description': "Indicates how efficiently the drug is loaded onto nanoparticles. Higher values mean more effective drug delivery."
        },
        'Targeting_Efficiency_%': {
            'ideal_range': (75, 100),
            'description': "Measures how well nanoparticles target cancer cells. Higher values indicate better specificity for cancer cells."
        },
        'Cytotoxicity_%': {
            'ideal_range': (70, 90),
            'description': "Indicates toxicity to cancer cells. Should be high enough to effectively kill cancer cells but not excessively toxic."
        }
    }

    # Function to generate binary suitability labels based on criteria
    def generate_suitability_labels(df, criteria):
        """Generate binary labels indicating if a sample meets all suitability criteria"""
        mask = pd.Series(True, index=df.index)
        for col, spec in criteria.items():
            if col in df.columns:
                low, high = spec['ideal_range']
                col_mask = (df[col] >= low) & (df[col] <= high)
                mask = mask & col_mask
            else:
                print(f"Warning: Criteria column '{col}' not found in DataFrame for label generation.")
                return pd.Series(0, index=df.index)
        return mask.astype(int)

    # Generate suitability labels
    df['Suitable_for_Cancer_Treatment'] = generate_suitability_labels(df, suitability_criteria)

    # Analyze the class balance
    suitable_count = df['Suitable_for_Cancer_Treatment'].sum()
    total_count = len(df)
    print(f"Suitable samples: {suitable_count} ({suitable_count/total_count*100:.2f}%)")
    print(f"Unsuitable samples: {total_count - suitable_count} ({(total_count - suitable_count)/total_count*100:.2f}%)")

    # Analyze features and save insights (still useful for prompt generation)
    feature_insights = analyze_features(df, TARGET_COLS)
    # We don't strictly need to save these to file if not running the API,
    # but keeping it for consistency or potential future use.
    feature_insights_path = os.path.join(MODEL_DIR, f"feature_insights_{timestamp}.json")
    with open(feature_insights_path, 'w') as f:
        json.dump(feature_insights, f, indent=2)
    print(f"Feature insights saved to: {feature_insights_path}")


    # Generate a consolidated explanation of feature relationships (still useful for prompt generation)
    def generate_feature_explanation(feature_insights, criteria):
        """Generate a text explanation of feature relationships for LLM context"""
        explanation = "# Gold Nanoparticle (AuNP) Synthesis for Cancer Treatment\n\n"
        explanation += "## Optimal Output Criteria for Cancer Treatment\n\n"
        for target, spec in criteria.items():
            low, high = spec['ideal_range']
            explanation += f"- **{target}**: Should be between {low} and {high}. {spec['description']}\n"
        explanation += "\n## Key Feature Relationships\n\n"
        for target, corr_dict in feature_insights['target_correlations'].items():
            explanation += f"### Factors influencing {target}:\n\n"
            explanation += "**Features that tend to increase this value:**\n"
            if corr_dict['positive']:
                for feature, corr in corr_dict['positive'].items():
                    explanation += f"- {feature} (correlation: {corr:.3f})\n"
            else:
                 explanation += "None significant.\n"
            explanation += "\n**Features that tend to decrease this value:**\n"
            if corr_dict['negative']:
                 for feature, corr in corr_dict['negative'].items():
                    explanation += f"- {feature} (correlation: {corr:.3f})\n"
            else:
                 explanation += "None significant.\n"
            explanation += "\n"
        explanation += "## Feature Statistics (for context)\n\n"
        input_features_stats = {k: v for k, v in feature_insights['feature_stats'].items()
                                if k not in criteria and k != 'Suitable_for_Cancer_Treatment'}
        if input_features_stats:
            for feature, stats in input_features_stats.items():
                explanation += f"- **{feature}**: Range [{stats['min']:.2f} to {stats['max']:.2f}], Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n"
        else:
             explanation += "No input feature statistics available.\n"
        return explanation

    # Generate the explanation text
    explanation_text = generate_feature_explanation(feature_insights, suitability_criteria)
    # We don't strictly need to save this to file if not running the API,
    # but keeping it for consistency or potential future use.
    explanation_path = os.path.join(MODEL_DIR, f"aunp_synthesis_explanation_{timestamp}.md")
    with open(explanation_path, 'w') as f:
        f.write(explanation_text)
    print(f"Synthesis explanation saved to: {explanation_path}")


    # Prepare data for LLM classification (keeping target columns in X for sending to API)
    input_features = [col for col in df.columns if col not in TARGET_COLS and col != 'Suitable_for_Cancer_Treatment']
    if not input_features:
         print("Error: No input features found after excluding target and suitability columns.")
         X = pd.DataFrame()
         y = pd.Series()
    else:
        X = df[input_features + TARGET_COLS] # Keep target columns in X to send to API
        y = df['Suitable_for_Cancer_Treatment']


    # Proceed only if there's data to split
    if not X.empty:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        print(f"Train set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # --- Function to create prompt for LLM classification ---
        def create_llm_classification_prompt(sample, feature_explanation, criteria):
            """Create a prompt for an LLM to classify if a sample is suitable for cancer treatment"""

            prompt = f"""Task: Determine if the gold nanoparticle (AuNP) synthesis parameters and resulting properties are suitable for cancer treatment.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{feature_explanation}

Sample to Evaluate:
"""

            # Add input features
            prompt += "## Input Parameters:\n"
            input_cols = [col for col in sample.index if col not in criteria]
            if input_cols:
                for col in input_cols:
                    prompt += f"- {col}: {sample[col]:.4f}\n"
            else:
                 prompt += "No input parameters provided.\n"


            # Add output properties
            prompt += "\n## Resulting Properties:\n"
            output_cols = [col for col in criteria.keys() if col in sample.index]
            if output_cols:
                for col in output_cols:
                    prompt += f"- {col}: {sample[col]:.4f}\n"
            else:
                 prompt += "No resulting properties provided.\n"


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
{feature_explanation}

Current Synthesis Parameters and Results:
"""

            prompt += "## Input Parameters:\n"
            input_cols = [col for col in sample.index if col not in criteria]
            if input_cols:
                 for col in input_cols:
                    prompt += f"- {col}: {sample[col]:.4f}\\n"
            else:
                 prompt += "No input parameters provided.\\n"

            prompt += "\\n## Resulting Properties:\\n"
            output_cols = [col for col in criteria.keys() if col in sample.index]
            if output_cols:
                 for col in output_cols:
                    prompt += f"- {{col}}: {{sample[col]:.4f}}\\n"
            else:
                 prompt += "No resulting properties provided.\\n"


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


        # --- Test LLM Classification Directly ---
        def test_llm_classification_direct(X_test, y_test, n_samples=5):
            """Test LLM classification by calling Groq API directly on a few samples."""

            # Check if API key is loaded before proceeding
            if not GROQ_API_KEY:
                 print("\n--- Skipping LLM Classification Test (Direct API) due to missing GROQ_API_KEY ---")
                 print("Please ensure GROQ_API_KEY is set in your GROQ_API_KEY.env file and the file is in the correct location.")
                 return [] # Skip test if API key is not set

            print("\n--- Running LLM Classification Test (Direct Groq API) ---")

            # Randomly sample a few test instances
            sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
            sampled_X_test = X_test.iloc[sample_indices]
            sampled_y_test = y_test.iloc[sample_indices]

            results = []
            for idx in range(len(sampled_X_test)):
                sample = sampled_X_test.iloc[idx]
                true_label = sampled_y_test.iloc[idx]
                original_index = sampled_X_test.index[idx] # Get original index

                # Create the prompt
                prompt = create_llm_classification_prompt(sample, explanation_text, suitability_criteria)

                # Call Groq API directly (with retry logic)
                llm_response, predicted_label = call_groq_api(prompt, model=GROQ_MODEL_NAME, max_tokens=500)

                # Handle cases where parsing failed or API call failed
                if predicted_label is None:
                    # Default to Not Suitable if parsing failed or API call failed
                    predicted_label_value = 0
                    prediction_status = "Could not parse/API Error"
                else:
                    predicted_label_value = predicted_label
                    prediction_status = 'CORRECT' if true_label == predicted_label_value else 'INCORRECT'

                results.append({
                    'sample_idx': original_index,
                    'true_label': true_label,
                    'predicted_label': predicted_label_value,
                    'prediction_status': prediction_status,
                    'llm_response': llm_response # Store the full LLM response
                })

                # Add a small delay after each API call for a sample
                if API_CALL_DELAY > 0:
                    print(f"Waiting for {API_CALL_DELAY} seconds before next sample...")
                    time.sleep(API_CALL_DELAY)


            return results

        # Run a small test using the direct API call
        llm_direct_test_results = test_llm_classification_direct(X_test, y_test, n_samples=5)

        # --- Save results to a file ---
        with open(RESULTS_FILE_PATH, 'w') as f:
            f.write(f"--- LLM Classification Test Results (Direct Groq API) ---\n\n")
            if llm_direct_test_results:
                for i, result in enumerate(llm_direct_test_results):
                    f.write(f"Sample {i+1} (Original Index: {result['sample_idx']}):\n")
                    f.write(f"  True label: {'Suitable' if result['true_label'] == 1 else 'Not suitable'}\n")
                    f.write(f"  Predicted label: {'Suitable' if result['predicted_label'] == 1 else 'Not suitable'}\n")
                    f.write(f"  Prediction was: {result['prediction_status']}\n")
                    f.write(f"  LLM Response:\n")
                    # Indent the LLM response for readability in the file
                    indented_response = result['llm_response'].replace('\n', '\n    ')
                    f.write(f"    {indented_response}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("LLM classification test was skipped or returned no results (API key might be missing).\n")

            # --- Example of how to get Optimization Suggestions Directly ---
            # Find an unsuitable sample from the test set to get optimization suggestions for
            unsuitable_samples = X_test[y_test == 0]

            f.write("\n--- Optimization Suggestions (Direct Groq API) ---\n\n")
            if not unsuitable_samples.empty:
                 sample_to_optimize = unsuitable_samples.iloc[0] # Take the first unsuitable sample

                 print("\n--- Requesting Optimization Suggestions (Direct Groq API) ---")
                 # Create optimization prompt
                 optimization_prompt = generate_synthesis_optimization_prompt(
                     sample_to_optimize, explanation_text, suitability_criteria
                 )

                 # Call Groq API directly for optimization (with retry logic)
                 optimization_suggestion, _ = call_groq_api(
                     optimization_prompt, model=GROQ_MODEL_NAME, max_tokens=1000 # More tokens for suggestions
                 )

                 f.write("Optimization Suggestion for Sample (Original Index: {}):\n".format(unsuitable_samples.index[0]))
                 # Indent the optimization suggestion for readability
                 indented_suggestion = optimization_suggestion.replace('\n', '\n  ')
                 f.write(f"  {indented_suggestion}\n")
                 print("\nOptimization Suggestion saved to file.")

            else:
                 f.write("No unsuitable samples found in the test set to demonstrate optimization.\n")
                 print("\nNo unsuitable samples found to demonstrate optimization.")

        print(f"\nResults saved to: {RESULTS_FILE_PATH}")


    else:
        print("Skipping data split, LLM classification test, and optimization example due to empty DataFrame.")

else:
    print("Skipping data analysis and subsequent steps due to empty DataFrame.")

