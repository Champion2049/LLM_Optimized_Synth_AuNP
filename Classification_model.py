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

# For LLM integration
import requests
from typing import List, Dict, Tuple, Union, Optional
import re

# Configuration
# NOTE: Update this path to your actual file location
DATA_FILE_PATH = r"C:\Users\Chirayu\Desktop\Coding\IMI\aunp_synthesis_cancer_treatment_v3_transformed.csv"
MODEL_DIR = "saved_models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target columns from the original model (these are the properties we evaluate suitability on)
TARGET_COLS = ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%',
               'Targeting_Efficiency_%', 'Cytotoxicity_%']

# Create output directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# Generate a timestamp for saving files
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Load Data
print(f"Loading data from: {DATA_FILE_PATH}")
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE_PATH}")
    # Ensure the script can continue for demonstration purposes even if data isn't loaded
    # In a real application, you might want to exit or handle this differently
    df = pd.DataFrame() # Create an empty DataFrame
    print("Proceeding with empty DataFrame. Some features may not work.")

# Proceed only if DataFrame is not empty
if not df.empty:
    # Feature Analysis - Generate feature descriptions and relationships
    def analyze_features(df, target_cols):
        """Analyze feature relationships and generate descriptions for LLM context"""
        feature_cols = [col for col in df.columns if col not in target_cols]

        # Calculate correlations
        # Ensure only numeric columns are used for correlation
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
                # Get top 5 positive and negative correlations with features
                # Exclude target columns from the features list
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


        # Convert correlation matrix to a simple dictionary (avoiding circular references)
        corr_dict = {}
        for col in correlation_matrix.columns:
            corr_dict[col] = {other_col: float(correlation_matrix.loc[col, other_col])
                              for other_col in correlation_matrix.columns}

        # Generate feature description dictionary
        feature_info = {
            "feature_stats": feature_stats,
            "target_correlations": target_correlations,
            "correlation_matrix": corr_dict
        }

        return feature_info

    # Define cancer treatment suitability criteria
    suitability_criteria = {
        'Particle_Size_nm': {
            'ideal_range': (40, 100),  # Gold nanoparticles typically 40-100nm for EPR effect
            'description': "Optimal particle size range for gold nanoparticles in cancer treatment. Particles should be large enough to carry drug payload but small enough to penetrate tumor tissue via the EPR effect."
        },
        'Zeta_Potential_mV': {
            'ideal_range': (-30, -5),  # Slightly negative for stability and cellular uptake
            'description': "Zeta potential indicates surface charge and stability. Slightly negative values promote stability while facilitating cellular uptake."
        },
        'Drug_Loading_Efficiency_%': {
            'ideal_range': (70, 100),  # Higher is better
            'description': "Indicates how efficiently the drug is loaded onto nanoparticles. Higher values mean more effective drug delivery."
        },
        'Targeting_Efficiency_%': {
            'ideal_range': (75, 100),  # Higher is better
            'description': "Measures how well nanoparticles target cancer cells. Higher values indicate better specificity for cancer cells."
        },
        'Cytotoxicity_%': {
            'ideal_range': (70, 90),  # High enough to kill cancer cells but not too toxic
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
                # Check if values are within the ideal range
                col_mask = (df[col] >= low) & (df[col] <= high)
                mask = mask & col_mask
            else:
                print(f"Warning: Criteria column '{col}' not found in DataFrame for label generation.")
                # If a criteria column is missing, no sample can meet all criteria
                return pd.Series(0, index=df.index)


        return mask.astype(int)  # Convert boolean to 0/1

    # Generate suitability labels
    df['Suitable_for_Cancer_Treatment'] = generate_suitability_labels(df, suitability_criteria)

    # Analyze the class balance
    suitable_count = df['Suitable_for_Cancer_Treatment'].sum()
    total_count = len(df)
    print(f"Suitable samples: {suitable_count} ({suitable_count/total_count*100:.2f}%)")
    print(f"Unsuitable samples: {total_count - suitable_count} ({(total_count - suitable_count)/total_count*100:.2f}%)")

    # Analyze features and save insights
    feature_insights = analyze_features(df, TARGET_COLS)
    feature_insights_path = os.path.join(MODEL_DIR, f"feature_insights_{timestamp}.json")
    with open(feature_insights_path, 'w') as f:
        json.dump(feature_insights, f, indent=2)
    print(f"Feature insights saved to: {feature_insights_path}")

    # Generate a consolidated explanation of feature relationships
    def generate_feature_explanation(feature_insights, criteria):
        """Generate a text explanation of feature relationships for LLM context"""
        explanation = "# Gold Nanoparticle (AuNP) Synthesis for Cancer Treatment\n\n"

        # Add criteria explanation
        explanation += "## Optimal Output Criteria for Cancer Treatment\n\n"
        for target, spec in criteria.items():
            low, high = spec['ideal_range']
            explanation += f"- **{target}**: Should be between {low} and {high}. {spec['description']}\n"

        explanation += "\n## Key Feature Relationships\n\n"

        # Add target correlations
        for target, corr_dict in feature_insights['target_correlations'].items():
            explanation += f"### Factors influencing {target}:\n\n"

            # Positive correlations
            explanation += "**Features that tend to increase this value:**\n"
            if corr_dict['positive']:
                for feature, corr in corr_dict['positive'].items():
                    explanation += f"- {feature} (correlation: {corr:.3f})\n"
            else:
                 explanation += "None significant.\n"


            # Negative correlations
            explanation += "\n**Features that tend to decrease this value:**\n"
            if corr_dict['negative']:
                 for feature, corr in corr_dict['negative'].items():
                    explanation += f"- {feature} (correlation: {corr:.3f})\n"
            else:
                 explanation += "None significant.\n"


            explanation += "\n"

        # Add feature statistics summary for context
        explanation += "## Feature Statistics (for context)\n\n"
        # Filter out target columns and the generated suitability column
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
    explanation_path = os.path.join(MODEL_DIR, f"aunp_synthesis_explanation_{timestamp}.md")
    with open(explanation_path, 'w') as f:
        f.write(explanation_text)
    print(f"Synthesis explanation saved to: {explanation_path}")

    # Prepare data for LLM training/evaluation
    # Drop target columns as they are part of the evaluation criteria, not input features for classification
    input_features = [col for col in df.columns if col not in TARGET_COLS and col != 'Suitable_for_Cancer_Treatment']
    # Ensure input features exist
    if not input_features:
         print("Error: No input features found after excluding target and suitability columns.")
         X = pd.DataFrame() # Create empty DataFrame
         y = pd.Series()    # Create empty Series
    else:
        X = df[input_features + TARGET_COLS] # Keep target columns in X for the prompt
        y = df['Suitable_for_Cancer_Treatment']


    # Proceed only if there's data to split
    if not X.empty:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        print(f"Train set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Create sample prompts for LLM classification
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

        # Function to simulate LLM API call (replace with actual LLM API integration)
        def simulate_llm_classification(prompt, true_label, criteria, sample):
            """Simulate an LLM classification (can be replaced with actual API call)"""

            # For simulation, we'll just check the criteria programmatically
            meets_criteria = True
            reasons = []

            for col, spec in criteria.items():
                if col in sample.index:
                    low, high = spec['ideal_range']
                    value = sample[col]

                    if value < low or value > high:
                        meets_criteria = False
                        if value < low:
                            reasons.append(f"{col} is {value:.2f}, which is below the ideal range of {low}-{high}")
                        else:
                            reasons.append(f"{col} is {value:.2f}, which is above the ideal range of {low}-{high}")
                    else:
                        reasons.append(f"{col} is {value:.2f}, which is within the ideal range of {low}-{high}")
                else:
                     reasons.append(f"Property '{col}' not found in sample data.")
                     meets_criteria = False # Cannot meet criteria if a property is missing


            # Generate a simulated LLM response
            response = "Analysis of Gold Nanoparticle Properties for Cancer Treatment:\n\n"

            for reason in reasons:
                response += f"- {reason}\n"

            response += f"\nConclusion: {'YES' if meets_criteria else 'NO'}, these properties are {'suitable' if meets_criteria else 'not suitable'} for cancer treatment."

            # In a real implementation, this would be the returned LLM response
            predicted_label = 1 if meets_criteria else 0

            return response, predicted_label

        # Test the LLM classification on a few samples
        def test_llm_classification(X_test, y_test, n_samples=5):
            """Test the LLM classification approach on a few samples"""

            # Randomly sample a few test instances
            sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

            results = []
            for idx in sample_indices:
                sample = X_test.iloc[idx]
                true_label = y_test.iloc[idx]

                # Create the prompt
                prompt = create_llm_classification_prompt(sample, explanation_text, suitability_criteria)

                # Get LLM response (simulated)
                llm_response, predicted_label = simulate_llm_classification(
                    prompt, true_label, suitability_criteria, sample
                )

                results.append({
                    'sample_idx': X_test.index[idx], # Use original dataframe index
                    'prompt': prompt,
                    'llm_response': llm_response,
                    'true_label': true_label,
                    'predicted_label': predicted_label
                })

            return results

        # Run a small test
        llm_test_results = test_llm_classification(X_test, y_test, n_samples=5)

        # Display results of the small test
        print("\n--- LLM Classification Test Results ---")
        for i, result in enumerate(llm_test_results):
            print(f"\nSample Original Index: {result['sample_idx']}")
            print(f"True label: {'Suitable' if result['true_label'] == 1 else 'Not suitable'}")
            print(f"Predicted label: {'Suitable' if result['predicted_label'] == 1 else 'Not suitable'}")
            print(f"Prediction was: {'CORRECT' if result['true_label'] == result['predicted_label'] else 'INCORRECT'}")
            print("\nLLM Response:")
            print(result['llm_response'])
            print("-" * 80)

        # In a real-world implementation, you'd use an actual LLM API here
        # For demonstration, we'll create a function to integrate with an LLM API
        def classify_with_llm(samples, feature_explanation, criteria, llm_api_url="your_llm_api_endpoint"):
            """Classify samples using an LLM API (implementation dependent on the LLM service)"""

            results = []
            for i, sample in samples.iterrows():
                # Create the prompt
                prompt = create_llm_classification_prompt(sample, feature_explanation, criteria)

                # In a real implementation, this would make an API call to an LLM service
                # Here we'll use our simulation for demonstration
                llm_response, predicted_label = simulate_llm_classification(
                    prompt, None, criteria, sample
                )

                results.append({
                    'sample_idx': i,
                    'prompt': prompt,
                    'llm_response': llm_response,
                    'predicted_label': predicted_label
                })

            return results

        # Function to create a Flask API for the LLM classifier
        def create_llm_classifier_api():
            """
            Create a Flask API for deploying the LLM classifier

            This function defines a Flask application that exposes endpoints for:
            1. Classifying new AuNP synthesis parameters
            2. Getting explanations for specific features

            In a real implementation, this would be expanded into a proper Flask application
            """

            # Embedding the Flask code directly as a string to be saved to a file
            # This avoids requiring Flask to be installed to *generate* the file
            # but Flask will be needed to *run* the generated file.
            api_code = f"""
from flask import Flask, request, jsonify
import pandas as pd
import json
import os # Import os to handle file paths

app = Flask(__name__)

# Load the explanation text and criteria
# Assuming the script generating this API saves these files in the same directory
MODEL_DIR = "{MODEL_DIR}" # Use the defined model directory
explanation_path = os.path.join(MODEL_DIR, "aunp_synthesis_explanation_{timestamp}.md")

try:
    with open(explanation_path, 'r') as f:
        explanation_text = f.read()
    print(f"Loaded explanation text from {{explanation_path}}")
except FileNotFoundError:
    explanation_text = "Explanation text not found."
    print(f"Error: Explanation text file not found at {{explanation_path}}")


# Define suitability criteria (should ideally be loaded from a config file in production)
# Embedding the criteria directly as a string representation of the dictionary
suitability_criteria_str = '''{json.dumps(suitability_criteria, indent=None)}''' # Use json.dumps for robust string representation
suitability_criteria = json.loads(suitability_criteria_str)
print("Loaded suitability criteria.")

# Function to create a prompt for an LLM to classify if a sample is suitable for cancer treatment
def create_llm_classification_prompt(sample, feature_explanation, criteria):
    prompt = f\"\"\"Task: Determine if the gold nanoparticle (AuNP) synthesis parameters and resulting properties are suitable for cancer treatment.

Background Knowledge on AuNP Synthesis for Cancer Treatment:
{{feature_explanation}}

Sample to Evaluate:
\"\"\"

    # Add input features
    prompt += "## Input Parameters:\\n"
    input_cols = [col for col in sample.index if col not in criteria]
    if input_cols:
        for col in input_cols:
            prompt += f"- {{col}}: {{sample[col]:.4f}}\\n"
    else:
         prompt += "No input parameters provided.\\n"


    # Add output properties
    prompt += "\\n## Resulting Properties:\\n"
    output_cols = [col for col in criteria.keys() if col in sample.index]
    if output_cols:
        for col in output_cols:
            prompt += f"- {{col}}: {{sample[col]:.4f}}\\n"
    else:
         prompt += "No resulting properties provided.\\n"


    # Add the classification question
    prompt += \"\"\"\\nBased on the criteria for optimal cancer treatment, are these gold nanoparticle properties suitable?
Please analyze each property against the ideal ranges and explain your reasoning.
Conclude with a clear YES or NO classification.\"\"\"

    return prompt

# Function to call your LLM service (replace with your actual implementation)
# THIS IS A SIMULATED LLM CALL FOR DEMONSTRATION
def call_llm_service(prompt, sample, criteria):
    # For demonstration purposes, this is a simple check against criteria
    # In production, this would be a call to your LLM API (e.g., OpenAI, Gemini, etc.)

    meets_criteria = True
    reasons = []

    for col, spec in criteria.items():
        if col in sample.index:
            low, high = spec['ideal_range']
            value = sample[col]

            if value < low or value > high:
                meets_criteria = False
                if value < low:
                    reasons.append(f"{{col}} is {{value:.2f}}, which is below the ideal range of {{low}}-{{high}}")
                else:
                    reasons.append(f"{{col}} is {{value:.2f}}, which is above the ideal range of {{low}}-{{high}}")
            else:
                reasons.append(f"{{col}} is {{value:.2f}}, which is within the ideal range of {{low}}-{{high}}")
        else:
             reasons.append(f"Property '{{col}}' not found in sample data.")
             meets_criteria = False # Cannot meet criteria if a property is missing


    # Generate a simulated LLM response
    response = "Analysis of Gold Nanoparticle Properties for Cancer Treatment:\\n\\n"

    for reason in reasons:
        response += f"- {{reason}}\\n"

    response += f"\\nConclusion: {{'YES' if meets_criteria else 'NO'}}, these properties are {{'suitable' if meets_criteria else 'not suitable'}} for cancer treatment."

    predicted_label = 1 if meets_criteria else 0

    return response, predicted_label

@app.route('/classify', methods=['POST'])
def classify():
    \"\"\"Endpoint to classify if AuNP synthesis parameters are suitable for cancer treatment\"\"\"
    try:
        data = request.json
        # Ensure data contains keys from TARGET_COLS + relevant input features
        required_keys = list(suitability_criteria.keys()) + [col for col in data.keys() if col not in suitability_criteria]
        sample_data = {{k: data.get(k, None) for k in required_keys}}

        # Convert dict to pandas Series
        sample = pd.Series(sample_data).dropna() # Drop missing values

        if not all(col in sample.index for col in suitability_criteria.keys()):
             missing = [col for col in suitability_criteria.keys() if col not in sample.index]
             return jsonify({{'error': f"Missing required properties for classification: {{', '.join(missing)}}"}}), 400

        # Create prompt
        prompt = create_llm_classification_prompt(sample, explanation_text, suitability_criteria)

        # Call LLM service (simulated)
        llm_response, predicted_label = call_llm_service(prompt, sample, suitability_criteria)

        return jsonify({{
            'suitable': bool(predicted_label),
            'explanation': llm_response,
            'prompt': prompt # Optional: include prompt for debugging
        }})

    except Exception as e:
        # Log the error in a real application
        print(f"Error during classification: {{e}}")
        return jsonify({{'error': str(e)}}), 400

@app.route('/explain_features', methods=['GET'])
def explain_features():
    \"\"\"Endpoint to get feature relationship explanations\"\"\"
    return jsonify({{
        'explanation': explanation_text,
        'criteria': suitability_criteria
    }})

if __name__ == '__main__':
    # To run this API, save the code as a Python file (e.g., api.py)
    # and run 'python api.py' in your terminal within the correct directory.
    # Make sure you have Flask installed (`pip install Flask pandas`)
    # and the required data/explanation files in the '{MODEL_DIR}' directory.
    # debug=True is useful for development but should be False in production.
    app.run(debug=True)
"""
            return api_code

        # Save the API code
        api_code = create_llm_classifier_api()
        api_code_path = os.path.join(MODEL_DIR, f"llm_classifier_api_{timestamp}.py")
        with open(api_code_path, 'w') as f:
            f.write(api_code)
        print(f"LLM Classifier API code saved to: {api_code_path}")

        # Create a function to output synthesis method optimization suggestions
        def generate_synthesis_optimization_prompt(sample, feature_explanation, criteria):
            """
            Generate a prompt for the LLM to suggest optimizations to the synthesis method
            based on the current parameters and desired outputs
            """

            # Check which criteria are not met
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


            # Add unmet criteria
            if unmet_criteria:
                prompt += "\n## Criteria Not Met:\n"
                for issue in unmet_criteria:
                    prompt += f"- {issue}\n"
            else:
                 prompt += "\n## All Criteria Met:\n\nThe current synthesis parameters appear suitable.\n"


            # Add the optimization request
            prompt += """\nBased on the feature relationships and current parameters, please suggest specific modifications to the synthesis method that would improve the properties to meet all criteria for cancer treatment.
Provide a step-by-step explanation of:
1. Which parameters should be adjusted and by how much
2. Expected impact on each target property
3. Scientific rationale for each suggested change
4. A revised synthesis protocol incorporating these changes"""

            return prompt

        # Function to call LLM API for optimization suggestions
        def call_llm_for_optimization(prompt, api_url="your_llm_api_endpoint", api_key=None):
            """
            Call an LLM API to get optimization suggestions
            In a real implementation, this would call an actual LLM API service
            """
            # This is where you would implement your actual LLM API call
            # For example, using requests:
            """
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "prompt": prompt,
                "max_tokens": 1000, # Adjust as needed
                "temperature": 0.7 # Adjust for creativity vs. determinism
            }

            try:
                response = requests.post(api_url, headers=headers, json=data)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                return response.json().get("choices")[0].get("text") # Adjust based on your LLM API response format
            except requests.exceptions.RequestException as e:
                return f"Error calling LLM API: {e}"
            """

            # For now, we'll use our simulation function
            # Note: The simulation uses the sample and feature_insights directly,
            # which is different from how a real API would work (it would only get the prompt).
            # This simulation is simplified for demonstration.
            # To use the simulation, you'd need the sample and feature_insights available here.
            # A better simulation would parse the relevant info from the prompt string.
            # For simplicity here, we'll just return a generic simulated response if sample/insights aren't provided to the simulation.
            return simulate_llm_optimization(prompt, None, None)


        # Example function to simulate LLM optimization suggestions
        def simulate_llm_optimization(prompt, sample, feature_insights):
            """
            Simulate an LLM generating optimization suggestions
            In a real implementation, this would call an actual LLM API
            """

            # For demonstration, we'll provide a template response if called without sample/insights
            # In a real LLM, it would rely solely on the prompt.
            if sample is None or feature_insights is None:
                 return """# Optimization Suggestions for Gold Nanoparticle Synthesis

## Current Issues

- Particle_Size_nm is too high (e.g., needs to decrease).
- Zeta_Potential_mV is too low (e.g., needs to decrease).
- Drug_Loading_Efficiency_% is too low (e.g., needs to increase).
- Targeting_Efficiency_% is too low (e.g., needs to increase).
- Cytotoxicity_% is outside the ideal range.

## Recommended Parameter Adjustments (Based on general knowledge and common correlations)

To decrease Particle_Size_nm:
- Decrease reducing agent concentration (e.g., Citrate_Concentration_mM).
- Increase reaction temperature (e.g., Reaction_Temperature_C).

To decrease Zeta_Potential_mV (make it more negative):
- Increase stabilizing agent concentration or modify surface coating (e.g., PEG_Molecular_Weight_Da, pH).

To increase Drug_Loading_Efficiency_%:
- Optimize the ratio of drug to nanoparticle.
- Improve the loading method (e.g., Stirring_Time_min).

To increase Targeting_Efficiency_%:
- Conjugate targeting ligands to the nanoparticle surface.
- Ensure appropriate size and surface charge for passive targeting (EPR effect).

To adjust Cytotoxicity_%:
- Ensure the drug dosage is within therapeutic window.
- Verify that the nanoparticle itself is not overly toxic.

## Revised Synthesis Protocol

Based on the above general suggestions:

1. Review the specific values of the current parameters against the ideal ranges.
2. Based on the feature relationships provided in the background knowledge, identify input parameters strongly correlated with the properties that need improvement.
3. Adjust those input parameters in the direction indicated by the correlation (e.g., if a feature has a positive correlation with a property that needs to decrease, decrease that feature).
4. For quantitative suggestions, consider making incremental changes (e.g., 10-20%) based on the strength of the correlation and the magnitude of the required change.
5. Formulate a detailed step-by-step protocol incorporating these specific adjustments.
6. Synthesize nanoparticles using the revised protocol.
7. Characterize the resulting nanoparticles to measure Particle_Size_nm, Zeta_Potential_mV, Drug_Loading_Efficiency_%, Targeting_Efficiency_%, and Cytotoxicity_%.
8. Evaluate if the properties now meet the suitability criteria.
9. If not, analyze the new results and iterate on the optimization process, making further adjustments as needed.

**Note:** Specific values for adjustments should be derived from the correlations and magnitudes of the required changes, as outlined in step 4. A real LLM would perform this calculation/reasoning based on the input prompt.
"""

            # If sample and insights are provided (for local testing/demonstration)
            # We'll generate a more specific simulation based on the provided data
            # This is a programmatic simulation, not a true LLM response generation.
            correlations = feature_insights.get('target_correlations', {})

            # Identify properties that need improvement
            improvements_needed = {}
            for col, spec in suitability_criteria.items():
                if col in sample.index:
                    low, high = spec['ideal_range']
                    value = sample[col]

                    if value < low:
                        improvements_needed[col] = ('increase', low - value)
                    elif value > high:
                        improvements_needed[col] = ('decrease', value - high)
                # else: property is missing, handled in prompt generation


            # Generate suggestions based on correlations
            suggestions_list = []

            if not improvements_needed:
                 response = "# Optimization Suggestions for Gold Nanoparticle Synthesis\n\n"
                 response += "## All Criteria Met\n\nThe current synthesis parameters appear suitable for cancer treatment.\n"
                 response += "No optimization suggestions are needed based on the defined criteria.\n"
                 return response

            response = "# Optimization Suggestions for Gold Nanoparticle Synthesis\n\n"
            response += "## Current Issues\n\n"

            for property_name, (direction, amount) in improvements_needed.items():
                current_value = sample[property_name]
                low, high = suitability_criteria[property_name]['ideal_range']
                response += f"- {property_name}: Current value ({current_value:.2f}) needs to be {direction}d by {amount:.2f} to reach the ideal range ({low}-{high}).\n"

            response += "\n## Recommended Parameter Adjustments\n\n"

            for property_name, (direction, amount) in improvements_needed.items():
                 suggestion_text = f"To {direction} {property_name} by approximately {amount:.2f}:\n"

                 # Look at relevant correlations
                 relevant_features = {}
                 if property_name in correlations:
                     if direction == 'increase':
                         relevant_features = correlations[property_name]['positive']
                     else:  # decrease
                         relevant_features = correlations[property_name]['negative']

                 # Generate 2-3 suggestions based on top correlations
                 count = 0
                 if relevant_features:
                     for feature, corr_value in relevant_features.items():
                         if feature in sample.index and feature not in suitability_criteria:
                             abs_corr = abs(corr_value)
                             if abs_corr > 0.1:  # Only suggest meaningful correlations
                                 if direction == 'increase':
                                     suggestion_text += f"- Consider increasing {feature} (current value: {sample[feature]:.2f}). "
                                     suggestion_text += f"This has a positive correlation of {corr_value:.3f} with {property_name}.\n"
                                 else:
                                     suggestion_text += f"- Consider decreasing {feature} (current value: {sample[feature]:.2f}). "
                                     suggestion_text += f"This has a negative correlation of {corr_value:.3f} with {property_name}.\n"
                                 count += 1
                                 if count >= 3:
                                     break
                 if count == 0:
                     suggestion_text += "No strongly correlated input features found for specific adjustment.\n"

                 suggestions_list.append(suggestion_text)

            for suggestion in suggestions_list:
                 response += suggestion + "\n"


            response += "\n## Revised Synthesis Protocol\n\n"
            response += "Based on the above recommendations, here is a revised synthesis protocol:\n\n"
            response += "1. Start with the current protocol as a base.\n"

            step_num = 2
            protocol_steps = []
            for property_name, (direction, amount) in improvements_needed.items():
                relevant_features = {}
                if property_name in correlations:
                     if direction == 'increase':
                        relevant_features = correlations[property_name]['positive']
                     else:
                        relevant_features = correlations[property_name]['negative']

                # Take top 2 relevant features for protocol steps
                top_features = list(relevant_features.keys())[:2]

                for feature in top_features:
                    if feature in sample.index and feature not in suitability_criteria:
                        if direction == 'increase':
                            protocol_steps.append(f"{step_num}. Increase {feature} by approximately 10-20% from current value of {sample[feature]:.2f}. This is suggested because {feature} has a positive correlation with {property_name}, which needs to increase.")
                        else:
                            protocol_steps.append(f"{step_num}. Decrease {feature} by approximately 10-20% from current value of {sample[feature]:.2f}. This is suggested because {feature} has a negative correlation with {property_name}, which needs to decrease.")
                        step_num += 1

            # Add unique steps to the response
            for step in protocol_steps:
                 response += step + "\n"


            response += f"{step_num}. Maintain all other parameters constant.\n"
            step_num += 1
            response += f"{step_num}. Synthesize nanoparticles using the revised protocol.\n"
            step_num += 1
            response += f"{step_num}. Characterize the resulting nanoparticles to measure Particle_Size_nm, Zeta_Potential_mV, Drug_Loading_Efficiency_%, Targeting_Efficiency_%, and Cytotoxicity_%.\n"
            step_num += 1
            response += f"{step_num}. Evaluate if the properties now meet the suitability criteria.\n"
            step_num += 1
            response += f"{step_num}. If not, analyze the new results and iterate on the optimization process, making further adjustments as needed.\n"


            return response

        # --- Example of how you might use the optimization prompt/simulation ---
        # Find an unsuitable sample from the test set
        unsuitable_samples = X_test[y_test == 0]

        if not unsuitable_samples.empty:
             sample_to_optimize = unsuitable_samples.iloc[0] # Take the first unsuitable sample

             print("\n--- Generating Optimization Prompt for an Unsuitable Sample ---")
             optimization_prompt = generate_synthesis_optimization_prompt(
                 sample_to_optimize, explanation_text, suitability_criteria
             )
             print(optimization_prompt)

             print("\n--- Simulating LLM Optimization Suggestion ---")
             # In a real scenario, you would call your LLM API here:
             # optimization_suggestion = call_llm_for_optimization(optimization_prompt, api_url="...", api_key="...")
             optimization_suggestion = simulate_llm_optimization(
                 optimization_prompt, sample_to_optimize, feature_insights # Pass sample and insights for specific simulation
             )
             print(optimization_suggestion)
             print("-" * 80)
        else:
             print("\nNo unsuitable samples found in the test set to demonstrate optimization.")


    else:
        print("Skipping data split, LLM classification test, API generation, and optimization example due to empty DataFrame.")

else:
    print("Skipping data analysis and subsequent steps due to empty DataFrame.")