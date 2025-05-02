
from flask import Flask, request, jsonify
import pandas as pd
import json
import os # Import os to handle file paths

app = Flask(__name__)

# Load the explanation text and criteria
# Assuming the script generating this API saves these files in the same directory
MODEL_DIR = "saved_models" # Use the defined model directory
explanation_path = os.path.join(MODEL_DIR, "aunp_synthesis_explanation_20250502_210724.md")

try:
    with open(explanation_path, 'r') as f:
        explanation_text = f.read()
    print(f"Loaded explanation text from {explanation_path}")
except FileNotFoundError:
    explanation_text = "Explanation text not found."
    print(f"Error: Explanation text file not found at {explanation_path}")


# Define suitability criteria (should ideally be loaded from a config file in production)
# Embedding the criteria directly as a string representation of the dictionary
suitability_criteria_str = '''{"Particle_Size_nm": {"ideal_range": [40, 100], "description": "Optimal particle size range for gold nanoparticles in cancer treatment. Particles should be large enough to carry drug payload but small enough to penetrate tumor tissue via the EPR effect."}, "Zeta_Potential_mV": {"ideal_range": [-30, -5], "description": "Zeta potential indicates surface charge and stability. Slightly negative values promote stability while facilitating cellular uptake."}, "Drug_Loading_Efficiency_%": {"ideal_range": [70, 100], "description": "Indicates how efficiently the drug is loaded onto nanoparticles. Higher values mean more effective drug delivery."}, "Targeting_Efficiency_%": {"ideal_range": [75, 100], "description": "Measures how well nanoparticles target cancer cells. Higher values indicate better specificity for cancer cells."}, "Cytotoxicity_%": {"ideal_range": [70, 90], "description": "Indicates toxicity to cancer cells. Should be high enough to effectively kill cancer cells but not excessively toxic."}}''' # Use json.dumps for robust string representation
suitability_criteria = json.loads(suitability_criteria_str)
print("Loaded suitability criteria.")

# Function to create a prompt for an LLM to classify if a sample is suitable for cancer treatment
def create_llm_classification_prompt(sample, feature_explanation, criteria):
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

    predicted_label = 1 if meets_criteria else 0

    return response, predicted_label

@app.route('/classify', methods=['POST'])
def classify():
    """Endpoint to classify if AuNP synthesis parameters are suitable for cancer treatment"""
    try:
        data = request.json
        # Ensure data contains keys from TARGET_COLS + relevant input features
        required_keys = list(suitability_criteria.keys()) + [col for col in data.keys() if col not in suitability_criteria]
        sample_data = {k: data.get(k, None) for k in required_keys}

        # Convert dict to pandas Series
        sample = pd.Series(sample_data).dropna() # Drop missing values

        if not all(col in sample.index for col in suitability_criteria.keys()):
             missing = [col for col in suitability_criteria.keys() if col not in sample.index]
             return jsonify({'error': f"Missing required properties for classification: {', '.join(missing)}"}), 400

        # Create prompt
        prompt = create_llm_classification_prompt(sample, explanation_text, suitability_criteria)

        # Call LLM service (simulated)
        llm_response, predicted_label = call_llm_service(prompt, sample, suitability_criteria)

        return jsonify({
            'suitable': bool(predicted_label),
            'explanation': llm_response,
            'prompt': prompt # Optional: include prompt for debugging
        })

    except Exception as e:
        # Log the error in a real application
        print(f"Error during classification: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/explain_features', methods=['GET'])
def explain_features():
    """Endpoint to get feature relationship explanations"""
    return jsonify({
        'explanation': explanation_text,
        'criteria': suitability_criteria
    })

if __name__ == '__main__':
    # To run this API, save the code as a Python file (e.g., api.py)
    # and run 'python api.py' in your terminal within the correct directory.
    # Make sure you have Flask installed (`pip install Flask pandas`)
    # and the required data/explanation files in the 'saved_models' directory.
    # debug=True is useful for development but should be False in production.
    app.run(debug=True)
