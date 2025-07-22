import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt
import json

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n_samples = 20000

# --- Data extracted and inspired by the provided JSON literature dataset ---

# Based on the JSON, HAuCl4 is the overwhelming choice.
PRECURSORS = ['HAuCl4']

# Common reducers from the literature
REDUCERS = ['NaBH4', 'ascorbic_acid', 'citrate', 'ethylene_glycol']

# Common stabilizers/surfactants from the literature
STABILIZERS = ['PVP', 'citrate', 'PEG', 'CTAB', 'TOAB']

# Common morphologies observed
MORPHOLOGIES = ['nanosphere', 'nanorod', 'nanocube', 'nanoplate', 'nanostar']

# --- Functions to simulate realistic chemical interactions ---

def get_reducer_effect(reducer):
    """
    Simulates the effect of a reducing agent on reaction kinetics and particle size.
    Stronger reducers lead to faster reactions and smaller initial particles.
    Values are based on general chemical knowledge reflected in the literature data.
    """
    if reducer == "NaBH4":  # Strong reducer
        return {'time_factor': np.random.uniform(0.3, 0.6), 'size_base_mod': np.random.uniform(-10, -5), 'pdi_mod': 0.08}
    elif reducer == "ascorbic_acid":  # Moderate reducer
        return {'time_factor': np.random.uniform(0.6, 0.9), 'size_base_mod': np.random.uniform(-5, 0), 'pdi_mod': 0.03}
    elif reducer == "ethylene_glycol": # Moderate reducer, high temp
        return {'time_factor': np.random.uniform(0.5, 0.8), 'size_base_mod': np.random.uniform(5, 15), 'pdi_mod': 0.05}
    else:  # 'citrate', a weak reducer
        return {'time_factor': 1.0, 'size_base_mod': 0, 'pdi_mod': 0.0}

def get_stabilizer_effect(stabilizer, pH):
    """
    Simulates the effect of a stabilizer on size, zeta potential, and morphology.
    These relationships are derived from patterns in the provided JSON data.
    """
    effects = {
        'size_add': 0,
        'zeta_base': 0,
        'morphology_boost': {} # Changed from priority to a boost system
    }
    if stabilizer == "PVP":
        effects['size_add'] = np.random.uniform(3, 8)
        effects['zeta_base'] = np.random.uniform(-15, -5)
        effects['morphology_boost'] = {'nanocube': 15, 'nanosphere': 5, 'nanoplate': 10}
    elif stabilizer == "citrate":
        effects['size_add'] = np.random.uniform(1, 4)
        effects['zeta_base'] = np.random.uniform(-50, -30)
        effects['morphology_boost'] = {'nanosphere': 20}
    elif stabilizer == "PEG":
        effects['size_add'] = np.random.uniform(5, 12)
        effects['zeta_base'] = np.random.uniform(-10, 0) # Near-neutral charge
        effects['morphology_boost'] = {'nanosphere': 10, 'nanorod': 5}
    elif stabilizer == "CTAB":
        effects['size_add'] = np.random.uniform(2, 6)
        effects['zeta_base'] = np.random.uniform(30, 60) # Cationic surfactant
        effects['morphology_boost'] = {'nanorod': 25}
    elif stabilizer == "TOAB":
        effects['size_add'] = np.random.uniform(1, 3)
        effects['zeta_base'] = np.random.uniform(20, 50)
        effects['morphology_boost'] = {'nanosphere': 5, 'nanorod': 10}
    
    # pH effect on zeta potential
    if pH < 6:
        effects['zeta_base'] += np.random.uniform(5, 15)
    elif pH > 8:
        effects['zeta_base'] -= np.random.uniform(5, 15)
        
    return effects

def get_synthesis_method_params(method):
    """
    Provides typical parameters for well-known synthesis methods found in the literature.
    This function introduces strong correlations between method, chemicals, and conditions.
    """
    if method == "Turkevich":
        return {
            'reducer': 'citrate',
            'stabilizer': 'citrate',
            'temp_range': (90, 100),
            'time_base_min': 15,
            'size_base': 18,
            'pdi_base': 0.15,
            'pH_range': (5, 7)
        }
    elif method == "Seed-Mediated":
        return {
            'reducer': 'ascorbic_acid',
            'stabilizer': 'CTAB',
            'temp_range': (25, 40),
            'time_base_min': 120,
            'size_base': 40, # This will be length for nanorods
            'pdi_base': 0.20,
            'pH_range': (2, 4)
        }
    elif method == "Brust-Schiffrin":
        return {
            'reducer': 'NaBH4',
            'stabilizer': 'TOAB',
            'temp_range': (0, 25),
            'time_base_min': 60,
            'size_base': 5,
            'pdi_base': 0.25,
            'pH_range': (8, 11)
        }
    elif method == "Polyol":
        return {
            'reducer': 'ethylene_glycol', # The solvent acts as the reducer
            'stabilizer': 'PVP',
            'temp_range': (140, 200),
            'time_base_min': 45,
            'size_base': 80,
            'pdi_base': 0.18,
            'pH_range': (7, 9)
        }

def generate_biomedical_properties(size, zeta, stabilizer, morphology, pH, pdi):
    """
    Generates plausible biomedical outcomes based on nanoparticle properties.
    This version has more complex relationships to create a stronger signal for the model.
    """
    # Drug Loading Efficiency (%) - now depends on more factors
    loading = 45 - (size / 4) + (10 * (1-pdi))
    if stabilizer in ['PEG', 'PVP']:
        loading += 10
    if morphology == 'nanocube': # Cubes have high surface area
        loading += 8
    loading -= abs(pH - 7) * 2 # Penalize non-neutral pH
    loading += np.random.normal(0, 2) # Reduced noise
    loading = np.clip(loading, 5, 98)

    # Targeting Efficiency (%) - now depends on more factors
    targeting = 55 - abs(zeta / 4) - (size / 6)
    if stabilizer == 'PEG':
        targeting += 25 # PEGylation helps avoid RES
    if morphology == 'nanostar': # Spiky shape can enhance targeting
        targeting += 15
    targeting += np.random.normal(0, 2) # Reduced noise
    targeting = np.clip(targeting, 5, 98)

    # Cytotoxicity (% cell death)
    cytotoxicity = 15 + abs(zeta / 5) + (size / 12)
    if stabilizer == 'CTAB':
        cytotoxicity += 35 # CTAB is known to be cytotoxic
    elif stabilizer == 'PEG' or stabilizer == 'citrate':
        cytotoxicity -= 12
    cytotoxicity += np.random.normal(0, 2) # Reduced noise
    cytotoxicity = np.clip(cytotoxicity, 2, 95)
    
    return loading, targeting, cytotoxicity

# --- Main Data Generation Loop ---

data = []
synthesis_methods = ["Turkevich", "Seed-Mediated", "Brust-Schiffrin", "Polyol"]

for _ in range(n_samples):
    # 1. Select a synthesis method
    method = np.random.choice(synthesis_methods)
    method_params = get_synthesis_method_params(method)

    # 2. Assign parameters based on the method
    precursor = 'HAuCl4'
    reducer = method_params['reducer']
    stabilizer = method_params['stabilizer']
    
    pH = np.random.uniform(*method_params['pH_range'])
    temperature = np.random.uniform(*method_params['temp_range'])
    
    reducer_effects = get_reducer_effect(reducer)
    stabilizer_effects = get_stabilizer_effect(stabilizer, pH)

    # 3. Calculate synthesis time
    time = method_params['time_base_min'] * reducer_effects['time_factor'] * np.random.uniform(0.8, 1.2)
    
    # 4. Determine particle morphology and size
    # --- New morphology generation logic for better balance ---
    morph_weights = np.ones(len(MORPHOLOGIES)) # Start with equal weights
    for morph, boost in stabilizer_effects['morphology_boost'].items():
        if morph in MORPHOLOGIES:
            idx = MORPHOLOGIES.index(morph)
            morph_weights[idx] += boost
    
    # Add a small chance for nanoplates in Polyol synthesis
    if method == 'Polyol' and 'nanoplate' in MORPHOLOGIES:
        idx = MORPHOLOGIES.index('nanoplate')
        morph_weights[idx] += 5

    morph_probs = morph_weights / np.sum(morph_weights)
    morphology = np.random.choice(MORPHOLOGIES, p=morph_probs)
    
    size_base = method_params['size_base']
    size = size_base + reducer_effects['size_base_mod'] + stabilizer_effects['size_add'] + np.random.normal(0, 5)
    
    size = max(2, size)

    pdi = method_params['pdi_base'] + reducer_effects['pdi_mod'] + np.random.normal(0, 0.05)
    pdi = np.clip(pdi, 0.05, 0.5)

    if morphology == 'nanorod':
        aspect_ratio = np.random.uniform(2.5, 5.0)
        width = size / aspect_ratio
    else:
        width = size
        aspect_ratio = 1.0

    # 5. Determine Zeta Potential
    zeta_potential = stabilizer_effects['zeta_base'] + np.random.normal(0, 5)

    # 6. Generate biomedical properties
    drug_loading, targeting_eff, cytotoxicity = generate_biomedical_properties(size, zeta_potential, stabilizer, morphology, pH, pdi)
    
    # 7. Simulate treatment outcome
    treatment_success_metric = (targeting_eff / 10) + (drug_loading / 10) - (cytotoxicity / 15) - (pdi * 10)
    if size > 100:
        treatment_success_metric -= 5
    
    # Lowered threshold for more balanced outcomes
    successful_treatment = 1 if treatment_success_metric > 6.0 else 0

    data.append([
        precursor, reducer, stabilizer, morphology,
        temperature, time, pH,
        size, width, aspect_ratio, pdi, zeta_potential,
        drug_loading, targeting_eff, cytotoxicity,
        successful_treatment
    ])

# --- Create and Save DataFrame ---

columns = [
    'Precursor', 'Reducer', 'Stabilizer', 'Morphology',
    'Temperature_C', 'Time_min', 'pH',
    'Particle_Size_nm', 'Particle_Width_nm', 'Aspect_Ratio', 'PDI', 'Zeta_Potential_mV',
    'Drug_Loading_Efficiency', 'Targeting_Efficiency', 'Cytotoxicity',
    'Successful_Treatment'
]
df = pd.DataFrame(data, columns=columns)

# Define the file path for the raw generated data
output_csv_path = "aunp_synthesis_realistic_v1.csv"
output_csv_path_full = os.path.join(os.getcwd(), output_csv_path)
df.to_csv(output_csv_path_full, index=False)

print(f"Realistic dataset generated and saved to {output_csv_path_full}")
print("\n--- Dataset Head ---")
print(df.head())
print("\n--- Data Description ---")
print(df.describe())
print("\n--- NEW DATASET BALANCE ---")
print("\nSuccessful Treatment Distribution:")
print(df['Successful_Treatment'].value_counts(normalize=True))
print("\nMorphology Distribution:")
print(df['Morphology'].value_counts(normalize=True))


# --- Data Transformation (as in original script) ---
# Separating features and target
X_to_transform = df.drop('Successful_Treatment', axis=1)
y_to_transform = df[['Successful_Treatment']]

# Identifying categorical and numerical columns
categorical_features = X_to_transform.select_dtypes(include=['object']).columns
numerical_features = X_to_transform.select_dtypes(include=np.number).columns

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Apply the transformations
X_transformed = preprocessor.fit_transform(X_to_transform)

# Get the new column names
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
transformed_columns = numerical_features.tolist() + ohe_feature_names.tolist()

# Create a new DataFrame with the transformed data
X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)

# Combine with target variable
transformed_df = pd.concat([X_transformed_df, y_to_transform.reset_index(drop=True)], axis=1)

# Define the file path for the final transformed data
output_csv_path_transformed = "aunp_synthesis_realistic_v1_transformed.csv"
output_csv_path_transformed_full = os.path.join(os.getcwd(), output_csv_path_transformed)
transformed_df.to_csv(output_csv_path_transformed_full, index=False)

print(f"\nTransformed dataset saved to {output_csv_path_transformed_full}")

# --- Basic Visualization of the New Data ---
plt.figure(figsize=(15, 10))
plt.suptitle('Distributions in the Realistically Generated Dataset', fontsize=16)

plt.subplot(2, 3, 1)
df['Particle_Size_nm'].hist(bins=50, color='skyblue')
plt.title('Particle Size Distribution')
plt.xlabel('Size (nm)')
plt.ylabel('Frequency')

plt.subplot(2, 3, 2)
df['Zeta_Potential_mV'].hist(bins=50, color='salmon')
plt.title('Zeta Potential Distribution')
plt.xlabel('Zeta Potential (mV)')

plt.subplot(2, 3, 3)
df['Morphology'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Morphology Distribution')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 3, 4)
df['Reducer'].value_counts().plot(kind='bar', color='gold')
plt.title('Reducer Distribution')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 3, 5)
df['Stabilizer'].value_counts().plot(kind='bar', color='cornflowerblue')
plt.title('Stabilizer Distribution')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 3, 6)
df['Successful_Treatment'].value_counts().plot(kind='bar', color='orchid')
plt.title('Treatment Outcome')
plt.xticks(rotation=0)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()