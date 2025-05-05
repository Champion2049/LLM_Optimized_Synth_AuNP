import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 20000


def get_reducer_effect(reducer):
    """Simulates effect of reducing agent on kinetics and size."""
    if reducer == "NaBH4":
        # Strong reducer
        return {'time_factor': 0.4, 'size_base_mod': -8, 'pdi_mod': 0.08}
    elif reducer == "ascorbic_acid":
        # Moderate reducer
        return {'time_factor': 0.7, 'size_base_mod': -3, 'pdi_mod': 0.03}
    else: # citrate
        # Weak reducer
        return {'time_factor': 1.0, 'size_base_mod': 0, 'pdi_mod': 0.0}

def get_stabilizer_effect(stabilizer, pH):
    """Simulates effect of stabilizer on size, zeta potential, loading, and biomedical properties."""
    size_mod = 0
    zeta_mod = 0
    loading_mod = 0
    targeting_mod = 0
    cytotoxicity_mod = 0

    if stabilizer == "PVP":
        size_mod = np.random.uniform(3, 7)
        zeta_mod = np.random.normal(-40, 8)
        loading_mod = 10
        targeting_mod = 5
        cytotoxicity_mod = -5
    elif stabilizer == "CTAB":
        size_mod = np.random.uniform(4, 10)
        zeta_mod = np.random.normal(55, 10)
        loading_mod = -15
        targeting_mod = -10
        cytotoxicity_mod = 30 # Significant toxicity contribution
    elif stabilizer == "PEG":
        size_mod = np.random.uniform(2, 5)
        zeta_mod = np.random.normal(-35, 7)
        loading_mod = 30
        targeting_mod = 20 # Significant targeting benefit
        cytotoxicity_mod = -10
    else: # "citrate" or other weak stabilizers
        zeta_mod = np.random.normal(-15, 12) + (pH - 7) * np.random.uniform(2.0, 4.0) # Stronger pH effect for citrate

    return {'size_mod': size_mod, 'zeta_mod': zeta_mod, 'loading_mod': loading_mod,
            'targeting_mod': targeting_mod, 'cytotoxicity_mod': cytotoxicity_mod}

# --- Function to Generate Synthesis Data ---
def generate_synthesis_data_v3(n_samples):
    """Generates a dataset simulating nanoparticle synthesis and properties."""
    data = []

    for i in range(n_samples):
        # Core synthesis parameters
        precursor_conc = np.random.uniform(0.1, 5.0)  # mM
        reducing_agent = np.random.choice(["citrate", "ascorbic_acid", "NaBH4"], p=[0.4, 0.35, 0.25])
        stabilizer = np.random.choice(["PVP", "CTAB", "PEG", "citrate"], p=[0.3, 0.15, 0.35, 0.2])
        pH = np.random.choice(np.arange(6, 23) * 0.5) # pH 3.0 to 11.0
        temperature = np.random.randint(15, 101) # °C
        mixing_speed = np.random.randint(100, 1201) # RPM
        background_ionic_strength_mM = np.random.uniform(1, 15)

        # Get effects based on categorical choices
        reducer_effects = get_reducer_effect(reducing_agent)
        stabilizer_effects = get_stabilizer_effect(stabilizer, pH)

        # --- Calculate Reaction Time ---
        base_time = 60 * reducer_effects['time_factor']
        temp_factor = np.exp(-0.06 * (temperature - 25)) # Arrhenius-like temp effect
        ph_factor = 1 + 0.07 * abs(pH - 7) # Deviation from neutral pH slows/speeds reaction
        conc_factor = 1 / (1 + 0.15 * precursor_conc) # Higher concentration consumes reactants faster
        reaction_time_float = base_time * temp_factor * ph_factor * conc_factor + np.random.normal(0, 4)
        reaction_time = int(np.clip(reaction_time_float, 5, 240)) # Clamp within realistic bounds (min)

        # --- Calculate Particle Size ---
        size_deterministic = (
            18 + reducer_effects['size_base_mod'] # Base size modified by reducer strength
            + stabilizer_effects['size_mod'] # Stabilizer steric/charge effects
            + 3 * np.log1p(precursor_conc) # Higher conc leads to more nuclei/larger particles
            + 0.2 * (temperature - 25) # Higher temp generally leads to larger particles (faster growth)
            - 0.7 * abs(pH - 8) # pH optimum effect (deviation increases/decreases size)
            - 0.008 * (mixing_speed - 500) # Faster mixing can lead to smaller particles (better reactant distribution)
            + 0.08 * (temperature - 60) * (precursor_conc - 1) # Interaction term
        )
        size_deterministic = max(5.0, size_deterministic) # Ensure minimum physical size
        particle_size_float = np.random.lognormal(mean=np.log(size_deterministic), sigma=0.08) # Log-normal distribution common for size
        particle_size = int(np.clip(particle_size_float, 15, 150)) # Clamp final size (nm)

        # --- Calculate Zeta Potential ---
        initial_zeta = stabilizer_effects['zeta_mod'] # Base zeta from stabilizer charge/coating
        total_ionic_strength_mM = background_ionic_strength_mM + precursor_conc * 0.5 # Estimate total ionic strength
        screening_factor = 1 / (1 + 0.2 * np.log1p(total_ionic_strength_mM)) # Ionic strength screens surface charge
        initial_zeta = np.sign(initial_zeta) * abs(initial_zeta) * screening_factor
        initial_zeta += np.random.normal(0, 1)
        # Push zeta towards more stable regions away from +/- 25mV
        if -25 < initial_zeta < 25:
             if initial_zeta >= 0:
                 zeta_potential = 25 + (initial_zeta / 25.0) * 35.0
             else:
                 zeta_potential = -25 + (initial_zeta / 25.0) * 35.0
        else:
             zeta_potential = initial_zeta
        zeta_potential = np.clip(zeta_potential, -60, 60) # Clamp final zeta (mV)

        # --- Calculate Polydispersity Index ---
        pdi = (
            0.15 # Base PDI
            + reducer_effects['pdi_mod'] # Stronger reducers can increase PDI if uncontrolled
            + 0.02 * precursor_conc # Higher conc can increase PDI
            + 0.001 * ((temperature - 60)/10)**2 # Deviation from optimal temp increases PDI
            + 0.0005 * ((reaction_time - 45)/10)**2 # Very short/long times can increase PDI
            + 0.0002 * (1200 - mixing_speed) # Slower mixing increases PDI
            + np.random.uniform(-0.01, 0.02)
        )
        polydispersity = np.clip(pdi, 0.08, 0.7) # Clamp PDI (dimensionless)

        # --- Calculate Biomedical Metrics ---
        # Drug Loading Efficiency (%)
        loading = (
            70 # Base loading
            + 500 / particle_size # Higher surface area (smaller size) generally increases loading
            + stabilizer_effects['loading_mod'] # Stabilizer interaction with drug affects loading
            - 20 * polydispersity # Higher PDI reduces effective loading
            + np.random.normal(0, 4)
        )
        drug_loading_efficiency = np.clip(loading, 10, 98)

        # Targeting Efficiency (%)
        # Simulate EPR effect peak size and influence of surface properties
        size_targeting_factor = 50 * np.exp(-((particle_size - 60)**2) / (2 * 30**2)) # Gaussian peak around 60nm
        zeta_targeting_factor = -0.8 * (zeta_potential)**2 / (2 * 25**2) + 15 # Moderate zeta can be beneficial
        zeta_targeting_factor = max(0, zeta_targeting_factor) # Ensure non-negative contribution
        targeting = (
            35 # Base targeting
            + size_targeting_factor
            + zeta_targeting_factor
            + stabilizer_effects['targeting_mod'] * 1.5 # Stabilizer (e.g., PEG) enhances targeting
            - 30 * polydispersity # Poorly defined particles target less effectively
            + np.random.normal(0, 5)
        )
        targeting_efficiency = np.clip(targeting, 15, 99)

        # Cytotoxicity (%)
        toxicity = (
            20 # Base toxicity
            + 100 * np.exp(-(particle_size / 15)**2) # Smaller particles often show higher toxicity
            + stabilizer_effects['cytotoxicity_mod'] # Some stabilizers (e.g., CTAB) are inherently toxic
            + 0.8 * precursor_conc # Residual precursors might contribute
            + 0.2 * (temperature - 20) # Synthesis temp might affect surface defects/toxicity
            # Interaction effects
            + (15 if stabilizer == "CTAB" and particle_size < 50 else 0)
            + (10 if stabilizer == "CTAB" and precursor_conc > 3 else 0)
            + np.random.normal(0, 4)
        )
        cytotoxicity = np.clip(toxicity, 5, 99)

        # Append the generated data point
        data.append([
            precursor_conc, reducing_agent, stabilizer, pH, temperature,
            reaction_time, mixing_speed, particle_size, zeta_potential,
            polydispersity, drug_loading_efficiency, targeting_efficiency,
            cytotoxicity
        ])

    # Create DataFrame
    columns = [
        "Precursor_Conc_mM", "Reducing_Agent", "Stabilizer", "pH",
        "Temperature_C", "Reaction_Time_min", "Mixing_Speed_RPM",
        "Particle_Size_nm", "Zeta_Potential_mV", "Polydispersity",
        "Drug_Loading_Efficiency_%", "Targeting_Efficiency_%", "Cytotoxicity_%"
    ]
    return pd.DataFrame(data, columns=columns)

# --- Main Execution ---
# Generate dataset
df_v3 = generate_synthesis_data_v3(n_samples)

# Define file path for the original (untransformed) data
output_csv_path_original = "aunp_synthesis_cancer_treatment_v3_original.csv"
output_csv_path_original_full = os.path.join(os.getcwd(), output_csv_path_original)
df_v3.to_csv(output_csv_path_original_full, index=False)

print(f"Refined dataset (v3) created with {n_samples} samples.")
print(f"Original dataset saved to {output_csv_path_original_full}")
print(df_v3.head())
print("\nSummary Statistics (Original Data):")
print(df_v3.describe())

# --- Apply Transformers ---

# Define input features used for prediction
input_features = ["Precursor_Conc_mM", "Reducing_Agent", "Stabilizer", "pH",
                  "Temperature_C", "Reaction_Time_min", "Mixing_Speed_RPM"]

# Define all output targets the model will predict
output_targets = ["Particle_Size_nm", "Zeta_Potential_mV", "Polydispersity",
                  "Drug_Loading_Efficiency_%", "Targeting_Efficiency_%", "Cytotoxicity_%"]

# Identify which output targets need scaling (most of them)
output_targets_to_scale = ["Zeta_Potential_mV", "Polydispersity",
                           "Drug_Loading_Efficiency_%", "Targeting_Efficiency_%", "Cytotoxicity_%"]

# Identify original features (including Particle_Size_nm target) that should pass through the input transformer without scaling/encoding
# Particle_Size_nm is an output, but also used as input for biomedical targets, so it needs to be passed through here.
passthrough_features = ["Precursor_Conc_mM", "pH", "Temperature_C", "Reaction_Time_min", "Particle_Size_nm"]

# Identify categorical input features needing OneHotEncoding
categorical_input_features = ['Reducing_Agent', 'Stabilizer']
# Identify numerical input features needing scaling (MinMaxScaler in this case)
numerical_input_to_scale = ["Mixing_Speed_RPM"]

# Create the preprocessor for input features and the Particle_Size_nm passthrough feature
# This prepares the 'X' data for the model
preprocessor_X = ColumnTransformer(
    transformers=[
        ('num_minmax', MinMaxScaler(feature_range=(0, 1)), numerical_input_to_scale), # Scale Mixing Speed [0, 1]
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_input_features) # One-hot encode categoricals
    ],
    remainder='passthrough', # Keep columns listed in passthrough_features and any others not specified
    verbose_feature_names_out=False # Keep original names where possible
)

# Select the columns from the original dataframe that preprocessor_X will handle
columns_for_preprocessor_X = numerical_input_to_scale + categorical_input_features + passthrough_features

# Fit the preprocessor and transform the selected columns
X_transformed_array = preprocessor_X.fit_transform(df_v3[columns_for_preprocessor_X])

# Get feature names after transformation (includes one-hot encoded names)
X_transformed_columns = preprocessor_X.get_feature_names_out()
X_transformed_df = pd.DataFrame(X_transformed_array, columns=X_transformed_columns)


# Create a scaler specifically for the output targets that need scaling
# This prepares the 'y' data for the model
target_scaler = MinMaxScaler(feature_range=(0, 1)) # Scales outputs to [0, 1]

# Select the output target columns from the original dataframe
y_to_scale_df = df_v3[output_targets_to_scale]
# Fit the scaler and transform these target columns
y_scaled_transformed_array = target_scaler.fit_transform(y_to_scale_df)

# Convert scaled targets back to a DataFrame
y_scaled_transformed_df = pd.DataFrame(y_scaled_transformed_array, columns=output_targets_to_scale)


print("\n--- Data After Transformation ---")
print("Shape of transformed inputs and passthrough features (X_transformed_df):", X_transformed_df.shape)

# Rounding the scaled 'Mixing_Speed_RPM' column for cleaner display
if 'Mixing_Speed_RPM' in X_transformed_df.columns:
    X_transformed_df['Mixing_Speed_RPM'] = X_transformed_df['Mixing_Speed_RPM'].round(4)

print("\nTransformed Input Features and Passthrough (first 5 rows):")
print(X_transformed_df.head())

print("\nShape of original output targets to scale (y_to_scale_df):", y_to_scale_df.shape)
print("Shape of transformed scaled output targets (y_scaled_transformed_df):", y_scaled_transformed_df.shape)
print("\nTransformed Scaled Output Targets (0-1 range) (first 5 rows):")
print(y_scaled_transformed_df.head())


# --- Combine and Save Transformed Data ---

# Combine the processed input features (X_transformed_df) and scaled output targets (y_scaled_transformed_df)
# This creates a single file useful for loading data, but remember to split X and y for model training.
# X = transformed_df[X_transformed_columns]
# y = transformed_df[output_targets_to_scale]
transformed_df = pd.concat([X_transformed_df, y_scaled_transformed_df], axis=1)

# Define the file path for the final transformed data
output_csv_path_transformed = "aunp_synthesis_cancer_treatment_v3_transformed.csv"
output_csv_path_transformed_full = os.path.join(os.getcwd(), output_csv_path_transformed)
transformed_df.to_csv(output_csv_path_transformed_full, index=False)

print(f"\nCombined transformed dataset saved to {output_csv_path_transformed_full}")
print("\nColumns in the final transformed dataset:")
print(transformed_df.columns.tolist())

# --- Basic Visualization of the Original Data ---
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(df_v3['Particle_Size_nm'], bins=50)
plt.title('Original Particle Size Distribution')
plt.xlabel('Size (nm)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(df_v3['Zeta_Potential_mV'], bins=50)
plt.title('Original Zeta Potential Distribution')
plt.xlabel('Zeta Potential (mV)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(df_v3['Polydispersity'], bins=50)
plt.title('Original Polydispersity Distribution')
plt.xlabel('PDI')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.scatter(df_v3['Particle_Size_nm'], df_v3['Targeting_Efficiency_%'], alpha=0.1)
plt.title('Original Targeting Efficiency vs Size')
plt.xlabel('Size (nm)')
plt.ylabel('Targeting Efficiency (%)')

plt.tight_layout()
plt.show()