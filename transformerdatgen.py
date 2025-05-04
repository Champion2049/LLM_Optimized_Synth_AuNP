import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 20000 # Keep your 20k samples

# --- More Realistic Helper Functions ---

def get_reducer_effect(reducer):
    """Simulates effect of reducing agent on kinetics and size."""
    if reducer == "NaBH4":
        # Strong reducer: faster reaction, smaller initial particles, potentially higher PDI if uncontrolled
        return {'time_factor': 0.4, 'size_base_mod': -8, 'pdi_mod': 0.08} # Adjusted values
    elif reducer == "ascorbic_acid":
        # Moderate reducer
        return {'time_factor': 0.7, 'size_base_mod': -3, 'pdi_mod': 0.03} # Adjusted values
    else: # citrate
        # Weak reducer: slower reaction, baseline size
        return {'time_factor': 1.0, 'size_base_mod': 0, 'pdi_mod': 0.0}

def get_stabilizer_effect(stabilizer, pH):
    """Simulates effect of stabilizer on size, zeta potential, and loading."""
    size_mod = 0
    zeta_mod = 0
    loading_mod = 0
    # Added factors for biomedical effects
    targeting_mod = 0
    cytotoxicity_mod = 0

    if stabilizer == "PVP":
        size_mod = np.random.uniform(3, 7) # Adjusted range
        zeta_mod = np.random.normal(-40, 8) # Adjusted mean/std
        loading_mod = 10 # Increased loading effect
        targeting_mod = 5 # Slight targeting benefit (reducing non-specific)
        cytotoxicity_mod = -5 # Slight reduction in toxicity
    elif stabilizer == "CTAB":
        size_mod = np.random.uniform(4, 10) # Adjusted range
        zeta_mod = np.random.normal(55, 10) # More positive
        loading_mod = -15 # Reduced loading
        targeting_mod = -10 # Increased non-specific binding/clearance
        cytotoxicity_mod = 30 # Significant toxicity contribution
    elif stabilizer == "PEG":
        size_mod = np.random.uniform(2, 5) # Adjusted range
        zeta_mod = np.random.normal(-35, 7) # Adjusted mean/std
        loading_mod = 30 # Higher loading
        targeting_mod = 20 # Significant targeting benefit (stealth effect/functionalization)
        cytotoxicity_mod = -10 # Reduced toxicity (stealth effect)
    else: # "citrate" or other weak stabilizers
        zeta_mod = np.random.normal(-15, 12) + (pH - 7) * np.random.uniform(2.0, 4.0) # Stronger pH effect
        # Default mods are 0

    return {'size_mod': size_mod, 'zeta_mod': zeta_mod, 'loading_mod': loading_mod,
            'targeting_mod': targeting_mod, 'cytotoxicity_mod': cytotoxicity_mod}

# --- Generate Interdependent Features (More Realistic Biomedical) ---

def generate_synthesis_data_v3(n_samples):
    data = []

    for i in range(n_samples):
        # Core synthesis parameters
        precursor_conc = np.random.uniform(0.1, 5.0)  # mM
        reducing_agent = np.random.choice(["citrate", "ascorbic_acid", "NaBH4"], p=[0.4, 0.35, 0.25]) # Adjusted probabilities
        stabilizer = np.random.choice(["PVP", "CTAB", "PEG", "citrate"], p=[0.3, 0.15, 0.35, 0.2]) # Adjusted probabilities
        pH = np.random.choice(np.arange(6, 23) * 0.5) # pH 3.0 to 11.0

        temperature = np.random.randint(15, 101) # °C
        mixing_speed = np.random.randint(100, 1201) # RPM

        background_ionic_strength_mM = np.random.uniform(1, 15) # Increased range

        # Get effects based on categorical choices
        reducer_effects = get_reducer_effect(reducing_agent)
        stabilizer_effects = get_stabilizer_effect(stabilizer, pH)

        # --- Calculate Reaction Time ---
        base_time = 60 * reducer_effects['time_factor']
        temp_factor = np.exp(-0.06 * (temperature - 25)) # Stronger temp effect
        ph_factor = 1 + 0.07 * abs(pH - 7) # Stronger pH effect
        conc_factor = 1 / (1 + 0.15 * precursor_conc) # Stronger conc effect
        # REDUCED NOISE FOR REACTION TIME
        reaction_time_float = base_time * temp_factor * ph_factor * conc_factor + np.random.normal(0, 4) # Noise reduced from 8 to 4
        reaction_time = int(np.clip(reaction_time_float, 5, 240)) # Adjusted range

        # --- Calculate Particle Size ---
        size_deterministic = (
            18 + reducer_effects['size_base_mod'] # Adjusted base size
            + stabilizer_effects['size_mod']
            + 3 * np.log1p(precursor_conc) # Stronger conc effect
            + 0.2 * (temperature - 25) # Stronger temp effect
            - 0.7 * abs(pH - 8) # Stronger pH optimum effect
            - 0.008 * (mixing_speed - 500) # Stronger mixing effect
            + 0.08 * (temperature - 60) * (precursor_conc - 1) # Stronger interaction term
        )
        size_deterministic = max(5.0, size_deterministic) # Ensure minimum size
        # REDUCED NOISE FOR PARTICLE SIZE
        particle_size_float = np.random.lognormal(mean=np.log(size_deterministic), sigma=0.08) # Sigma reduced from 0.15 to 0.08
        particle_size = int(np.clip(particle_size_float, 15, 150)) # Adjusted realistic range


        # --- Calculate Zeta Potential ---
        initial_zeta = stabilizer_effects['zeta_mod']

        total_ionic_strength_mM = background_ionic_strength_mM + precursor_conc * 0.5 # Precursor contributes to IS
        screening_factor = 1 / (1 + 0.2 * np.log1p(total_ionic_strength_mM)) # Stronger screening effect
        initial_zeta = np.sign(initial_zeta) * abs(initial_zeta) * screening_factor
        # REDUCED NOISE FOR ZETA POTENTIAL
        initial_zeta += np.random.normal(0, 1) # Noise reduced from 6 to 3

        # Apply transformation to push towards stable regions, but less aggressively
        if -25 < initial_zeta < 25: # Adjusted range for transformation
             if initial_zeta >= 0:
                 zeta_potential = 25 + (initial_zeta / 25.0) * 35.0 # Map to [25, 60)
             else:
                 zeta_potential = -25 + (initial_zeta / 25.0) * 35.0 # Map to (-60, -25]
        else:
             zeta_potential = initial_zeta

        zeta_potential = np.clip(zeta_potential, -60, 60) # Adjusted overall range


        # --- Calculate Polydispersity Index ---
        pdi = (
            0.15 # Base PDI
            + reducer_effects['pdi_mod']
            + 0.02 * precursor_conc # Stronger conc effect
            + 0.001 * ((temperature - 60)/10)**2 # Stronger temp effect
            + 0.0005 * ((reaction_time - 45)/10)**2 # Adjusted time optimum
            + 0.0002 * (1200 - mixing_speed) # Stronger mixing effect
            # REDUCED NOISE FOR PDI
            + np.random.uniform(-0.01, 0.02) # Range reduced from (-0.02, 0.06) to (-0.01, 0.02)
        )
        polydispersity = np.clip(pdi, 0.08, 0.7) # Adjusted range


        # --- Calculate Biomedical Metrics (More Realistic) ---

        # Drug Loading Efficiency (%)
        # More dependency on size (surface area ~ 1/size) and stabilizer
        loading = (
            70 # Adjusted base loading
            + 500 / particle_size # Inverse relationship with size (surface area)
            + stabilizer_effects['loading_mod']
            - 20 * polydispersity # Stronger PDI effect
            # REDUCED NOISE FOR DRUG LOADING EFFICIENCY
            + np.random.normal(0, 4) # Noise reduced from 8 to 4
        )
        drug_loading_efficiency = np.clip(loading, 10, 98) # Adjusted range

        # Targeting Efficiency (%)
        # More complex size dependency (stronger peak), zeta dependency, stabilizer, and PDI
        # Optimal size range for EPR effect (e.g., 20-100 nm)
        size_targeting_factor = 50 * np.exp(-((particle_size - 60)**2) / (2 * 30**2)) # Stronger peak around 60nm
        # Zeta potential effect: Moderate negative/positive better than extreme or neutral?
        zeta_targeting_factor = -0.8 * (zeta_potential)**2 / (2 * 25**2) + 15 # Stronger quadratic, optimum near 0, drops off faster
        # Ensure zeta_targeting_factor is not negative
        zeta_targeting_factor = max(0, zeta_targeting_factor)


        targeting = (
            35 # Adjusted base targeting
            + size_targeting_factor
            + zeta_targeting_factor
            + stabilizer_effects['targeting_mod'] * 1.5 # Increased stabilizer effect
            - 30 * polydispersity # Stronger PDI effect
            # NOISE FOR TARGETING WAS ALREADY REDUCED IN THE LAST ITERATION (0, 5)
            + np.random.normal(0, 5)
        )
        targeting_efficiency = np.clip(targeting, 15, 99) # Adjusted range

        # Cytotoxicity (%)
        # Stronger stabilizer effect, size effect, and potential interaction
        toxicity = (
            20 # Adjusted base toxicity
            + 100 * np.exp(-(particle_size / 15)**2) # Stronger size effect (smaller is more toxic)
            + stabilizer_effects['cytotoxicity_mod'] # Stabilizer specific high/low toxicity
            + 0.8 * precursor_conc # Stronger conc effect
            + 0.2 * (temperature - 20) # Stronger temp effect
            # Interaction: CTAB toxicity might be worse with smaller particles or higher conc
            + (15 if stabilizer == "CTAB" and particle_size < 50 else 0)
            + (10 if stabilizer == "CTAB" and precursor_conc > 3 else 0)
            # REDUCED NOISE FOR CYTOTOXICITY
            + np.random.normal(0, 4) # Noise reduced from 8 to 4
        )
        cytotoxicity = np.clip(toxicity, 5, 99) # Adjusted range


        # Append to dataset
        data.append([
            precursor_conc,
            reducing_agent,
            stabilizer,
            pH,
            temperature,
            reaction_time,
            mixing_speed,
            particle_size,
            zeta_potential,
            polydispersity,
            drug_loading_efficiency,
            targeting_efficiency,
            cytotoxicity,
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

# Define the file path for the original data
output_csv_path_original = "aunp_synthesis_cancer_treatment_v3_original.csv"

# Save original data to CSV (optional, but good for comparison)
# Use os.path.join for cross-platform compatibility
output_csv_path_original_full = os.path.join(os.getcwd(), output_csv_path_original)
df_v3.to_csv(output_csv_path_original_full, index=False)

print(f"Refined dataset (v3 - Further Reduced Noise) created with {n_samples} samples.")
print("Further reduced noise for Particle Size, Zeta Potential, Polydispersity, Drug Loading, and Cytotoxicity.")
print(f"Original dataset saved to {output_csv_path_original_full}")
print(df_v3.head())
print("\nSummary Statistics (v3 - Further Reduced Noise):")
print(df_v3.describe())

# --- Apply Transformers ---

# Define input features
input_features = ["Precursor_Conc_mM", "Reducing_Agent", "Stabilizer", "pH",
                  "Temperature_C", "Reaction_Time_min", "Mixing_Speed_RPM"]

# Define ALL output targets (both scaled and passthrough)
output_targets = ["Particle_Size_nm", "Zeta_Potential_mV", "Polydispersity",
                  "Drug_Loading_Efficiency_%", "Targeting_Efficiency_%", "Cytotoxicity_%"]

# Output targets for the model that WILL be scaled
# Particle_Size_nm is treated as a passthrough feature in the input transformer
output_targets_to_scale = ["Zeta_Potential_mV", "Polydispersity",
                           "Drug_Loading_Efficiency_%", "Targeting_Efficiency_%", "Cytotoxicity_%"]

# Identify features from the *original* dataframe that should NOT be transformed (passthrough)
# This includes the Particle_Size_nm target which is used as an input feature for biomedical targets
passthrough_features = ["Precursor_Conc_mM", "pH", "Temperature_C", "Reaction_Time_min", "Particle_Size_nm"]

# Identify features from the *input_features* list that SHOULD be transformed
categorical_input_features = ['Reducing_Agent', 'Stabilizer']
numerical_input_to_scale = ["Mixing_Speed_RPM"] # Only Mixing_Speed_RPM is scaled here

# Create the preprocessor for input features AND the passthrough output feature (Particle_Size_nm)
# This transformer prepares the X data for the model
preprocessor_X = ColumnTransformer(
    transformers=[
        ('num_minmax', MinMaxScaler(feature_range=(0, 1)), numerical_input_to_scale), # Scale Mixing Speed
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_input_features) # One-hot encode categoricals
    ],
    remainder='passthrough', # Pass through all features not listed above (includes passthrough_features)
    verbose_feature_names_out=False # Use this for cleaner output column names
)

# Select the columns that will be processed by preprocessor_X
columns_for_preprocessor_X = numerical_input_to_scale + categorical_input_features + passthrough_features

# Fit and transform these selected columns from the original dataframe
X_transformed_array = preprocessor_X.fit_transform(df_v3[columns_for_preprocessor_X])

# Convert transformed X_transformed_array back to a DataFrame for easier inspection and saving
X_transformed_columns = preprocessor_X.get_feature_names_out()
X_transformed_df = pd.DataFrame(X_transformed_array, columns=X_transformed_columns)


# Create a scaler specifically for the *actual* output targets that need scaling
# This scaler prepares the y data for the model
target_scaler = MinMaxScaler(feature_range=(0, 1)) # Scales to the range [0, 1]

# Fit and transform the output targets that need scaling
y_to_scale_df = df_v3[output_targets_to_scale] # Select the columns to scale
y_scaled_transformed_array = target_scaler.fit_transform(y_to_scale_df)

# Convert transformed y_scaled_transformed_array back to a DataFrame for easier inspection and saving
y_scaled_transformed_df = pd.DataFrame(y_scaled_transformed_array, columns=output_targets_to_scale)


print("\n--- Data After Transformation ---")
print("Shape of transformed inputs and passthrough features (X_transformed_df):", X_transformed_df.shape)

# Round the 'Mixing_Speed_RPM' column to 4 decimal places for display
if 'Mixing_Speed_RPM' in X_transformed_df.columns:
    X_transformed_df['Mixing_Speed_RPM'] = X_transformed_df['Mixing_Speed_RPM'].round(4)

print("\nTransformed Input Features and Passthrough (first 5 rows):")
print(X_transformed_df.head())

print("\nShape of original output targets to scale (y_to_scale_df):", y_to_scale_df.shape)
print("Shape of transformed scaled output targets (y_scaled_transformed_df):", y_scaled_transformed_df.shape)

# The targets are now consistently scaled between 0 and 1 for training.
# Multiplication by 100 will be done during evaluation after inverse transformation.

print("\nTransformed Scaled Output Targets (0-1 range) (first 5 rows):")
print(y_scaled_transformed_df.head())


# --- Combine and Save Transformed Data ---

# Combine the transformed input features + passthrough features with the scaled output targets.
# Note: The columns in X_transformed_df are the model's inputs.
# The columns in y_scaled_transformed_df are the model's outputs (targets).
# We are saving them together in one CSV for convenience, but remember the split for training.
transformed_df = pd.concat([X_transformed_df, y_scaled_transformed_df], axis=1)


# Define the file path for the transformed data
output_csv_path_transformed = "aunp_synthesis_cancer_treatment_v3_transformed.csv"

# Save the combined transformed data to CSV
output_csv_path_transformed_full = os.path.join(os.getcwd(), output_csv_path_transformed)
transformed_df.to_csv(output_csv_path_transformed_full, index=False)

print(f"\nCombined transformed dataset saved to {output_csv_path_transformed_full}")

# Print the columns of the final transformed DataFrame to confirm their presence
print("\nColumns in the final transformed dataset:")
print(transformed_df.columns.tolist())

# --- Optional: Basic Visualization ---
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(df_v3['Particle_Size_nm'], bins=50)
plt.title('Particle Size Distribution')
plt.xlabel('Size (nm)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(df_v3['Zeta_Potential_mV'], bins=50)
plt.title('Zeta Potential Distribution')
plt.xlabel('Zeta Potential (mV)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(df_v3['Polydispersity'], bins=50)
plt.title('Polydispersity Distribution')
plt.xlabel('PDI')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.scatter(df_v3['Particle_Size_nm'], df_v3['Targeting_Efficiency_%'], alpha=0.1)
plt.title('Targeting Efficiency vs Size')
plt.xlabel('Size (nm)')
plt.ylabel('Targeting Efficiency (%)')

plt.tight_layout()
plt.show()
