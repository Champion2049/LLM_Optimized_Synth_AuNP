# Gold Nanoparticle (AuNP) Synthesis for Cancer Treatment

## Optimal Output Criteria for Cancer Treatment

- **Particle_Size_nm**: Should be between 40 and 100. Optimal particle size range for gold nanoparticles in cancer treatment. Particles should be large enough to carry drug payload but small enough to penetrate tumor tissue via the EPR effect.
- **Zeta_Potential_mV**: Should be between -30 and -5. Zeta potential indicates surface charge and stability. Slightly negative values promote stability while facilitating cellular uptake.
- **Drug_Loading_Efficiency_%**: Should be between 70 and 100. Indicates how efficiently the drug is loaded onto nanoparticles. Higher values mean more effective drug delivery.
- **Targeting_Efficiency_%**: Should be between 75 and 100. Measures how well nanoparticles target cancer cells. Higher values indicate better specificity for cancer cells.
- **Cytotoxicity_%**: Should be between 70 and 90. Indicates toxicity to cancer cells. Should be high enough to effectively kill cancer cells but not excessively toxic.

## Key Feature Relationships

### Factors influencing Particle_Size_nm:

**Features that tend to increase this value:**
- Temperature_C (correlation: 0.789)
- Reducing_Agent_citrate (correlation: 0.238)
- Precursor_Conc_mM (correlation: 0.158)
- Polydispersity (correlation: 0.152)
- Stabilizer_CTAB (correlation: 0.118)

**Features that tend to decrease this value:**
- Reaction_Time_min (correlation: -0.471)
- Reducing_Agent_NaBH4 (correlation: -0.258)
- Mixing_Speed_RPM (correlation: -0.244)
- Stabilizer_citrate (correlation: -0.163)
- Stabilizer_PEG (correlation: -0.024)

### Factors influencing Zeta_Potential_mV:

**Features that tend to increase this value:**
- Stabilizer_CTAB (correlation: 0.861)
- pH (correlation: 0.061)
- Reducing_Agent_NaBH4 (correlation: 0.012)
- Polydispersity (correlation: 0.004)
- Reducing_Agent_ascorbic_acid (correlation: 0.002)

**Features that tend to decrease this value:**
- Stabilizer_PEG (correlation: -0.369)
- Stabilizer_PVP (correlation: -0.245)
- Stabilizer_citrate (correlation: -0.054)
- Reducing_Agent_citrate (correlation: -0.012)
- Temperature_C (correlation: -0.011)

### Factors influencing Drug_Loading_Efficiency_%:

**Features that tend to increase this value:**
- Stabilizer_PEG (correlation: 0.549)
- Reaction_Time_min (correlation: 0.178)
- Stabilizer_PVP (correlation: 0.143)
- Mixing_Speed_RPM (correlation: 0.136)
- Reducing_Agent_NaBH4 (correlation: 0.053)

**Features that tend to decrease this value:**
- Stabilizer_CTAB (correlation: -0.791)
- Temperature_C (correlation: -0.264)
- Polydispersity (correlation: -0.114)
- Stabilizer_citrate (correlation: -0.108)
- Precursor_Conc_mM (correlation: -0.068)

### Factors influencing Targeting_Efficiency_%:

**Features that tend to increase this value:**
- Stabilizer_PEG (correlation: 0.754)
- Temperature_C (correlation: 0.357)
- Reducing_Agent_citrate (correlation: 0.144)
- pH (correlation: 0.025)
- Reducing_Agent_ascorbic_acid (correlation: 0.011)

**Features that tend to decrease this value:**
- Stabilizer_CTAB (correlation: -0.550)
- Stabilizer_citrate (correlation: -0.360)
- Reaction_Time_min (correlation: -0.201)
- Reducing_Agent_NaBH4 (correlation: -0.174)
- Polydispersity (correlation: -0.065)

### Factors influencing Cytotoxicity_%:

**Features that tend to increase this value:**
- Stabilizer_CTAB (correlation: 0.824)
- Reducing_Agent_NaBH4 (correlation: 0.185)
- Mixing_Speed_RPM (correlation: 0.150)
- Reaction_Time_min (correlation: 0.146)
- Precursor_Conc_mM (correlation: 0.083)

**Features that tend to decrease this value:**
- Stabilizer_PEG (correlation: -0.417)
- Stabilizer_PVP (correlation: -0.266)
- Temperature_C (correlation: -0.252)
- Reducing_Agent_citrate (correlation: -0.150)
- pH (correlation: -0.040)

## Feature Statistics (for context)

- **Mixing_Speed_RPM**: Range [0.00 to 1.00], Mean: 0.50, Std: 0.29
- **Reducing_Agent_NaBH4**: Range [0.00 to 1.00], Mean: 0.25, Std: 0.43
- **Reducing_Agent_ascorbic_acid**: Range [0.00 to 1.00], Mean: 0.35, Std: 0.48
- **Reducing_Agent_citrate**: Range [0.00 to 1.00], Mean: 0.40, Std: 0.49
- **Stabilizer_CTAB**: Range [0.00 to 1.00], Mean: 0.15, Std: 0.36
- **Stabilizer_PEG**: Range [0.00 to 1.00], Mean: 0.35, Std: 0.48
- **Stabilizer_PVP**: Range [0.00 to 1.00], Mean: 0.30, Std: 0.46
- **Stabilizer_citrate**: Range [0.00 to 1.00], Mean: 0.20, Std: 0.40
- **Precursor_Conc_mM**: Range [0.10 to 5.00], Mean: 2.55, Std: 1.41
- **pH**: Range [3.00 to 11.00], Mean: 7.00, Std: 2.45
- **Temperature_C**: Range [15.00 to 100.00], Mean: 57.47, Std: 24.82
- **Reaction_Time_min**: Range [5.00 to 133.00], Mean: 15.42, Std: 17.98
- **Polydispersity**: Range [0.00 to 1.00], Mean: 0.48, Std: 0.19
