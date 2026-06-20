# AuNP Synthesis Optimization (Deep Learning + Deterministic Rule Engine)

This project combines machine learning (Deep Learning with Keras) with a **deterministic rule engine** to analyze and optimize Gold Nanoparticle (AuNP) synthesis for cancer treatment applications.

The core idea is to:

1.  Train a predictive model to forecast key AuNP properties based on synthesis parameters.
2.  Score each predicted property against predefined suitability criteria — the Δₖ deviation tells us *which* property is out of range and *in which direction*.
3.  Map each deviation to a **pre-validated optimization recommendation**, keyed by *(synthesis method, property, direction of deviation)*. Every recommendation is written and approved in advance by a domain chemist and ships with a one-line mechanistic justification.

> **Why a rule engine?** Earlier versions sent predictions to an external generative model to produce optimization advice. That carried hallucination risk, API cost/latency, and could suggest chemistry inconsistent with the actual synthesis method. The lookup table is **traceable** (every cell cites a mechanism), **zero-cost / zero-latency** (no API dependency), and makes method-inconsistency *structurally impossible*. The trade-off — by design — is that on a truly novel combination the table degrades gracefully to *"no rule found; flagged for chemist review"* rather than guessing. See [`recommender.py`](recommender.py). An optional explanation layer can expand the pre-approved recommendations into a fuller written explanation; it is tightly grounded in the table and never decides chemistry.

## Features

* **Data Loading and Analysis:** Loads and performs initial analysis on AuNP synthesis data.
* **Correlation Matrix Plotting:** Generates and saves a correlation matrix heatmap to visualize relationships between features and target properties.
* **Deep Learning Model Training:** Trains a Keras Deep Learning model to predict AuNP properties (Particle Size, Zeta Potential, Drug Loading Efficiency, Targeting Efficiency, Cytotoxicity) from synthesis parameters.
* **XGBoost Model Training (Optional):** Includes code for training an XGBoost model as an alternative or comparison (based on `Keras.py`).
* **DCN Model Training (Optional):** Includes code for training a DCN model as an alternative or comparison (based on `Keras.py`).
* **MLP Model Training (Optional):** Includes code for training a MLP model as an alternative or comparison (based on `Keras.py`).
* **Model and Scaler Saving:** Saves the trained Keras model, input scaler (`scaler_X`), and output scaler (`scaler_y`) using Keras native format and `joblib`.
* **Feature Insights and Explanation Generation:** Analyzes feature statistics and correlations to create a detailed explanation text.
* **Deterministic Recommendation Engine ([`recommender.py`](recommender.py)):**
    * **Synthesis-method inference:** Derives the method (Turkevich, Seed-Mediated, Brust-Schiffrin, Polyol/Green) from the chosen reducer/stabilizer — works on both plain categorical and one-hot–encoded inputs.
    * **Δₖ deviation scoring:** Ranks out-of-range properties by range-normalized severity.
    * **Pre-validated lookup:** Maps *(method, property, direction)* to a chemist-written recommendation + mechanism.
    * **Multi-deviation handling:** Concatenates the top recommendations by severity and adds a deterministic synergy/conflict note (do two fixes reinforce, e.g. *"both call for more citrate"*, or trade off?).
    * **Reconciliation with the success classifier:** Cross-checks the model's success prediction so the advice never contradicts it (see below).
* **Optional detailed explanation:** Expands the rule-based recommendations with deeper mechanism and practical lab guidance. The recommendations themselves stay rule-based and unchanged.
* **Single Sample Prediction and Analysis:** Lets users enter synthesis parameters for a single sample, get property predictions from the ML model, and receive rule-based optimization recommendations.
* **Environment Variable Loading:** Uses `python-dotenv` to load configuration from a `.env` file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Champion2049/LLM_Transformer_Model.git](https://github.com/Champion2049/LLM_Transformer_Model.git)
    cd LLM_Transformer_Model
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow keras joblib requests python-dotenv matplotlib seaborn optuna xgboost
    ```

4.  **Obtain a Groq API Key:**
    * Sign up for Groq Cloud at <https://console.groq.com/keys>.
    * Generate a new API key.

5.  **Create a `.env` file:**
    * In the root directory of the project (where `classification_model_direct_groq.py` is located), create a file named `GROQ_API_KEY.env`.
    * Add the following line to the file, replacing `<your_actual_groq_api_key>` with the key you obtained in the previous step:
        ```
        GROQ_API_KEY="<your_actual_groq_api_key>"
        ```

6.  **Place your data file:**
    * Ensure your transformed data file (`aunp_synthesis_cancer_treatment_v3_transformed.csv`) is located at the path specified by `DATA_FILE_PATH` in the scripts. Update the `DATA_FILE_PATH` variable in `Keras.py`, `LLM.py` and `Home.py` if your file is located elsewhere.

## Usage

### 1. Training the ML Models and Generating Analysis Files

Run the `Keras.py` script first. This script will:

* Load the data.
* Split the data into training and validation sets.
* Scale the data using `MinMaxScaler` and save the fitted scalers (`scaler_X_*.joblib`, `scaler_y_*.joblib`).
* Train the Keras Deep Learning model and save the best version (`best_dl_model_*.keras`).
* (Optional) Run Optuna optimization and train an XGBoost model, saving the model and hyperparameters (`aunp_xgboost_model_*.joblib`, `best_hyperparameters_xgboost_*.txt`).
* Generate and save feature insights (`feature_insights_*.json`) and the synthesis explanation text (`aunp_synthesis_explanation_*.md`).
* Plot and save R² scores for both models.

```bash
python Keras.py
```

Ensure this script runs successfully and generates the necessary files in the `saved_models` directory. Note the timestamps in the generated filenames, as the prediction script will look for the latest ones.

---

### 2. Predicting Properties and Getting Recommendations (CLI)

Run the `LLM.py` script. This script will:

* Load the latest trained Keras model and scalers from the `saved_models` directory.
* Load the latest feature insights and explanation text.
* Plot and save the correlation matrix of your data.
* Prompt you to enter synthesis parameters for a single sample.
* Use the loaded Keras model to predict the resulting AuNP properties.
* Perform a suitability classification of the predicted properties (a separate remote call).
* Generate **deterministic optimization recommendations** from the rule engine in [`recommender.py`](recommender.py) — this step makes no external call.
* Print the predicted properties, suitability classification, and optimization recommendations to the console.

```bash
python LLM.py
```

---


### 3. Using the UI Interface

Run `Home.py` using streamlit. This will start up a local server to host the UI for the implementation.

```bash
streamlit run Home.py
```

Follow the prompts to enter your desired synthesis parameters.

---

## File Structure

```
.
├── Keras.py                          # Script for training ML models and generating analysis files
├── classification_model_direct_groq.py # Script for single sample prediction and LLM analysis
├── GROQ_API_KEY.env                  # File to store your Groq API key (create this file)
├── aunp_synthesis_cancer_treatment_v3_transformed.csv # Your dataset (ensure path is correct)
└── saved_models/                     # Directory to store saved models, scalers, and analysis outputs
    ├── best_dl_model_YYYYMMDD_HHMMSS.keras
    ├── scaler_X_YYYYMMDD_HHMMSS.joblib
    ├── scaler_y_YYYYMMDD_HHMMSS.joblib
    ├── aunp_synthesis_explanation_YYYYMMDD_HHMMSS.md
    ├── feature_insights_YYYYMMDD_HHMMSS.json
    ├── correlation_matrix_YYYYMMDD_HHMMSS.png
    ├── aunp_xgboost_model_YYYYMMDD_HHMMSS.joblib  # (If XGBoost training is run)
    ├── best_hyperparameters_xgboost_YYYYMMDD_HHMMSS.txt # (If XGBoost training is run)
    └── llm_classification_results_YYYYMMDD_HHMMSS.txt # (If test set evaluation is re-enabled)
```
*(Filenames in `saved_models` will include timestamps)*

---

## Models Used

- **Deep Learning Model**: A Sequential model implemented with TensorFlow/Keras for multi-output regression.
- **XGBoost Model (Optional)**: An XGBoost Regressor wrapped in `MultiOutputRegressor` for multi-output regression.
- **Recommendation Engine**: A deterministic, chemist-validated *(method, property, direction)* lookup table ([`recommender.py`](recommender.py)) that generates optimization recommendations — no external service, no API.
- **Detailed explanation (optional)**: If enabled, a natural-language layer expands the rule-based recommendations into a fuller written explanation. It is tightly grounded in the lookup table and never changes the chemistry. The legacy CLI ([`LLM.py`](LLM.py)) additionally performs a separate suitability-classification step via a remote call.

---

## Suitability Criteria

For the Streamlit UI's **"Cancer Treatment (General)"** use-case, the ideal ranges are **data-derived** — the 5th–95th percentile of the `Successful_Treatment == 1` subset (n = 4639), regenerated by [`derive_criteria.py`](derive_criteria.py):

- `Particle_Size_nm`: **(2.0, 28.0)**
- `Particle_Width_nm`: **(2.0, 28.0)**
- `Drug_Loading_Efficiency`: **(42.6, 52.9)**
- `Targeting_Efficiency`: **(38.8, 58.5)**
- `Cytotoxicity`: **(7.0, 18.4)**

This grounding matters: with the original hand-picked ranges, only **82%** of "all-in-range" syntheses were actually successful and the ranges captured just **48%** of true successes — so the recommender could report "no adjustments required" next to a model verdict of *Unsuccessful*. The derived ranges raise success-recall to **69%** at comparable precision, so "in range" now lines up with the model. The other two UI use-cases (Targeted Drug Delivery, Bio-imaging) have **no matching success label in this dataset**, so their ranges remain literature targets and are flagged as unvalidated in the UI.

> The legacy CLI ([`LLM.py`](LLM.py)) still uses its own historical criteria (`Particle_Size_nm` (10,100), `Zeta_Potential_mV` (-30,-5), `Drug_Loading_Efficiency_%` (70,100), `Targeting_Efficiency_%` (75,100), `Cytotoxicity_%` (70,90)) for the separate `*_transformed.csv` dataset.

---

## Recommendation Engine

Once the Δₖ deviation score identifies *which* property is out of range and *in which direction*, the recommendation is a deterministic lookup:

```
(Synthesis Method, Property, Direction of deviation) → pre-validated, chemist-written recommendation
```

The synthesis method is inferred from the reducer/stabilizer (citrate → Turkevich, ascorbic acid → Seed-Mediated, NaBH₄ → Brust-Schiffrin, ethylene glycol / extract → Polyol/Green). A few representative rows from the table in [`recommender.py`](recommender.py):

| Method | Property off | Direction | Recommendation | Mechanism |
| --- | --- | --- | --- | --- |
| Turkevich | Particle Size | Too large | Increase citrate-to-gold ratio | More citrate → more nucleation sites → smaller particles |
| Turkevich | Zeta Potential | Not negative enough | Increase citrate or raise pH | More adsorbed citrate anions strengthen negative charge |
| Seed-Mediated | Aspect Ratio | Too low | Increase AgNO₃ relative to seed | Silver underpotential deposition directs anisotropic growth |
| Seed-Mediated | Zeta Potential | Too positive (CTAB) | CTAB→PEG-thiol ligand exchange | Removes the cationic CTAB bilayer |
| Brust-Schiffrin | Particle Size | Too large (>5 nm) | Increase thiol-to-gold ratio | More capping ligand terminates growth earlier |
| Brust-Schiffrin | Cytotoxicity | Too high | Reduce free thiol; add wash step | Unbound thiol byproducts are a known toxicity source |
| Polyol/Green | PDI | Too high (batch variability) | Standardize/filter extract; raise extract ratio | Reduces variability inherent to natural extracts |

**Multiple simultaneous deviations** are sorted by |Δₖ| and the top recommendations are concatenated, followed by an automatically derived synergy/conflict note (e.g. *"several fixes point the same way — increasing citrate — so a single change addresses more than one deviation"*). This note is computed from set logic over the reagents each recommendation adjusts; no generative model is involved.

**Reconciliation with the success classifier.** The engine also takes the model's binary success prediction. Because the per-property ranges are necessary but not sufficient (they can't capture every parameter interaction), the report explicitly reconciles the two: if the model predicts *Unsuccessful* yet every property is in range, it says so honestly ("in-range values are necessary but not sufficient … revisit the route holistically") instead of claiming "no adjustments required". An audit over 400 real model predictions found **zero** silent contradictions.

Run `python recommender.py` to see the engine's worked examples (including the reconciliation cases) without launching the UI.

---

## Authors & Contributors
[Chirayu Chaudhari](https://github.com/Champion2049)  
[Janya Billa](https://github.com/janya26)  
[Manasvini Kandikonda](https://github.com/vini-ai2)  

---

## Contributing

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

