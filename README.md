# LLM Transformer Model for AuNP Synthesis Optimization

This project combines machine learning (Deep Learning with Keras, potentially XGBoost) with a Large Language Model (LLM) accessed via Groq to analyze and optimize Gold Nanoparticle (AuNP) synthesis for cancer treatment applications.

The core idea is to:

1.  Train a predictive model to forecast key AuNP properties based on synthesis parameters.
2.  Use an LLM to interpret these predicted properties against predefined suitability criteria for cancer treatment.
3.  Leverage the LLM's understanding of the data and criteria to provide intelligent suggestions for optimizing the synthesis method to achieve desired properties.

## Features

* **Data Loading and Analysis:** Loads and performs initial analysis on AuNP synthesis data.
* **Correlation Matrix Plotting:** Generates and saves a correlation matrix heatmap to visualize relationships between features and target properties.
* **Deep Learning Model Training:** Trains a Keras Deep Learning model to predict AuNP properties (Particle Size, Zeta Potential, Drug Loading Efficiency, Targeting Efficiency, Cytotoxicity) from synthesis parameters.
* **XGBoost Model Training (Optional):** Includes code for training an XGBoost model as an alternative or comparison (based on `Keras.py`).
* **DCN Model Training (Optional):** Includes code for training a DCN model as an alternative or comparison (based on `Keras.py`).
* **MLP Model Training (Optional):** Includes code for training a MLP model as an alternative or comparison (based on `Keras.py`).
* **Model and Scaler Saving:** Saves the trained Keras model, input scaler (`scaler_X`), and output scaler (`scaler_y`) using Keras native format and `joblib`.
* **Feature Insights and Explanation Generation:** Analyzes feature statistics and correlations to create a detailed explanation text, used as context for the LLM.
* **LLM Integration (via Groq API):**
    * **Suitability Classification:** Classifies whether predicted AuNP properties are suitable for cancer treatment based on defined criteria, providing a detailed explanation.
    * **Optimization Suggestions:** Suggests specific modifications to synthesis parameters to improve properties towards the ideal ranges.
* **Single Sample Prediction and Analysis:** Allows users to input synthesis parameters for a single sample, get property predictions from the ML model, and receive suitability classification and optimization suggestions from the LLM.
* **Rate Limit Handling:** Includes retry logic with exponential backoff for Groq API calls to manage rate limits.
* **Environment Variable Loading:** Uses `python-dotenv` to load the Groq API key from a `.env` file.

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

### 2. Predicting Properties and Getting LLM Analysis

Run the `LLM.py` script. This script will:

* Load the latest trained Keras model and scalers from the `saved_models` directory.
* Load the latest feature insights and explanation text.
* Plot and save the correlation matrix of your data.
* Prompt you to enter synthesis parameters for a single sample.
* Use the loaded Keras model to predict the resulting AuNP properties.
* Call the Groq API to classify the suitability of the predicted properties and provide an explanation.
* Call the Groq API to suggest optimization steps based on the input parameters and predicted/classified properties.
* Print the predicted properties, suitability classification, explanation, and optimization suggestion to the console.

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
- **Large Language Model**: Accessed via the Groq API (specifically `llama3-8b-8192` or `llama3-70b-8192` as configured) for suitability classification and optimization suggestions.

---

## Suitability Criteria

The LLM classifies suitability based on the following predefined criteria for optimal AuNP properties in cancer treatment:

- `Particle_Size_nm`: Ideal range **(40, 100)**
- `Zeta_Potential_mV`: Ideal range **(-30, -5)**
- `Drug_Loading_Efficiency_%`: Ideal range **(70, 100)**
- `Targeting_Efficiency_%`: Ideal range **(75, 100)**
- `Cytotoxicity_%`: Ideal range **(70, 90)**

---

## Authors & Contributors
[Chirayu Chaudhari](https://github.com/Champion2049)
[Janya Billa](https://github.com/YourUsername)
[Manasvini Kandikonda](https://github.com/YourUsername)

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

