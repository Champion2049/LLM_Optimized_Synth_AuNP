import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import time
import os
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor # Strategy for multi-output

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt

# For saving/loading models and scalers
import joblib

# --- Configuration ---
# MAKE SURE THIS PATH IS CORRECT FOR YOUR SYSTEM
DATA_FILE_PATH = r"C:\Users\Chirayu\Desktop\Coding\IMI\aunp_synthesis_cancer_treatment_v3_transformed.csv"
TARGET_COLS = ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%']
TEST_SIZE = 0.2
RANDOM_STATE = 42
OPTUNA_N_TRIALS = 50 # Number of trials for XGBoost hyperparameter search (increase for better results)
MODEL_DIR = "saved_models" # Changed directory name to be more general

# --- Create output directory ---
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# --- Load Data ---
print(f"Loading data from: {DATA_FILE_PATH}")
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE_PATH}")
    exit() # Exit if data file is not found

# --- Feature and Target Split ---
X = df.drop(columns=TARGET_COLS)
y = df[TARGET_COLS]
print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")

# --- Scaling ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
print("Data scaling applied.")

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Generate a timestamp for saving files
timestamp = time.strftime("%Y%m%d_%H%M%S")

# --- Save the Scalers ---
# Scalers are used by both models, save them once
scaler_x_path = os.path.join(MODEL_DIR, f"scaler_X_{timestamp}.joblib")
scaler_y_path = os.path.join(MODEL_DIR, f"scaler_y_{timestamp}.joblib")
try:
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"Scaler X saved to: {scaler_x_path}")
    print(f"Scaler y saved to: {scaler_y_path}")
except Exception as e:
    print(f"Error saving scalers: {e}")


# ==============================================================================
# --- Deep Learning Model Implementation (using Keras) ---
# ==============================================================================

print("\n" + "="*30)
print("--- Setting up and Training Deep Learning Model ---")
print("="*30 + "\n")

# Define the model architecture
# This is a starting point; architecture tuning is crucial!
# Use the functional API or Sequential depending on complexity. Sequential is simpler here.
dl_model = Sequential([
    # Input layer: needs input_shape = number of features
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)), # Increased neurons
    BatchNormalization(), # Helps stabilize training
    Dropout(0.3), # Added dropout for regularization

    # Hidden layer 1
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden layer 2
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2), # Slightly less dropout

    # Output layer: Number of neurons = number of targets, linear activation for regression
    Dense(len(TARGET_COLS), activation='linear')
])

# Compile the model
# Using Adam optimizer, common learning rate
optimizer = Adam(learning_rate=0.001) # You can tune this
dl_model.compile(optimizer=optimizer,
                 loss='mean_squared_error',
                 metrics=['mean_squared_error', 'mean_absolute_error'])

print("Deep Learning Model Summary:")
dl_model.summary()

# Define callbacks for training
# Early Stopping: Stop training if validation loss doesn't improve
early_stopping_dl = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True) # Increased patience
# Model Checkpoint: Save the best model during training based on validation loss
best_dl_model_filepath = os.path.join(MODEL_DIR, f"best_dl_model_{timestamp}.keras")
model_checkpoint_dl = ModelCheckpoint(
    best_dl_model_filepath, # Keras native format (.keras)
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1 # Show when saving
)
# Reduce Learning Rate on Plateau: Reduce learning rate if validation loss plateaus
reduce_lr_dl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)


# Train the model
print("\nTraining Deep Learning model...")
start_dl_train_time = time.time()

history = dl_model.fit(X_train, y_train,
                       epochs=1000, # Set a high number, EarlyStopping will stop it
                       batch_size=64, # Batch size (experiment with this: 32, 64, 128)
                       validation_data=(X_val, y_val),
                       callbacks=[early_stopping_dl, model_checkpoint_dl, reduce_lr_dl],
                       verbose=1) # Show training progress

dl_train_duration = time.time() - start_dl_train_time
print(f"Deep Learning model training finished in: {str(timedelta(seconds=int(dl_train_duration)))}")


# --- Evaluate Final Deep Learning Model ---
print("\n" + "="*30)
print("--- Evaluating Final Deep Learning Model ---")
print("="*30 + "\n")

# Load the best model saved by ModelCheckpoint for evaluation
if os.path.exists(best_dl_model_filepath):
    print(f"\nLoading best DL model from: {best_dl_model_filepath}")
    loaded_best_dl_model = tf.keras.models.load_model(best_dl_model_filepath)
else:
     # This case should ideally not happen if training ran successfully with ModelCheckpoint
    print("\nBest DL model not found at checkpoint path, evaluating the last state of the trained model.")
    loaded_best_dl_model = dl_model # Fallback

# Predict on validation set (scaled) using the best loaded model
y_pred_val_scaled_dl = loaded_best_dl_model.predict(X_val)

# Inverse transform predictions and true values to original scale
y_pred_val_inv_dl = scaler_y.inverse_transform(y_pred_val_scaled_dl)
y_val_inv = scaler_y.inverse_transform(y_val) # Use the same original y_val_inv

# Calculate R² scores for each target on the original scale
r2_scores_list_dl = []
print("\n--- Deep Learning R² Scores per Target ---")
for i, target_name in enumerate(TARGET_COLS):
    r2_dl = r2_score(y_val_inv[:, i], y_pred_val_inv_dl[:, i])
    r2_scores_list_dl.append(r2_dl)
    print(f"{target_name}: {r2_dl:.4f}")

# --- Plot Deep Learning R² Scores ---
plt.figure(figsize=(10, 6))
bars_dl = plt.bar(TARGET_COLS, r2_scores_list_dl, color='teal') # Use a different color
# Dynamic ylim based on scores, ensuring 0 is included if scores are positive
min_score_dl = min(r2_scores_list_dl + [0])
max_score_dl = max(r2_scores_list_dl + [1])
plt.ylim(min_score_dl - 0.05, max_score_dl + 0.05)

plt.title('Deep Learning Model - R² Score per Target (Validation Set)')
plt.ylabel('R² Score')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add R² values on top of bars
for bar, score in zip(bars_dl, r2_scores_list_dl):
     # Position text slightly above the bar top
    y_text_pos = score + 0.01 if score >= 0 else score - 0.05 # Adjust for negative scores
    plt.text(bar.get_x() + bar.get_width() / 2, y_text_pos,
             f'{score:.2f}', ha='center', va='bottom' if score >= 0 else 'top', fontsize=9)

plt.tight_layout()
plot_path_dl = os.path.join(MODEL_DIR, f"r2_scores_deep_learning_{timestamp}.png")
try:
    plt.savefig(plot_path_dl)
    print(f"Deep Learning R² score plot saved to: {plot_path_dl}")
except Exception as e:
    print(f"Error saving DL plot: {e}")
# plt.show() # Show plot if needed, but saving is done

# --- Example of Loading the Saved DL Model ---
print("\n--- Example: Loading Saved DL Model ---")
print("To load the saved Keras model later:")
print(f"loaded_dl_model = tf.keras.models.load_model('{best_dl_model_filepath}')")
# Remember you still need to load the scalers using joblib as before
print(f"loaded_scaler_X = joblib.load('{scaler_x_path}')")
print(f"loaded_scaler_y = joblib.load('{scaler_y_path}')")


# ==============================================================================
# --- Optuna Hyperparameter Optimization & XGBoost Model ---
# ==============================================================================

print("\n\n" + "="*30)
print("--- Starting XGBoost Optuna Optimization ---")
print("="*30 + "\n")


# We use MultiOutputRegressor, which trains one XGBoost model per target.
# The objective function aims to minimize the average Mean Squared Error across all targets.
def objective(trial):
    """Optuna objective function for XGBoost hyperparameter tuning."""

    # Define hyperparameters to tune
    param = {
        'objective': 'reg:squarederror', # Regression task with squared error
        'eval_metric': 'rmse',           # Evaluation metric (RMSE is sqrt of MSE)
        'booster': 'gbtree',             # Tree-based booster
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100), # Increased range
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True), # Adjusted range
        'max_depth': trial.suggest_int('max_depth', 3, 12), # Increased range
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # Adjusted range
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # Adjusted range
        'gamma': trial.suggest_float('gamma', 0, 10), # Adjusted range
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # Adjusted range
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # Adjusted range
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # Added
        'random_state': RANDOM_STATE,
        'n_jobs': -1 # Use all available CPU cores
    }

    # Create an XGBoost Regressor model instance
    xgb_regressor = xgb.XGBRegressor(**param)

    # Wrap it with MultiOutputRegressor
    multioutput_model = MultiOutputRegressor(xgb_regressor)

    # Train the model
    # No early stopping within MultiOutputRegressor fit directly based on validation set
    # MultiOutputRegressor fits each estimator independently
    multioutput_model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred_val = multioutput_model.predict(X_val)

    # Calculate Mean Squared Error on the validation set
    # This is the metric Optuna will minimize (average across targets)
    mse = mean_squared_error(y_val, y_pred_val)

    return mse

# --- Run Optuna Hyperparameter Optimization ---
print(f"\nStarting XGBoost hyperparameter optimization with Optuna ({OPTUNA_N_TRIALS} trials)...")
start_opt_time = time.time()

study = optuna.create_study(direction="minimize") # We want to minimize MSE
study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True) # Set show_progress_bar=False if not desired

opt_duration = time.time() - start_opt_time
print(f"\nXGBoost Optimization finished in: {str(timedelta(seconds=int(opt_duration)))}")
print("Best XGBoost trial score (MSE):", study.best_trial.value)
print("Best XGBoost hyperparameters:", study.best_trial.params)

# --- Train Final XGBoost Model with Best Hyperparameters ---
print("\nTraining final XGBoost model with best hyperparameters...")
start_train_time = time.time()

best_params_xgb = study.best_trial.params
# Add fixed parameters back
best_params_xgb['objective'] = 'reg:squarederror'
best_params_xgb['eval_metric'] = 'rmse'
best_params_xgb['booster'] = 'gbtree'
best_params_xgb['random_state'] = RANDOM_STATE
best_params_xgb['n_jobs'] = -1

# Create and train the final model
final_xgb_regressor = xgb.XGBRegressor(**best_params_xgb)
final_model_xgb = MultiOutputRegressor(final_xgb_regressor)

# Fit on the entire training data
final_model_xgb.fit(X_train, y_train)

train_duration_xgb = time.time() - start_train_time
print(f"Final XGBoost model training finished in: {str(timedelta(seconds=int(train_duration_xgb)))}")

# --- Save the Final XGBoost Model and Hyperparameters ---
# Note: XGBoost models can be saved directly, and MultiOutputRegressor pickles the base estimators.
# We'll save the MultiOutputRegressor wrapper.
model_path_xgb = os.path.join(MODEL_DIR, f"aunp_xgboost_model_{timestamp}.joblib")
hyperparams_path_xgb = os.path.join(MODEL_DIR, f"best_hyperparameters_xgboost_{timestamp}.txt")

try:
    joblib.dump(final_model_xgb, model_path_xgb)
    with open(hyperparams_path_xgb, 'w') as f:
        f.write(str(best_params_xgb))
    print(f"Final XGBoost model saved to: {model_path_xgb}")
    print(f"Best XGBoost hyperparameters saved to: {hyperparams_path_xgb}")
except Exception as e:
    print(f"Error saving XGBoost model/hyperparameters: {e}")


# --- Evaluate Final XGBoost Model ---
print("\n" + "="*30)
print("--- Evaluating Final XGBoost Model ---")
print("="*30 + "\n")

# Predict on validation set (scaled)
y_pred_val_scaled_xgb = final_model_xgb.predict(X_val)

# Inverse transform predictions and true values to original scale
y_pred_val_inv_xgb = scaler_y.inverse_transform(y_pred_val_scaled_xgb)
# y_val_inv is already computed once

# Calculate R² scores for each target on the original scale
r2_scores_list_xgb = []
print("\n--- XGBoost R² Scores per Target ---")
for i, target_name in enumerate(TARGET_COLS):
    r2_xgb = r2_score(y_val_inv[:, i], y_pred_val_inv_xgb[:, i])
    r2_scores_list_xgb.append(r2_xgb)
    print(f"{target_name}: {r2_xgb:.4f}")

# --- Plot XGBoost R² Scores ---
plt.figure(figsize=(10, 6))
bars_xgb = plt.bar(TARGET_COLS, r2_scores_list_xgb, color='lightcoral') # Original color
# Dynamic ylim
min_score_xgb = min(r2_scores_list_xgb + [0])
max_score_xgb = max(r2_scores_list_xgb + [1])
plt.ylim(min_score_xgb - 0.05, max_score_xgb + 0.05)

plt.title('XGBoost Model - R² Score per Target (Validation Set)')
plt.ylabel('R² Score')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add R² values on top of bars
for bar, score in zip(bars_xgb, r2_scores_list_xgb):
    # Position text slightly above the bar top
    y_text_pos = score + 0.01 if score >= 0 else score - 0.05 # Adjust for negative scores
    plt.text(bar.get_x() + bar.get_width() / 2, y_text_pos,
             f'{score:.2f}', ha='center', va='bottom' if score >= 0 else 'top', fontsize=9)

plt.tight_layout()
plot_path_xgb = os.path.join(MODEL_DIR, f"r2_scores_xgboost_{timestamp}.png")
try:
    plt.savefig(plot_path_xgb)
    print(f"XGBoost R² score plot saved to: {plot_path_xgb}")
except Exception as e:
    print(f"Error saving XGBoost plot: {e}")
# plt.show() # Show plot if needed

# --- Example of Loading the Saved XGBoost Model ---
print("\n--- Example: Loading Saved XGBoost Model ---")
print("To load the saved XGBoost model later:")
print(f"loaded_model_xgb = joblib.load('{model_path_xgb}')")
# Remember you still need to load the scalers using joblib as before
print(f"loaded_scaler_X = joblib.load('{scaler_x_path}')")
print(f"loaded_scaler_y = joblib.load('{scaler_y_path}')")


# --- Final Comparison (Optional) ---
print("\n--- Comparison of R² Scores ---")
print("Target Name         | XGBoost R² | Deep Learning R²")
print("--------------------|------------|-----------------")
for i, target_name in enumerate(TARGET_COLS):
    print(f"{target_name:<19}| {r2_scores_list_xgb[i]:<11.4f}| {r2_scores_list_dl[i]:<.4f}")

# Display plots if you want to see them automatically
plt.show()


print("\n--- Script Finished ---")