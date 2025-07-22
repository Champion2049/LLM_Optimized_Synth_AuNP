import pandas as pd
import numpy as np
import time
import os
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import (
    r2_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.utils import class_weight

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import joblib

# --- GPU Configuration Check ---
# Add a check to confirm TensorFlow can see the GPU.
# TensorFlow will automatically use an available GPU without further code changes.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"\n{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.\n")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("\nNo GPU found. The model will run on the CPU.\n")


# --- Configuration ---
# This script uses the RAW (untransformed) dataset to handle all preprocessing internally.
DATA_FILE_PATH = "./aunp_synthesis_realistic_v1.csv" 
MODEL_DIR = "saved_hybrid_models"

# Define the two types of target variables
REGRESSION_TARGETS = [
    'Particle_Size_nm', 'Particle_Width_nm', 'Drug_Loading_Efficiency', 
    'Targeting_Efficiency', 'Cytotoxicity'
]
BINARY_CLASS_TARGET = 'Successful_Treatment'
# Morphology is now an input feature, not a target.


TEST_SIZE = 0.2
RANDOM_STATE = 42

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
    exit()

# --- Feature and Target Split ---
# Define all targets to correctly separate them from features
all_targets = REGRESSION_TARGETS + [BINARY_CLASS_TARGET]
# Drop the constant 'Precursor' and all target columns from the features
# Morphology is now kept in X as an input feature.
X = df.drop(columns=['Precursor'] + all_targets)
y = df[all_targets]


# --- Preprocessing ---
# Morphology will be automatically detected as a categorical feature
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessor for the input features (X)
feature_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- Train/Validation Split (on raw data before processing) ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y[BINARY_CLASS_TARGET]
)

# Fit the preprocessor on the training data and transform both train and validation features
X_train_processed = feature_preprocessor.fit_transform(X_train)
X_val_processed = feature_preprocessor.transform(X_val)

# --- Preprocess the TARGETS (y) ---
# 1. Scale the regression targets
scaler_y_reg = MinMaxScaler()
y_train_reg_scaled = scaler_y_reg.fit_transform(y_train[REGRESSION_TARGETS])
y_val_reg_scaled = scaler_y_reg.transform(y_val[REGRESSION_TARGETS])

# 2. Separate the binary classification target
y_train_binary_class = y_train[BINARY_CLASS_TARGET].values
y_val_binary_class = y_val[BINARY_CLASS_TARGET].values


# Prepare target data for Keras model (a dictionary with a key for each output)
y_train_dict = {
    'regression_output': y_train_reg_scaled, 
    'binary_class_output': y_train_binary_class
}
y_val_dict = {
    'regression_output': y_val_reg_scaled, 
    'binary_class_output': y_val_binary_class
}

# --- Calculate Class Weights to Handle Imbalance ---
class_weights_binary = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_binary_class), y=y_train_binary_class)
class_weight_dict_binary = dict(enumerate(class_weights_binary))
print(f"Binary Class Weights: {class_weight_dict_binary}")


# --- Save Preprocessors ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
joblib.dump(feature_preprocessor, os.path.join(MODEL_DIR, f"feature_preprocessor_{timestamp}.joblib"))
joblib.dump(scaler_y_reg, os.path.join(MODEL_DIR, f"regression_target_scaler_{timestamp}.joblib"))


# ==============================================================================
# --- Custom Loss Functions to Handle Imbalance ---
# ==============================================================================
def weighted_mse_loss(y_true, y_pred):
    """Custom MSE that applies higher penalties for errors on specific regression targets."""
    # Weights for Size, Width, Drug Loading, Targeting, Cytotoxicity
    # Aggressively increased weights for the two targets we want to improve.
    weights = tf.constant([1.0, 1.0, 18.0, 18.0, 2.0]) 
    squared_errors = tf.square(y_true - y_pred)
    weighted_squared_errors = squared_errors * weights
    return tf.reduce_mean(weighted_squared_errors)

def create_weighted_binary_crossentropy(weights_dict):
    """Creates a weighted binary cross-entropy loss function."""
    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weights = y_true * weights_dict[1] + (1.0 - y_true) * weights_dict[0]
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce
    return weighted_binary_crossentropy


# ==============================================================================
# --- Multi-Branch Deep Learning Model ---
# ==============================================================================
print("\n" + "="*30)
print("--- Setting up and Training Multi-Branch Hybrid Model ---")
print("="*30 + "\n")

input_layer = Input(shape=(X_train_processed.shape[1],), name='main_input')

# Shared Foundation Layers (Deeper and Wider)
shared_base = Dense(1024, activation='relu')(input_layer)
shared_base = BatchNormalization()(shared_base)
shared_base = Dropout(0.5)(shared_base)
shared_base = Dense(512, activation='relu')(shared_base)
shared_base = BatchNormalization()(shared_base)
shared_base = Dropout(0.4)(shared_base)
shared_base = Dense(256, activation='relu')(shared_base) 
shared_base = BatchNormalization()(shared_base)
shared_base = Dropout(0.4)(shared_base)


# Branch 1: Regression Expert (Deeper)
regression_branch = Dense(256, activation='relu')(shared_base)
regression_branch = BatchNormalization()(regression_branch)
regression_branch = Dropout(0.4)(regression_branch)
regression_branch = Dense(128, activation='relu')(regression_branch) 
regression_branch = BatchNormalization()(regression_branch)
regression_branch = Dropout(0.3)(regression_branch)
regression_output = Dense(len(REGRESSION_TARGETS), activation='linear', name='regression_output')(regression_branch)

# Branch 2: Binary Classification Expert
binary_class_branch = Dense(128, activation='relu')(shared_base)
binary_class_branch = BatchNormalization()(binary_class_branch)
binary_class_branch = Dropout(0.3)(binary_class_branch)
binary_class_output = Dense(1, activation='sigmoid', name='binary_class_output')(binary_class_branch)


# Create the model
model = Model(inputs=input_layer, outputs=[regression_output, binary_class_output])

# Compile the model with our custom weighted loss functions
losses = {
    'regression_output': weighted_mse_loss,
    'binary_class_output': create_weighted_binary_crossentropy(class_weight_dict_binary)
}
metrics = {
    'regression_output': 'mean_absolute_error',
    'binary_class_output': 'accuracy'
}
# Using a slightly lower learning rate for more stable convergence
model.compile(optimizer=Adam(learning_rate=0.0005), loss=losses, metrics=metrics)

model.summary()

# --- Training ---
# Increased patience to allow for more convergence time
callbacks = [
    EarlyStopping(monitor='val_loss', patience=70, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, f"best_hybrid_model_{timestamp}.keras"), monitor='val_loss', save_best_only=True, mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-7)
]

print("\nTraining Hybrid model with custom weighted losses...")
start_train_time = time.time()
history = model.fit(X_train_processed, y_train_dict,
                    epochs=500,
                    batch_size=128, # Increased batch size
                    validation_data=(X_val_processed, y_val_dict),
                    callbacks=callbacks,
                    verbose=1)
print(f"Hybrid model training finished in: {str(timedelta(seconds=int(time.time() - start_train_time)))}")

# ==============================================================================
# --- Evaluate Final Hybrid Model ---
# ==============================================================================
print("\n" + "="*30)
print("--- Evaluating Final Hybrid Model ---")
print("="*30 + "\n")

best_model_filepath = os.path.join(MODEL_DIR, f"best_hybrid_model_{timestamp}.keras")
print(f"Loading best model from: {best_model_filepath}")
loaded_best_model = tf.keras.models.load_model(
    best_model_filepath, 
    custom_objects={
        'weighted_mse_loss': weighted_mse_loss,
        'weighted_binary_crossentropy': create_weighted_binary_crossentropy(class_weight_dict_binary)
    }
)

y_pred_reg_scaled, y_pred_binary_proba = loaded_best_model.predict(X_val_processed)

# --- Evaluate REGRESSION Output ---
print("\n--- Regression Performance ---")
y_pred_reg_inv = scaler_y_reg.inverse_transform(y_pred_reg_scaled)
y_val_reg_inv = y_val[REGRESSION_TARGETS].values
r2_scores_list = [r2_score(y_val_reg_inv[:, i], y_pred_reg_inv[:, i]) for i in range(len(REGRESSION_TARGETS))]
for i, target_name in enumerate(REGRESSION_TARGETS):
    print(f"R² Score for {target_name}: {r2_scores_list[i]:.4f}")
plt.figure(figsize=(10, 6))
plt.bar(REGRESSION_TARGETS, r2_scores_list, color='mediumpurple')
plt.title('Hybrid Model - R² Score per Regression Target')
plt.ylabel('R² Score')
plt.xticks(rotation=25, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, f"r2_scores_{timestamp}.png"))

# --- Evaluate BINARY CLASSIFICATION Output ---
print("\n\n--- Binary Classification Performance (Successful Treatment) ---")
y_pred_binary_class = (y_pred_binary_proba > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val_binary_class, y_pred_binary_class, target_names=['Unsuccessful', 'Successful'], zero_division=0))
cm_binary = confusion_matrix(y_val_binary_class, y_pred_binary_class)
disp_binary = ConfusionMatrixDisplay(confusion_matrix=cm_binary, display_labels=['Unsuccessful', 'Successful'])
fig, ax = plt.subplots(figsize=(6, 6))
disp_binary.plot(cmap=plt.cm.Greens, ax=ax)
plt.title("Confusion Matrix (Successful Treatment)")
plt.savefig(os.path.join(MODEL_DIR, f"confusion_matrix_binary_{timestamp}.png"))


plt.show()
print("\n--- Script Finished ---")