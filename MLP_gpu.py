import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

import optuna
import matplotlib.pyplot as plt
import joblib

# --- Configuration ---
# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define file paths and model directory
DATA_FILE_PATH = "./aunp_synthesis_realistic_v1.csv"
MODEL_DIR = "saved_pytorch_models"

# Define the model's output targets
REGRESSION_TARGETS = [
    'Particle_Size_nm', 'Particle_Width_nm', 'Drug_Loading_Efficiency', 
    'Targeting_Efficiency', 'Cytotoxicity'
]
BINARY_CLASS_TARGET = 'Successful_Treatment'

# --- Create output directory ---
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# --- Load and Preprocess Data ---
print(f"Loading data from: {DATA_FILE_PATH}")
df = pd.read_csv(DATA_FILE_PATH)

# Feature and Target Split
all_targets = REGRESSION_TARGETS + [BINARY_CLASS_TARGET]
X = df.drop(columns=['Precursor'] + all_targets)
y = df[all_targets]

# Preprocessing pipeline for input features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

feature_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y[BINARY_CLASS_TARGET]
)

# Apply preprocessing
X_train_processed = feature_preprocessor.fit_transform(X_train)
X_val_processed = feature_preprocessor.transform(X_val)

# Preprocess Targets
scaler_y_reg = MinMaxScaler()
y_train_reg_scaled = scaler_y_reg.fit_transform(y_train[REGRESSION_TARGETS])
y_val_reg_scaled = scaler_y_reg.transform(y_val[REGRESSION_TARGETS])

y_train_binary_class = y_train[BINARY_CLASS_TARGET].values
y_val_binary_class = y_val[BINARY_CLASS_TARGET].values

# --- FIX: Conditionally convert to dense array before creating tensor ---
# The output of ColumnTransformer can be sparse or dense. This handles both cases.
if hasattr(X_train_processed, 'toarray'):
    X_train_dense = X_train_processed.toarray()
    X_val_dense = X_val_processed.toarray()
else:
    X_train_dense = X_train_processed
    X_val_dense = X_val_processed

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_dense, dtype=torch.float32).to(device)
y_train_reg_tensor = torch.tensor(y_train_reg_scaled, dtype=torch.float32).to(device)
y_val_reg_tensor = torch.tensor(y_val_reg_scaled, dtype=torch.float32).to(device)
y_train_binary_tensor = torch.tensor(y_train_binary_class, dtype=torch.float32).unsqueeze(1).to(device)
y_val_binary_tensor = torch.tensor(y_val_binary_class, dtype=torch.float32).unsqueeze(1).to(device)


# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_reg_tensor, y_train_binary_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# --- Class Imbalance Handling ---
class_weights_binary = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_binary_class), y=y_train_binary_class)
pos_weight = torch.tensor(class_weights_binary[1] / class_weights_binary[0], dtype=torch.float32).to(device)
print(f"Binary classification positive weight: {pos_weight.item():.2f}")

# ==============================================================================
# --- Multi-Head MLP Model ---
# ==============================================================================
class MultiHeadNet(nn.Module):
    def __init__(self, input_dim, reg_output_dim, trial):
        super(MultiHeadNet, self).__init__()
        layers = []
        n_layers = trial.suggest_int('n_layers', 2, 5)
        in_dim = input_dim
        
        # Shared Layers
        for i in range(n_layers):
            out_dim = trial.suggest_int(f'n_units_l{i}', 64, 512, step=64)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            dropout = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.shared_layers = nn.Sequential(*layers)

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(in_dim, trial.suggest_int('reg_head_units', 32, 128)),
            nn.ReLU(),
            nn.Linear(trial.suggest_int('reg_head_units', 32, 128), reg_output_dim)
        )
        
        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(in_dim, trial.suggest_int('class_head_units', 32, 128)),
            nn.ReLU(),
            nn.Linear(trial.suggest_int('class_head_units', 32, 128), 1)
        )

    def forward(self, x):
        shared_output = self.shared_layers(x)
        reg_output = self.regression_head(shared_output)
        class_output = self.classification_head(shared_output)
        return reg_output, class_output

# ==============================================================================
# --- Optuna Objective Function ---
# ==============================================================================
def objective(trial):
    model = MultiHeadNet(X_train_processed.shape[1], len(REGRESSION_TARGETS), trial).to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Define loss functions
    criterion_reg = nn.MSELoss()
    criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    loss_weight_reg = trial.suggest_float('loss_weight_reg', 0.5, 2.0)

    model.train()
    for epoch in range(50): # Reduced epochs for faster Optuna trials
        for batch_x, batch_y_reg, batch_y_class in train_loader:
            optimizer.zero_grad()
            outputs_reg, outputs_class = model(batch_x)
            loss_reg = criterion_reg(outputs_reg, batch_y_reg)
            loss_class = criterion_class(outputs_class, batch_y_class)
            total_loss = (loss_reg * loss_weight_reg) + loss_class
            total_loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds_reg, val_preds_class = model(X_val_tensor)
        val_loss_reg = criterion_reg(val_preds_reg, y_val_reg_tensor)
        val_loss_class = criterion_class(val_preds_class, y_val_binary_tensor)
        total_val_loss = (val_loss_reg * loss_weight_reg) + val_loss_class
        
    return total_val_loss.item()

# --- Run Optuna Study ---
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20) # Increase trials for better results
print("Best trial score (combined loss):", study.best_trial.value)
print("Best hyperparameters:", study.best_trial.params)

# ==============================================================================
# --- Final Model Training ---
# ==============================================================================
print("\nTraining final model with best hyperparameters...")
final_model = MultiHeadNet(X_train_processed.shape[1], len(REGRESSION_TARGETS), study.best_trial).to(device)
optimizer = optim.Adam(final_model.parameters(), lr=study.best_trial.params['lr'])
criterion_reg = nn.MSELoss()
criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_weight_reg = study.best_trial.params['loss_weight_reg']

num_epochs = 200 # More epochs for final training
for epoch in range(num_epochs):
    final_model.train()
    for batch_x, batch_y_reg, batch_y_class in train_loader:
        optimizer.zero_grad()
        # --- FIX: Use final_model for training, not the temporary 'model' from Optuna ---
        outputs_reg, outputs_class = final_model(batch_x)
        loss_reg = criterion_reg(outputs_reg, batch_y_reg)
        loss_class = criterion_class(outputs_class, batch_y_class)
        total_loss = (loss_reg * loss_weight_reg) + loss_class
        total_loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")

# --- Save Final Model and Preprocessors ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(MODEL_DIR, f"mlp_gpu_model_{timestamp}.pth")
torch.save(final_model.state_dict(), model_path)
joblib.dump(feature_preprocessor, os.path.join(MODEL_DIR, f"feature_preprocessor_{timestamp}.joblib"))
joblib.dump(scaler_y_reg, os.path.join(MODEL_DIR, f"regression_scaler_{timestamp}.joblib"))
print(f"Final model and preprocessors saved with timestamp: {timestamp}")

# ==============================================================================
# --- Evaluation ---
# ==============================================================================
final_model.eval()
with torch.no_grad():
    preds_reg_scaled, preds_class_logits = final_model(X_val_tensor)

# Process Regression Results
preds_reg_unscaled = scaler_y_reg.inverse_transform(preds_reg_scaled.cpu().numpy())
# --- FIX: Compare against the validation set (y_val), not the training set (y_train) ---
y_val_reg_unscaled = y_val[REGRESSION_TARGETS].values 
r2_scores = [r2_score(y_val_reg_unscaled[:, i], preds_reg_unscaled[:, i]) for i in range(len(REGRESSION_TARGETS))]

print("\n--- Regression Performance ---")
for i, target in enumerate(REGRESSION_TARGETS):
    print(f"R² Score for {target}: {r2_scores[i]:.4f}")

# Process Classification Results
preds_class_probs = torch.sigmoid(preds_class_logits).cpu().numpy()
preds_binary_class = (preds_class_probs > 0.5).astype(int)

print("\n--- Classification Performance ---")
print(classification_report(y_val_binary_class, preds_binary_class, target_names=['Unsuccessful', 'Successful']))

# --- Plotting ---
# R² Score Plot
plt.figure(figsize=(10, 5))
plt.bar(REGRESSION_TARGETS, r2_scores, color='skyblue')
plt.title('R² Score per Regression Target')
plt.ylabel('R² Score')
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, f"r2_scores_mlp_{timestamp}.png"))
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val_binary_class, preds_binary_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Unsuccessful', 'Successful'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(MODEL_DIR, f"confusion_matrix_mlp_{timestamp}.png"))
plt.show()
