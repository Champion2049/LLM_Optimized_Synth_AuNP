import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import optuna
import os
import time
from datetime import timedelta
from sklearn.metrics import r2_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and clean data
def load_and_clean_data(file_path):
    """
    Loads data from a CSV file, fixes potential merged numbers (e.g., '1.2.3' -> 1.2),
    converts columns to numeric, and drops rows with NaN values.
    """
    df = pd.read_csv(file_path, header=None, on_bad_lines='skip')

    def fix_merged_numbers(cell):
        """Helper function to fix strings like '1.2.3'."""
        if isinstance(cell, str):
            parts = cell.split('.')
            if len(parts) > 2:
                # Take the first part and the rest after the first dot
                # This assumes '1.2.3' should be 1.23, '1.2.3.4' should be 1.234 etc.
                return float(parts[0] + '.' + ''.join(parts[1:]))
        return cell

    # Apply the fix_merged_numbers function to all cells
    # Use applymap for element-wise application across the DataFrame
    df = df.applymap(fix_merged_numbers)
    # Convert all columns to numeric, coercing errors (non-convertible values become NaN)
    df = df.apply(pd.to_numeric, errors='coerce')
    # Drop any rows that contain NaN values after conversion
    df = df.dropna()
    return df

# Preprocess data
def preprocess_data(df, target_col_indices):
    """
    Separates features (X) and targets (y), scales them using MinMaxScaler,
    and splits the data into training and validation sets.
    """
    # Separate features (X) and targets (y)
    # Use df.columns[target_col_indices] to select columns by index
    X = df.drop(columns=df.columns[target_col_indices]).values
    y = df[df.columns[target_col_indices]].values

    # Initialize scalers for features and targets
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit and transform the data
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split data into training and validation sets
    return train_test_split(X_scaled, y_scaled, test_size=0.11, random_state=42), scaler_X, scaler_y

# --- Simple MLP Architecture ---
class SimpleMLP(nn.Module):
    """
    A simple Feedforward Neural Network (MLP) with tunable layers and hidden size.
    Includes BatchNorm and Dropout for regularization.
    """
    def __init__(self, input_dim, output_dim, trial):
        super(SimpleMLP, self).__init__()
        # Hyperparameters to tune using Optuna
        self.num_layers = trial.suggest_int('mlp_num_layers', 2, 8)
        self.hidden_size = trial.suggest_int('mlp_hidden_size', 64, 512, step=64)

        layers = []
        in_dim = input_dim
        for i in range(self.num_layers):
            out_dim = self.hidden_size
            # Add Linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            # Add BatchNorm layer for stability
            layers.append(nn.BatchNorm1d(out_dim))
            # Add ReLU activation
            layers.append(nn.ReLU())
            # Add Dropout for regularization (except after the last hidden layer)
            if i < self.num_layers - 1:
                 layers.append(nn.Dropout(trial.suggest_float(f'mlp_dropout_l{i}', 0.1, 0.5)))
            in_dim = out_dim # Update input dimension for the next layer

        # Create the main MLP sequential model
        self.mlp = nn.Sequential(*layers)

        # Output heads for the two target variables
        self.head_cytotoxicity = nn.Linear(in_dim, 1)
        self.head_target_efficiency = nn.Linear(in_dim, 1)


    def forward(self, x):
        """
        Forward pass through the MLP and output heads.
        """
        # Pass input through the MLP layers
        mlp_out = self.mlp(x)
        # Get predictions for each target variable
        cytotoxicity = self.head_cytotoxicity(mlp_out)
        target_efficiency = self.head_target_efficiency(mlp_out)
        # Concatenate the predictions
        return torch.cat([cytotoxicity, target_efficiency], dim=-1)

# --- Original DCN Architecture (kept for comparison, not used in this example) ---
# class CrossLayer(nn.Module):
#     def __init__(self, input_dim):
#         super(CrossLayer, self).__init__()
#         self.W = nn.Linear(input_dim, input_dim)
#         self.b = nn.Parameter(torch.zeros(input_dim))

#     def forward(self, x):
#         return x * self.W(x) + self.b + x  # Cross operation

# class DCN(nn.Module):
#     def __init__(self, input_dim, output_dim, trial):
#         super(DCN, self).__init__()
#         self.num_cross_layers = trial.suggest_int('num_cross_layers', 1, 5)
#         self.num_deep_layers = trial.suggest_int('num_deep_layers', 2, 6)
#         self.hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)

#         # Cross Network
#         self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(self.num_cross_layers)])

#         # Deep Network
#         layers = []
#         in_dim = input_dim
#         for _ in range(self.num_deep_layers):
#             layers.append(nn.Linear(in_dim, self.hidden_size))
#             layers.append(nn.BatchNorm1d(self.hidden_size))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(trial.suggest_float('deep_dropout', 0.1, 0.5)))
#             in_dim = self.hidden_size
#         self.deep = nn.Sequential(*layers)

#         # Output heads
#         self.head_cytotoxicity = nn.Linear(input_dim + self.hidden_size, 1)
#         self.head_target_efficiency = nn.Linear(input_dim + self.hidden_size, 1)

#     def forward(self, x):
#         # Cross Network
#         cross_out = x
#         for layer in self.cross_layers:
#             cross_out = layer(cross_out)

#         # Deep Network
#         deep_out = self.deep(x)

#         # Concatenate cross + deep outputs
#         combined = torch.cat([cross_out, deep_out], dim=-1)

#         # Separate heads
#         cytotoxicity = self.head_cytotoxicity(combined)
#         target_efficiency = self.head_target_efficiency(combined)

#         return torch.cat([cytotoxicity, target_efficiency], dim=-1)


# Objective for Optuna
def objective(trial):
    """
    Optuna objective function to train and evaluate a model (MLP in this case).
    Minimizes validation MSE loss.
    """
    # Instantiate the SimpleMLP model instead of DCN
    model = SimpleMLP(X_train.shape[1], 2, trial).to(device)

    # Hyperparameters to tune using Optuna
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True) # Added log=True for wider search
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128]) # Added 128

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train[:, :2], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15 # Patience for early stopping during Optuna trials

    # Time tracking for Optuna trials
    start_time = time.time()
    # Estimate total batches for progress tracking (assuming max 100 epochs)
    total_batches = 100 * len(train_loader)
    batch_counter = 0

    for epoch in range(100): # Max epochs for Optuna trial
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Progress tracking inside the batch loop
            batch_counter += 1
            elapsed_time = time.time() - start_time
            progress = batch_counter / total_batches

            # Print progress every 10 batches
            if batch_counter % 10 == 0:
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    time_left = estimated_total_time - elapsed_time
                    time_left_str = str(timedelta(seconds=int(time_left)))
                    print(f"[Trial {trial.number}] {progress*100:.1f}% complete | "
                          f"Time left: {time_left_str} | "
                          f"Epoch {epoch+1}/100")


        # Validation after each epoch
        model.eval()
        with torch.no_grad():
            val_inputs_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_targets_tensor = torch.tensor(y_val[:, :2], dtype=torch.float32).to(device)
            # *** FIX: Calculate validation loss using model outputs and targets ***
            val_outputs_tensor = model(val_inputs_tensor)
            val_loss = criterion(val_outputs_tensor, val_targets_tensor).item()

        # Report intermediate value to Optuna
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Trial {trial.number}: Early stopping triggered at epoch {epoch+1}")
            break # Stop training for this trial

    return best_val_loss # Return the best validation loss found during the trial


# Main Pipeline
if __name__ == "__main__":
    file_path = r"C:\Users\Chirayu\Desktop\Coding\IMI\aunp_synthesis_cancer_treatment_v3_transformed.csv"
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        # Exit or handle the error appropriately
        exit()

    df = load_and_clean_data(file_path)

    # Ensure DataFrame is not empty after cleaning
    if df.empty:
        print("Error: DataFrame is empty after loading and cleaning. Check data and cleaning steps.")
        exit()

    num_cols = df.shape[1]
    # Define target column indices (assuming last 2 columns are targets)
    target_col_indices = list(range(num_cols - 2, num_cols))
    target_names = ['Cytotoxicity_%', 'Targeting_Efficiency_%']

    # Check if target_col_indices are valid
    if any(idx < 0 or idx >= num_cols for idx in target_col_indices):
         print(f"Error: Target column indices {target_col_indices} are out of bounds for DataFrame with {num_cols} columns.")
         exit()


    (X_train, X_val, y_train, y_val), scaler_X, scaler_y = preprocess_data(df, target_col_indices)

    # Hyperparameter Optimization using Optuna
    # Increase n_trials to allow Optuna to find better hyperparameters
    study = optuna.create_study(direction="minimize")
    print("Starting hyperparameter optimization...")
    # *** IMPORTANT: INCREASE N_TRIALS HERE ***
    study.optimize(objective, n_trials=1) # Increased from 1 to 50 trials

    best_params = study.best_trial.params
    print("Best Hyperparameters found by Optuna:", best_params)

    # --- Final Model Training with Best Hyperparameters ---
    # Define a class that uses the best parameters found by Optuna
    class FinalMLP(SimpleMLP):
        def __init__(self, input_dim, output_dim):
            # Pass the best_params to the parent class constructor via a FixedTrial
            super(FinalMLP, self).__init__(input_dim, output_dim, optuna.trial.FixedTrial(best_params))

    # Instantiate the final model using the best hyperparameters
    final_model = FinalMLP(X_train.shape[1], 2).to(device)

    # Use the best optimizer and learning rate from Optuna
    optimizer = getattr(optim, best_params['optimizer'])(final_model.parameters(),
                                                        lr=best_params['lr'],
                                                        weight_decay=best_params['weight_decay'])
    criterion = nn.MSELoss()

    # Use the best batch size from Optuna
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train[:, :2], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0
    # ReduceLROnPlateau scheduler to decrease LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5) # Increased patience

    # Final Training Loop with Time Tracking
    print("\nStarting final model training with best hyperparameters...")
    num_epochs = 300 # Increased max epochs for final training
    total_batches = num_epochs * len(train_loader)
    batch_counter = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        final_model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_counter += 1
            elapsed_time = time.time() - start_time
            progress = batch_counter / total_batches

            # Progress tracking
            if batch_counter % 10 == 0: # Print progress every 10 batches
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                time_left = estimated_total_time - elapsed_time
                time_left_str = str(timedelta(seconds=int(time_left)))
                print(f"[Final Training] {progress*100:.1f}% complete | "
                      f"Time left: {time_left_str} | "
                      f"Epoch {epoch+1}/{num_epochs}")

            # Training step
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation after each epoch
        final_model.eval()
        with torch.no_grad():
            val_inputs_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_targets_tensor = torch.tensor(y_val[:, :2], dtype=torch.float32).to(device)
            # Get predictions from the final model
            val_outputs_tensor = final_model(val_inputs_tensor)
            val_loss = criterion(val_outputs_tensor, val_targets_tensor).item()

        # Step the learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping for final training
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model state dict when validation loss improves
            torch.save(final_model.state_dict(), "final_mlp_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 30: # Increased patience for final training
            print("Early stopping triggered in final training")
            break # Stop final training

    # Save final model state, scalers, and hyperparameters
    print("\nTraining complete. Saving model and scalers...")
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'target_columns': target_names,
        'hyperparameters': best_params
    }, "final_mlp_model_with_scalers.pth") # Changed filename

    # Print total training time
    total_time = time.time() - start_time
    print(f"\nTotal final training time: {str(timedelta(seconds=int(total_time)))}")

    # Evaluate final model on validation set
    print("\nEvaluating final model on validation set...")
    final_model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        val_inputs_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_targets_tensor = torch.tensor(y_val[:, :2], dtype=torch.float32).to(device)
        # Get predictions from the final model
        val_outputs_tensor = final_model(val_inputs_tensor)
        val_outputs = val_outputs_tensor.cpu().numpy()


    # Rescale predictions and targets back to original range
    val_outputs_rescaled = scaler_y.inverse_transform(val_outputs)
    val_targets_rescaled = scaler_y.inverse_transform(y_val[:, :2]) # Use y_val directly for original targets

    # Compute R² scores for each target variable
    r2_cytotoxicity = r2_score(val_targets_rescaled[:, 0], val_outputs_rescaled[:, 0])
    r2_targeting_efficiency = r2_score(val_targets_rescaled[:, 1], val_outputs_rescaled[:, 1])

    print(f"\nValidation R² Scores (using Simple MLP):")
    print(f"  Cytotoxicity (%):         {r2_cytotoxicity:.4f}")
    print(f"  Targeting Efficiency (%): {r2_targeting_efficiency:.4f}")

    # Optional: Add code here to load the best saved model state_dict
    # final_model.load_state_dict(torch.load("final_mlp_model.pth"))
    # Then re-evaluate if needed, although the above evaluation is based on the best model during training.