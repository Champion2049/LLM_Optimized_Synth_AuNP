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
import matplotlib.pyplot as plt # Added for plotting

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and clean data
def load_and_clean_data(file_path):
    """
    Loads data from a CSV file, fixes potential merged numbers (e.g., '1.2.3' -> 1.23),
    converts columns to numeric, and drops rows with NaN values.
    """
    df = pd.read_csv(file_path, header=None, on_bad_lines='skip')

    def fix_merged_numbers(cell):
        """Helper function to fix strings like '1.2.3'."""
        if isinstance(cell, str):
            parts = cell.split('.')
            if len(parts) > 2:
                try:
                    # Try to form a valid float like '1.23' from '1.2.3'
                    return float(parts[0] + '.' + ''.join(parts[1:]))
                except ValueError:
                    # If it still fails, return NaN to be handled later
                    return np.nan
            elif len(parts) == 2 and parts[1] == '': # Handle cases like '1.'
                 try:
                     return float(parts[0])
                 except ValueError:
                     return np.nan
        # If it's already a number or a convertible string, return it
        # Let pd.to_numeric handle standard conversion and errors
        return cell

    # Apply the fix_merged_numbers function using applymap
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
    y_scaled = scaler_y.fit_transform(y) # Scale all target columns

    # Split data into training and validation sets
    return train_test_split(X_scaled, y_scaled, test_size=0.11, random_state=42), scaler_X, scaler_y

# --- Simple MLP Architecture ---
class SimpleMLP(nn.Module):
    """
    A simple Feedforward Neural Network (MLP) with tunable layers and hidden size.
    Includes BatchNorm and Dropout for regularization. Outputs 'output_dim' values.
    """
    def __init__(self, input_dim, output_dim, trial): # output_dim will be 5
        super(SimpleMLP, self).__init__()
        # Hyperparameters to tune using Optuna
        self.num_layers = trial.suggest_int('mlp_num_layers', 2, 8)
        self.hidden_size = trial.suggest_int('mlp_hidden_size', 64, 512, step=64)

        layers = []
        in_dim = input_dim
        for i in range(self.num_layers):
            out_dim_layer = self.hidden_size
            # Add Linear layer
            layers.append(nn.Linear(in_dim, out_dim_layer))
            # Add BatchNorm layer for stability
            layers.append(nn.BatchNorm1d(out_dim_layer))
            # Add ReLU activation
            layers.append(nn.ReLU())
            # Add Dropout for regularization
            layers.append(nn.Dropout(trial.suggest_float(f'mlp_dropout_l{i}', 0.1, 0.5)))
            in_dim = out_dim_layer # Update input dimension for the next layer

        # Create the main MLP sequential model
        self.mlp = nn.Sequential(*layers)

        # *** CHANGE: Single output layer for all targets ***
        self.output_layer = nn.Linear(in_dim, output_dim)


    def forward(self, x):
        """
        Forward pass through the MLP and the final output layer.
        """
        # Pass input through the MLP layers
        mlp_out = self.mlp(x)
        # Get predictions from the single output layer
        outputs = self.output_layer(mlp_out)
        return outputs

# --- Original DCN Architecture (Commented out - not used) ---
# ... (DCN code remains commented out) ...


# Objective for Optuna
def objective(trial):
    """
    Optuna objective function to train and evaluate a model (MLP in this case).
    Minimizes validation MSE loss. Now predicts 5 targets.
    """
    n_targets = 5 # Define number of targets
    # Instantiate the SimpleMLP model for 5 outputs
    model = SimpleMLP(X_train.shape[1], n_targets, trial).to(device) # Pass n_targets (5)

    # Hyperparameters to tune using Optuna
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True) # Adjusted range slightly
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # *** CHANGE: Use all target columns (y_train) ***
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15 # Patience for early stopping during Optuna trials

    # Time tracking for Optuna trials
    start_time = time.time()
    max_epochs_optuna = 100
    total_batches_estimated = max_epochs_optuna * len(train_loader) # Estimate total batches
    batch_counter = 0


    for epoch in range(max_epochs_optuna): # Max epochs for Optuna trial
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
            progress = batch_counter / total_batches_estimated if total_batches_estimated > 0 else 0

            # Print progress periodically (e.g., every 50 batches or based on time)
            if batch_counter % 50 == 0: # Print less frequently
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    time_left = estimated_total_time - elapsed_time
                    time_left_str = str(timedelta(seconds=int(time_left))) if time_left > 0 else "0s"
                    print(f"\r[Trial {trial.number}] Epoch {epoch+1}/{max_epochs_optuna} | Batch {batch_counter}/{total_batches_estimated} | "
                          f"{progress*100:.1f}% | ETA: {time_left_str} | Loss: {loss.item():.4f}", end="")


        # Validation after each epoch
        model.eval()
        with torch.no_grad():
            val_inputs_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            # *** CHANGE: Use all target columns (y_val) ***
            val_targets_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            val_outputs_tensor = model(val_inputs_tensor)
            val_loss = criterion(val_outputs_tensor, val_targets_tensor).item()

        # Clear the progress line before printing validation info
        print(f"\r{' ' * 100}\r", end="") # Clear line
        print(f"[Trial {trial.number}] Epoch {epoch+1}/{max_epochs_optuna} completed. Avg Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")


        # Report intermediate value to Optuna
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            print(f"Trial {trial.number}: Pruned at epoch {epoch+1}")
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
    file_path = r"C:\Users\Chirayu\Desktop\Coding\IMI\aunp_synthesis_cancer_treatment_v3_transformed.csv" # Use raw string for path
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit()

    df = load_and_clean_data(file_path)

    # Ensure DataFrame is not empty after cleaning
    if df.empty:
        print("Error: DataFrame is empty after loading and cleaning. Check data and cleaning steps.")
        exit()

    num_cols = df.shape[1]
    n_targets = 5 # Number of target variables

    # *** CHANGE: Define target column indices (assuming last 5 columns are targets) ***
    if num_cols < n_targets:
         print(f"Error: DataFrame has only {num_cols} columns, but {n_targets} target columns were expected.")
         exit()
    target_col_indices = list(range(num_cols - n_targets, num_cols))

    # *** CHANGE: Update target names to match the 5 targets ***
    # Ensure these names correspond EXACTLY to your last 5 columns in the CSV
    target_names = ['Cytotoxicity_%', 'Targeting_Efficiency_%', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%', 'Particle_Size_nm']
    if len(target_names) != n_targets:
        print(f"Error: Mismatch between number of targets ({n_targets}) and number of target names provided ({len(target_names)}).")
        exit()


    # Check if target_col_indices are valid (redundant check, but good practice)
    if any(idx < 0 or idx >= num_cols for idx in target_col_indices):
        print(f"Error: Target column indices {target_col_indices} are out of bounds for DataFrame with {num_cols} columns.")
        exit()

    # Preprocess data for 5 targets
    (X_train, X_val, y_train, y_val), scaler_X, scaler_y = preprocess_data(df, target_col_indices)

    # --- Hyperparameter Optimization using Optuna ---
    study = optuna.create_study(direction="minimize")
    print("Starting hyperparameter optimization...")
    # *** INCREASE N_TRIALS for better results (e.g., 50-100) ***
    n_optuna_trials = 10 # Example: run 50 trials
    study.optimize(objective, n_trials=n_optuna_trials)

    best_params = study.best_trial.params
    print("\nBest Hyperparameters found by Optuna:", best_params)
    print(f"Best Validation Loss (MSE) during Optuna: {study.best_trial.value:.6f}")


    # --- Final Model Training with Best Hyperparameters ---
    # Define a class that uses the best parameters found by Optuna
    class FinalMLP(SimpleMLP):
        def __init__(self, input_dim, output_dim):
            # Pass the best_params to the parent class constructor via a FixedTrial
            # Optuna requires the trial object, so we create a 'FixedTrial' from the best params
            fixed_trial = optuna.trial.FixedTrial(best_params)
            super(FinalMLP, self).__init__(input_dim, output_dim, fixed_trial)

    # Instantiate the final model using the best hyperparameters for 5 outputs
    final_model = FinalMLP(X_train.shape[1], n_targets).to(device) # Pass n_targets (5)

    # Use the best optimizer and learning rate from Optuna
    optimizer = getattr(optim, best_params['optimizer'])(final_model.parameters(),
                                                          lr=best_params['lr'],
                                                          weight_decay=best_params['weight_decay'])
    criterion = nn.MSELoss()

    # Use the best batch size from Optuna
    # *** CHANGE: Use all target columns (y_train) ***
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0
    # ReduceLROnPlateau scheduler to decrease LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5) # Increased patience, added verbose

    # --- Final Training Loop with Time Tracking ---
    print("\nStarting final model training with best hyperparameters...")
    num_epochs = 500 # Increased max epochs for final training (adjust as needed)
    final_early_stop_patience = 40 # Increased patience for final training

    total_batches = num_epochs * len(train_loader)
    batch_counter = 0
    start_time = time.time()
    saved_model_path = "final_mlp_model_best.pth" # Path to save the best model state

    for epoch in range(num_epochs):
        final_model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time() # Time epoch

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Training step
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Progress tracking within epoch
            batch_counter += 1
            # Print progress less frequently during final training
            if (batch_idx + 1) % (len(train_loader) // 4 + 1) == 0: # Print ~4 times per epoch
                 elapsed_time = time.time() - start_time
                 progress = batch_counter / total_batches if total_batches > 0 else 0
                 estimated_total_time = elapsed_time / progress if progress > 0 else 0
                 time_left = estimated_total_time - elapsed_time
                 time_left_str = str(timedelta(seconds=int(time_left))) if time_left > 0 else "0s"
                 print(f"\r[Final Training] Epoch {epoch+1}/{num_epochs} [{batch_idx+1}/{len(train_loader)}] | "
                       f"{progress*100:.1f}% | ETA: {time_left_str} | Batch Loss: {loss.item():.4f}", end="")


        # Validation after each epoch
        final_model.eval()
        with torch.no_grad():
            val_inputs_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            # *** CHANGE: Use all target columns (y_val) ***
            val_targets_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            val_outputs_tensor = final_model(val_inputs_tensor)
            val_loss = criterion(val_outputs_tensor, val_targets_tensor).item()

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Clear progress line and print epoch summary
        print(f"\r{' ' * 120}\r", end="") # Clear line
        print(f"[Final Training] Epoch {epoch+1}/{num_epochs} | Avg Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s")


        # Step the learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping for final training
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.6f} --> {val_loss:.6f}). Saving model...")
            best_val_loss = val_loss
            # Save the model state dict when validation loss improves
            torch.save(final_model.state_dict(), saved_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{final_early_stop_patience}")


        if patience_counter >= final_early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs in final training.")
            break # Stop final training

    # Save final model state, scalers, and hyperparameters bundle
    print("\nTraining complete. Saving final bundle...")
    # Load the best model state saved during training
    if os.path.exists(saved_model_path):
         print(f"Loading best model state from {saved_model_path}")
         final_model.load_state_dict(torch.load(saved_model_path))
    else:
         print("Warning: Best model state file not found. Using the model state from the last epoch.")

    torch.save({
        'model_state_dict': final_model.state_dict(), # Save the best loaded state
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'target_columns': target_names, # Save the target names list
        'input_dim': X_train.shape[1], # Save input dimension
        'output_dim': n_targets,       # Save output dimension
        'hyperparameters': best_params # Save hyperparameters
    }, "final_mlp_model_with_scalers_5targets.pth") # Changed filename

    # Print total training time
    total_time = time.time() - start_time
    print(f"\nTotal final training time: {str(timedelta(seconds=int(total_time)))}")

    # --- Evaluate final model on validation set ---
    print("\nEvaluating final model (best state) on validation set...")
    final_model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        val_inputs_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        # Get predictions from the final model
        val_outputs_tensor = final_model(val_inputs_tensor)
        val_outputs_scaled = val_outputs_tensor.cpu().numpy() # Predictions are still scaled

    # Rescale predictions and targets back to original range
    # *** CHANGE: Use the full y_val for inverse transform ***
    val_outputs_rescaled = scaler_y.inverse_transform(val_outputs_scaled)
    val_targets_rescaled = scaler_y.inverse_transform(y_val) # Use y_val directly

    # --- Compute and Print R² scores for each target variable ---
    r2_scores = []
    print(f"\nValidation R² Scores (using Simple MLP):")
    for i in range(n_targets):
        score = r2_score(val_targets_rescaled[:, i], val_outputs_rescaled[:, i])
        r2_scores.append(score)
        # Use target_names for printing
        print(f"  {target_names[i]:<25}: {score:.4f}")

    # --- Plot R² Scores ---
    print("\nGenerating R² score comparison plot...")
    plt.figure(figsize=(10, 6)) # Adjust figure size for better label visibility
    bars = plt.bar(target_names, r2_scores, color='skyblue')
    plt.xlabel("Target Variable")
    plt.ylabel("R² Score")
    plt.title("R² Scores for Predicted Outputs on Validation Set")
    plt.xticks(rotation=30, ha='right') # Rotate labels for readability
    plt.ylim(bottom=min(0, min(r2_scores) - 0.05), top=max(1, max(r2_scores) + 0.05)) # Adjust y-axis limits
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

    # Add R² values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center') # Adjust position based on value

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig("r2_score_comparison_plot.png") # Save the plot as an image
    print("Plot saved as r2_score_comparison_plot.png")
    plt.show() # Display the plot

    print("\nScript finished.")