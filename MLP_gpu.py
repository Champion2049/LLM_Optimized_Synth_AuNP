import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import optuna
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from datetime import timedelta

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your DataFrame
df = pd.read_csv(r"C:\Users\Chirayu\Desktop\Coding\IMI\aunp_synthesis_cancer_treatment_v3_transformed.csv")

target_cols = ['Particle_Size_nm', 'Zeta_Potential_mV', 'Drug_Loading_Efficiency_%', 'Targeting_Efficiency_%', 'Cytotoxicity_%']
X = df.drop(columns=target_cols)
y = df[target_cols]

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# Dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a directory to save models if it doesn't exist
model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define model
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, trial):
        super(Net, self).__init__()
        layers = []
        n_layers = trial.suggest_int('n_layers', 1, 3)
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = trial.suggest_int(f'n_units_l{i}', 32, 256, step=32)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            dropout = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective(trial):
    model = Net(X_train.shape[1], len(target_cols), trial).to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_epochs = 100
    total_batches = num_epochs * len(train_loader)
    batch_counter = 0
    start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_counter += 1
            elapsed_time = time.time() - start_time
            progress = batch_counter / total_batches

            if batch_counter % 10 == 0:
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    time_left = estimated_total_time - elapsed_time
                    time_left_str = str(timedelta(seconds=int(time_left)))
                    print(f"Trial {trial.number}: {progress*100:.1f}% complete, Time left: {time_left_str}, Epoch {epoch+1}/{num_epochs}")

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        loss = criterion(val_preds, y_val_tensor)
    return loss.item()

print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1)
print("Best trial score:", study.best_trial.value)
print("Best hyperparameters:", study.best_trial.params)

best_trial = study.best_trial

class FinalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FinalNet, self).__init__()
        layers = []
        in_dim = input_dim
        for i in range(best_trial.params['n_layers']):
            out_dim = best_trial.params[f'n_units_l{i}']
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(best_trial.params[f'dropout_l{i}']))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

final_model = FinalNet(X_train.shape[1], len(target_cols)).to(device)
optimizer = optim.Adam(final_model.parameters(), lr=best_trial.params['lr'])
criterion = nn.MSELoss()

print("\nTraining final model...")
num_epochs = 100
total_batches = num_epochs * len(train_loader)
batch_counter = 0
start_time = time.time()

final_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_counter += 1
        elapsed_time = time.time() - start_time
        progress = batch_counter / total_batches

        if batch_counter % 10 == 0:
            if progress > 0:
                estimated_total_time = elapsed_time / progress
                time_left = estimated_total_time - elapsed_time
                time_left_str = str(timedelta(seconds=int(time_left)))
                print(f"Final model: {progress*100:.1f}% complete, Time left: {time_left_str}, Epoch {epoch+1}/{num_epochs}")

        optimizer.zero_grad()
        outputs = final_model(batch_x)
        loss = criterion(outputs, batch_y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.6f}")

# Move model to CPU before saving
final_model.cpu()

# Save the trained model
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(model_dir, f"aunp_model_{timestamp}.pth")
torch.save({
    'model_state_dict': final_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': best_trial.params,
    'input_dim': X_train.shape[1],
    'output_dim': len(target_cols),
    'target_columns': target_cols,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y
}, model_path)
print(f"Model saved to {model_path}")

# Evaluate
final_model.eval()
with torch.no_grad():
    preds = final_model(X_val_tensor.cpu()).numpy()
    true = y_val_tensor.cpu().numpy()

preds_inv = scaler_y.inverse_transform(preds)
true_inv = scaler_y.inverse_transform(true)

r2_scores = [r2_score(true_inv[:, i], preds_inv[:, i]) for i in range(len(target_cols))]

# Plot R² scores
plt.figure(figsize=(8, 5))
bars = plt.bar(target_cols, r2_scores, color='skyblue')
plt.ylim(0, 1)
plt.title('R² Score per Target')
plt.ylabel('R² Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{score:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(model_dir, f"r2_scores_{timestamp}.png"))
plt.show()

# Function to load the model
def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = FinalNet(checkpoint['input_dim'], checkpoint['output_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

print("\nExample of how to load the saved model:")
print("model, checkpoint = load_model('path_to_your_model.pth')")
print("# Access scalers with: checkpoint['scaler_X'], checkpoint['scaler_y']")

# Total training time
total_time = time.time() - start_time
print(f"\nTotal training time: {str(timedelta(seconds=int(total_time)))}")