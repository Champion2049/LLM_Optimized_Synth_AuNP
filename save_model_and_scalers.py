import os
import joblib
import json
from keras.models import model_from_json
from datetime import datetime

# === Paths to your existing files ===
json_path = "best_dl_model_20250502_201740.keras/config.json"
weights_path = "best_dl_model_20250502_201740.keras/model.weights.h5"
scaler_X_path = "scaler_X.pkl"  # <-- Replace with actual path to StandardScaler for input
scaler_y_path = "scaler_y.pkl"  # <-- Replace with actual path to StandardScaler for output

# === Load model ===
with open(json_path, "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(weights_path)

# === Load scalers ===
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# === Save all in expected format ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, f"best_dl_model_{timestamp}.keras"))
joblib.dump(scaler_X, os.path.join(save_dir, f"scaler_X_{timestamp}.pkl"))
joblib.dump(scaler_y, os.path.join(save_dir, f"scaler_y_{timestamp}.pkl"))

print("✅ Model and scalers saved successfully.")
