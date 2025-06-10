from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Folder model
MODEL_DIR = "models"

# Load ARIMA models (joblib)
arima_models = {}
for filename in os.listdir(MODEL_DIR):
    if filename.endswith(".joblib"):
        model_key = filename.replace(".joblib", "")
        model_path = os.path.join(MODEL_DIR, filename)
        arima_models[model_key] = joblib.load(model_path)

# Load LSTM models (h5)
lstm_models = {}
for filename in os.listdir(MODEL_DIR):
    if filename.endswith(".h5"):
        model_key = filename.replace(".h5", "")
        model_path = os.path.join(MODEL_DIR, filename)
        lstm_models[model_key] = load_model(model_path)

@app.route("/")
def index():
    return "API is running"

@app.route("/predict/arima/<model_name>", methods=["POST"])
def predict_arima(model_name):
    model = arima_models.get(model_name)
    if not model:
        return jsonify({"error": "ARIMA model not found"}), 404

    try:
        n_periods = int(request.json.get("n_periods", 1))
        pred = model.predict(n_periods=n_periods)
        return jsonify({"prediction": pred.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/lstm/<model_name>", methods=["POST"])
def predict_lstm(model_name):
    model = lstm_models.get(model_name)
    if not model:
        return jsonify({"error": "LSTM model not found"}), 404

    try:
        input_data = request.json.get("input")
        input_array = np.array(input_data).reshape(1, -1, 1)
        pred = model.predict(input_array)
        return jsonify({"prediction": pred.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
