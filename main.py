import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from wsgi2asgi import WSGI2ASGI # <-- Add this import

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS untuk mengizinkan permintaan dari frontend Next.js

# Wrap the Flask app with WSGI2ASGI to make it ASGI compatible
asgi_app = WSGI2ASGI(app) # <-- NEW LINE: Create an ASGI-compatible app

# Path ke folder model dan data historis
MODEL_DIR = 'model'
DATA_HARGA_PATH = 'data_harga.json'

# Cache untuk data historis
HISTORICAL_DATA_CACHE = None

def load_historical_data():
    """Memuat data historis dari data_harga.json."""
    global HISTORICAL_DATA_CACHE
    if HISTORICAL_DATA_CACHE is None:
        try:
            with open(DATA_HARGA_PATH, 'r') as f:
                HISTORICAL_DATA_CACHE = json.load(f)
            HISTORICAL_DATA_CACHE.sort(key=lambda x: x['date'])
        except FileNotFoundError:
            print(f"Error: {DATA_HARGA_PATH} not found.")
            HISTORICAL_DATA_CACHE = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {DATA_HARGA_PATH}.")
            HISTORICAL_DATA_CACHE = []
    return HISTORICAL_DATA_CACHE

def get_last_n_values(data, commodity_type, n):
    """Mendapatkan n nilai terakhir dari data historis untuk komoditas tertentu."""
    prices = [entry[commodity_type] for entry in data if commodity_type in entry]
    return prices[-n:]

def get_most_recent_data(data, commodity_type):
    """Mendapatkan data terbaru dari data historis untuk komoditas tertentu."""
    if not data:
        return None
    latest_entry = data[-1]
    return latest_entry.get(commodity_type)

@app.before_request
def before_first_request():
    """Muat data historis saat aplikasi pertama kali dimulai."""
    load_historical_data()

@app.route('/')
def home():
    return "API Prediksi Harga Beras Berjalan!"

@app.route('/predict/<model_type>/<commodity_type>', methods=['GET'])
def predict(model_type, commodity_type):
    """
    Endpoint untuk melakukan prediksi harga beras.
    Args:
        model_type (str): Tipe model (e.g., 'arima', 'lstm')
        commodity_type (str): Jenis beras (e.g., 'medium_silinda', 'premium_bapanas')
    Query Params:
        steps_ahead (int): Berapa hari ke depan untuk diprediksi. Default 1.
    """
    
    steps_ahead_str = request.args.get('steps_ahead', '1')
    try:
        steps_ahead = int(steps_ahead_str)
        if steps_ahead <= 0:
            return jsonify({'error': 'steps_ahead must be a positive integer'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid steps_ahead value. Must be an integer.'}), 400

    valid_commodity_types = ['medium_silinda', 'premium_silinda', 'medium_bapanas', 'premium_bapanas']
    if commodity_type not in valid_commodity_types:
        return jsonify({'error': 'Invalid commodity type'}), 400

    historical_data = HISTORICAL_DATA_CACHE
    if not historical_data:
        return jsonify({'error': 'Historical data not loaded or empty'}), 500

    try:
        model_name = f"{commodity_type}_{model_type}"
        model_path_prefix = os.path.join(MODEL_DIR, model_name)

        if model_type == 'arima':
            model_file = f"{model_path_prefix}.joblib"
            if not os.path.exists(model_file):
                return jsonify({'error': f'ARIMA model file not found for {commodity_type}'}), 404
            
            model = joblib.load(model_file)
            series_data = [entry[commodity_type] for entry in historical_data if commodity_type in entry]
            
            # Assuming the loaded ARIMA model has a predict or forecast method
            predictions = model.predict(n_periods=steps_ahead)
            predicted_prices = predictions.tolist()


        elif model_type == 'lstm':
            model_file = f"{model_path_prefix}.h5"
            if not os.path.exists(model_file):
                return jsonify({'error': f'LSTM model file not found for {commodity_type}'}), 404
            
            model = load_model(model_file)
            
            look_back = 10 # Ganti dengan nilai look_back yang sebenarnya dari model Anda
            
            last_n_values = get_last_n_values(historical_data, commodity_type, look_back)
            if len(last_n_values) < look_back:
                return jsonify({'error': f'Not enough historical data for {commodity_type} to make LSTM prediction (need at least {look_back} days)'}), 400

            input_sequence = np.array(last_n_values).reshape(1, look_back, 1)

            predicted_prices = []
            current_input = input_sequence
            
            for _ in range(steps_ahead):
                prediction = model.predict(current_input)[0][0]
                predicted_prices.append(prediction)
                current_input = np.append(current_input[:, 1:, :], [[[new_value]]], axis=1)

            predicted_prices = [round(p) for p in predicted_prices]

        else:
            return jsonify({'error': 'Invalid model type. Choose "arima" or "lstm"'}), 400
        
        return jsonify({'predictions': predicted_prices}), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # For local development with Flask's dev server, you still use app.run()
    app.run(debug=True, port=5000)