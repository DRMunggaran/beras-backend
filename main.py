import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd # Digunakan dalam asumsi model ARIMA Anda mungkin butuh Pandas Series
from datetime import datetime, timedelta
from uvicorn.wsgi import WSGIMiddleware # Import WSGIMiddleware dari uvicorn

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS untuk mengizinkan permintaan dari frontend Next.js

# Wrap the Flask app with WSGIMiddleware to make it ASGI compatible
asgi_app = WSGIMiddleware(app)

# Path ke folder model dan data historis
MODEL_DIR = 'model' # Pastikan folder 'model' berada di root backend
DATA_HARGA_PATH = 'data_harga.json' # Pastikan 'data_harga.json' berada di root backend

# Cache untuk data historis
HISTORICAL_DATA_CACHE = None

def load_historical_data():
    """Memuat data historis dari data_harga.json."""
    global HISTORICAL_DATA_CACHE
    if HISTORICAL_DATA_CACHE is None:
        try:
            with open(DATA_HARGA_PATH, 'r') as f:
                HISTORICAL_DATA_CACHE = json.load(f)
            # Urutkan berdasarkan tanggal untuk memastikan urutan yang benar
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
    prices = [entry.get(commodity_type) for entry in data if commodity_type in entry]
    # Filter out None values just in case
    prices = [p for p in prices if p is not None]
    return prices[-n:]

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
        model_name_prefix = commodity_type.replace('_', '') # e.g., mediumsilinda
        
        # Load the correct model file (e.g., medium_silinda_arima.joblib -> mediumsilinda_arima.joblib if your file names are like that)
        # However, your image shows 'medium_bapanas_arima.joblib', so we stick to the original naming convention for loading
        model_file_name = f"{commodity_type}_{model_type}"
        
        if model_type == 'arima':
            model_path = os.path.join(MODEL_DIR, f"{model_file_name}.joblib")
            if not os.path.exists(model_path):
                return jsonify({'error': f'ARIMA model file not found for {commodity_type}'}), 404
            
            model = joblib.load(model_path)
            
            # For ARIMA, assume the model itself handles forecasting based on its internal state
            # or it expects a fresh series for re-fitting/updating.
            # If your ARIMA model (e.g., pmdarima) expects previous data to make forecast,
            # you might need to pass the last known historical values.
            # However, `model.predict(n_periods=steps_ahead)` is a common pattern for trained models.
            predictions = model.predict(n_periods=steps_ahead)
            predicted_prices = predictions.tolist()

        elif model_type == 'lstm':
            model_path = os.path.join(MODEL_DIR, f"{model_file_name}.h5")
            if not os.path.exists(model_path):
                return jsonify({'error': f'LSTM model file not found for {commodity_type}'}), 404
            
            model = load_model(model_path)
            
            # LSTM requires a sequence of recent historical data as input
            # IMPORTANT: Adjust `look_back` to match the window size used during your LSTM model's training!
            look_back = 10 # Example: assuming your LSTM model was trained with 10 past days
            
            last_n_values = get_last_n_values(historical_data, commodity_type, look_back)
            if len(last_n_values) < look_back:
                return jsonify({'error': f'Not enough historical data for {commodity_type} to make LSTM prediction (need at least {look_back} days). Available: {len(last_n_values)}'}), 400

            # Reshape input for LSTM: (batch_size, timesteps, features)
            # Here: (1, look_back, 1)
            input_sequence = np.array(last_n_values).reshape(1, look_back, 1)

            predicted_prices = []
            current_input = input_sequence
            
            for _ in range(steps_ahead):
                prediction = model.predict(current_input, verbose=0)[0][0] # verbose=0 to suppress Keras output
                predicted_prices.append(prediction)
                
                # Update input sequence for the next prediction (sliding window)
                new_value = prediction # Use the previous prediction as input for the next
                current_input = np.append(current_input[:, 1:, :], [[[new_value]]], axis=1) # Shift the window

            predicted_prices = [round(p) for p in predicted_prices] # Round to nearest integer

        else:
            return jsonify({'error': 'Invalid model type. Choose "arima" or "lstm"'}), 400
        
        return jsonify({'predictions': predicted_prices}), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({'error': f'An error occurred during prediction: {str(e)}. Check server logs for details.'}), 500

if __name__ == '__main__':
    # This block is only executed when running `python main.py` directly for local Flask dev server
    # It will not be used when running via gunicorn/uvicorn.
    app.run(debug=True, port=5000)