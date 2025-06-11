import os
import json
import traceback # Untuk mencetak traceback lengkap ke log untuk debugging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib # Digunakan untuk model ARIMA
from tensorflow.keras.models import load_model # Digunakan untuk model LSTM
import numpy as np # Digunakan oleh TensorFlow dan Joblib
import pandas as pd # Seringkali digunakan oleh pustaka ML
from datetime import datetime, timedelta
# from uvicorn.wsgi import WSGIMiddleware # BARIS INI DIHAPUS

# Inisialisasi aplikasi Flask Anda
app = Flask(__name__)

# Aktifkan CORS untuk mengizinkan permintaan dari frontend Next.js.
CORS(app) 

# asgi_app = WSGIMiddleware(app) # BARIS INI DIHAPUS, ganti dengan langsung menggunakan `app`

# Definisikan path ke folder model dan file data historis Anda.
MODEL_DIR = 'model' 
DATA_HARGA_PATH = 'data_harga.json' 

# Gunakan cache global untuk data historis agar hanya dimuat sekali saat aplikasi dimulai.
HISTORICAL_DATA_CACHE = None

def load_historical_data():
    """
    Memuat data historis dari data_harga.json ke dalam cache.
    """
    global HISTORICAL_DATA_CACHE
    if HISTORICAL_DATA_CACHE is None:
        try:
            print(f"[*] Attempting to load historical data from: {DATA_HARGA_PATH}")
            with open(DATA_HARGA_PATH, 'r') as f:
                data_from_file = json.load(f)
            data_from_file.sort(key=lambda x: x['date'])
            HISTORICAL_DATA_CACHE = data_from_file
            print(f"[*] Successfully loaded historical data. Total entries: {len(HISTORICAL_DATA_CACHE)}")
        except FileNotFoundError:
            print(f"[ERROR] Historical data file NOT FOUND at: {DATA_HARGA_PATH}. Please ensure it's in the backend root.")
            HISTORICAL_DATA_CACHE = []
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON from {DATA_HARGA_PATH}. Check file format for errors.")
            HISTORICAL_DATA_CACHE = []
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while loading historical data: {e}")
            traceback.print_exc()
            HISTORICAL_DATA_CACHE = []
    return HISTORICAL_DATA_CACHE

def get_last_n_values(data, commodity_type, n):
    """
    Mendapatkan n nilai harga terakhir dari data historis untuk jenis komoditas tertentu.
    """
    prices = [entry.get(commodity_type) for entry in data if commodity_type in entry and entry.get(commodity_type) is not None]
    if len(prices) < n:
        print(f"[WARNING] Not enough historical data for {commodity_type}. Required: {n}, Available: {len(prices)}")
    return prices[-n:]

@app.before_request
def before_first_request():
    """
    Fungsi ini dijalankan oleh Flask sebelum memproses permintaan pertama.
    """
    if HISTORICAL_DATA_CACHE is None:
        load_historical_data()

@app.route('/')
def home():
    """
    Endpoint root untuk memeriksa apakah API berjalan.
    """
    return "API Prediksi Harga Beras Berjalan!"

@app.route('/predict/<model_type>/<commodity_type>', methods=['GET'])
def predict(model_type, commodity_type):
    """
    Endpoint utama untuk melakukan prediksi harga beras menggunakan model yang dipilih.
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
    if not historical_data or len(historical_data) == 0:
        load_historical_data() 
        if not HISTORICAL_DATA_CACHE or len(HISTORICAL_DATA_CACHE) == 0:
            print("[ERROR] Historical data is still empty after attempt to load in predict route.")
            return jsonify({'error': 'Historical data not available on server. Cannot make predictions.'}), 500

    try:
        model_file_name_prefix = f"{commodity_type}_{model_type}"
        predicted_prices = []

        if model_type == 'arima':
            model_path = os.path.join(MODEL_DIR, f"{model_file_name_prefix}.joblib")
            if not os.path.exists(model_path):
                print(f"[ERROR] ARIMA model file not found at: {model_path}")
                return jsonify({'error': f'ARIMA model file not found for {commodity_type}'}), 404
            
            print(f"[*] Loading ARIMA model from: {model_path}")
            model = joblib.load(model_path)
            predictions = model.predict(n_periods=steps_ahead)
            predicted_prices = predictions.tolist()
            print(f"[*] ARIMA prediction successful for {commodity_type} for {steps_ahead} steps. Predicted: {predicted_prices}")

        elif model_type == 'lstm':
            model_path = os.path.join(MODEL_DIR, f"{model_file_name_prefix}.h5")
            if not os.path.exists(model_path):
                print(f"[ERROR] LSTM model file not found at: {model_path}")
                return jsonify({'error': f'LSTM model file not found for {commodity_type}'}), 404
            
            print(f"[*] Loading LSTM model from: {model_path}")
            model = load_model(model_path)
            
            look_back = 10 # GANTI DENGAN NILAI AKTUAL DARI MODEL ANDA!
            
            last_n_values = get_last_n_values(historical_data, commodity_type, look_back)
            
            if len(last_n_values) < look_back:
                print(f"[ERROR] Not enough historical data for LSTM prediction for {commodity_type}. Need {look_back} days, got {len(last_n_values)}.")
                return jsonify({'error': f'Not enough historical data for {commodity_type} to make LSTM prediction (need at least {look_back} days). Available: {len(last_n_values)}'}), 400

            input_sequence = np.array(last_n_values).reshape(1, look_back, 1)

            current_input_for_loop = input_sequence
            for i in range(steps_ahead):
                prediction = model.predict(current_input_for_loop, verbose=0)[0][0]
                predicted_prices.append(prediction)
                new_value = prediction 
                current_input_for_loop = np.append(current_input_for_loop[:, 1:, :], [[[new_value]]], axis=1)
            
            predicted_prices = [round(p) for p in predicted_prices]
            print(f"[*] LSTM prediction successful for {commodity_type} for {steps_ahead} steps. Predicted: {predicted_prices}")

        else:
            return jsonify({'error': 'Invalid model type. Choose "arima" or "lstm"'}), 400
        
        return jsonify({'predictions': predicted_prices}), 200

    except Exception as e:
        print(f"[CRITICAL ERROR] An unhandled exception occurred during prediction process for {commodity_type} with {model_type}: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred: {str(e)}. Please check server logs.'}), 500

if __name__ == '__main__':
    print("[*] Running Flask development server locally...")
    app.run(debug=True, port=5000)