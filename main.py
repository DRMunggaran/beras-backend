import os
import json
import traceback # Untuk mencetak traceback lengkap ke log untuk debugging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib # Digunakan untuk model ARIMA
from tensorflow.keras.models import load_model # Digunakan untuk model LSTM
import numpy as np # Digunakan oleh TensorFlow dan Joblib
import pandas as pd # Seringkali digunakan oleh pustaka ML, meskipun tidak langsung di kode ini
from datetime import datetime, timedelta
from uvicorn.wsgi import WSGIMiddleware # Adaptor WSGI ke ASGI

# Inisialisasi aplikasi Flask Anda
app = Flask(__name__)

# Aktifkan CORS untuk mengizinkan permintaan dari frontend Next.js.
# Untuk keamanan yang lebih baik di produksi, pertimbangkan untuk membatasi `origins` ke URL frontend Anda yang sebenarnya.
CORS(app) 

# Wrap aplikasi Flask (WSGI) dengan WSGIMiddleware untuk membuatnya kompatibel dengan Uvicorn (ASGI).
# Ini mengatasi error umum seperti `TypeError: Flask.__call__() missing 1 required positional argument: 'start_response'`.
asgi_app = WSGIMiddleware(app)

# Definisikan path ke folder model dan file data historis Anda.
# Pastikan folder 'model' dan file 'data_harga.json' berada di root direktori backend Anda di Railway.
MODEL_DIR = 'model' 
DATA_HARGA_PATH = 'data_harga.json' 

# Gunakan cache global untuk data historis agar hanya dimuat sekali saat aplikasi dimulai.
HISTORICAL_DATA_CACHE = None

def load_historical_data():
    """
    Memuat data historis dari data_harga.json ke dalam cache saat aplikasi dimulai.
    Ini penting agar data siap saat permintaan prediksi datang.
    """
    global HISTORICAL_DATA_CACHE
    if HISTORICAL_DATA_CACHE is None:
        try:
            print(f"[*] Attempting to load historical data from: {DATA_HARGA_PATH}")
            with open(DATA_HARGA_PATH, 'r') as f:
                data_from_file = json.load(f)
            # Penting: Urutkan data berdasarkan tanggal untuk memastikan urutan yang benar
            # Ini krusial untuk model time series yang mengandalkan urutan data.
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
            traceback.print_exc() # Cetak traceback lengkap untuk debugging di log Railway
            HISTORICAL_DATA_CACHE = []
    return HISTORICAL_DATA_CACHE

def get_last_n_values(data, commodity_type, n):
    """
    Mendapatkan n nilai harga terakhir dari data historis untuk jenis komoditas tertentu.
    Ini digunakan sebagai input (sequence) untuk model LSTM.
    """
    prices = [entry.get(commodity_type) for entry in data if commodity_type in entry and entry.get(commodity_type) is not None]
    if len(prices) < n:
        print(f"[WARNING] Not enough historical data for {commodity_type}. Required: {n}, Available: {len(prices)}")
        # Jika data tidak cukup, kembalikan semua yang ada, atau mungkin lemparkan error
    return prices[-n:]

@app.before_request
def before_first_request():
    """
    Fungsi ini dijalankan oleh Flask sebelum memproses permintaan pertama.
    Digunakan untuk memuat data historis ke cache jika belum dimuat, memastikan ketersediaan.
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
    
    Args:
        model_type (str): Tipe model yang akan digunakan ('arima' atau 'lstm').
        commodity_type (str): Jenis beras yang akan diprediksi (misalnya 'medium_silinda').
    Query Params:
        steps_ahead (int): Berapa hari ke depan untuk diprediksi. Default adalah 1.
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

    # Pastikan data historis tersedia di cache dan tidak kosong sebelum memproses permintaan.
    historical_data = HISTORICAL_DATA_CACHE
    if not historical_data or len(historical_data) == 0:
        # Jika cache kosong, coba muat ulang (mungkin gagal pada startup awal)
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
            # joblib.load() akan memuat model ARIMA yang disimpan.
            # Asumsi: model yang diload adalah objek yang sudah fit dan bisa langsung memprediksi `n_periods` ke depan.
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
            # load_model() dari TensorFlow/Keras akan memuat model .h5.
            model = load_model(model_path)
            
            # PENTING: Sesuaikan `look_back` dengan ukuran window yang digunakan saat melatih model LSTM Anda!
            # Nilai ini HARUS cocok dengan hyperparameter yang digunakan selama training.
            look_back = 10 # Default contoh. GANTI DENGAN NILAI AKTUAL DARI MODEL ANDA!
            
            last_n_values = get_last_n_values(historical_data, commodity_type, look_back)
            
            if len(last_n_values) < look_back:
                print(f"[ERROR] Not enough historical data for LSTM prediction for {commodity_type}. Need {look_back} days, got {len(last_n_values)}.")
                return jsonify({'error': f'Not enough historical data for {commodity_type} to make LSTM prediction (need at least {look_back} days). Available: {len(last_n_values)}'}), 400

            # Reshape input untuk LSTM: (batch_size, timesteps, features)
            # Untuk satu sequence dan satu fitur (harga), bentuknya adalah (1, look_back, 1).
            input_sequence = np.array(last_n_values).reshape(1, look_back, 1)

            # Lakukan prediksi iteratif untuk `steps_ahead` menggunakan sliding window.
            # Ini berarti setiap prediksi baru digunakan sebagai input untuk prediksi berikutnya.
            current_input_for_loop = input_sequence
            for i in range(steps_ahead):
                # `verbose=0` untuk menekan output log Keras/TensorFlow selama prediksi.
                prediction = model.predict(current_input_for_loop, verbose=0)[0][0]
                predicted_prices.append(prediction)
                
                # Update input sequence: geser window ke depan dengan menambahkan prediksi baru.
                new_value = prediction 
                current_input_for_loop = np.append(current_input_for_loop[:, 1:, :], [[[new_value]]], axis=1)
            
            # Bulatkan hasil prediksi ke integer terdekat (harga biasanya bilangan bulat).
            predicted_prices = [round(p) for p in predicted_prices]
            print(f"[*] LSTM prediction successful for {commodity_type} for {steps_ahead} steps. Predicted: {predicted_prices}")

        else:
            return jsonify({'error': 'Invalid model type. Choose "arima" or "lstm"'}), 400
        
        return jsonify({'predictions': predicted_prices}), 200

    except Exception as e:
        # Tangani error tak terduga dan cetak traceback lengkap ke log server Railway.
        # Ini sangat membantu untuk debugging masalah yang tidak terlihat di lingkungan lokal.
        print(f"[CRITICAL ERROR] An unhandled exception occurred during prediction process for {commodity_type} with {model_type}: {e}")
        traceback.print_exc() # Mencetak tumpukan panggilan lengkap ke log
        return jsonify({'error': f'An internal server error occurred: {str(e)}. Please check server logs for details.'}), 500

# Blok ini hanya akan dieksekusi saat Anda menjalankan `python main.py` secara langsung dari terminal.
# Ini untuk lingkungan pengembangan lokal menggunakan server pengembangan Flask built-in.
# Gunicorn/Uvicorn tidak akan menggunakan blok ini saat deploy ke Railway.
if __name__ == '__main__':
    print("[*] Running Flask development server locally...")
    app.run(debug=True, port=5000)