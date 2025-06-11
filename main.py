import os
import json
import traceback # Diperlukan untuk mencetak traceback lengkap ke log untuk debugging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib # Digunakan untuk model ARIMA dan mungkin scaler
# Tambahkan impor lapisan Keras secara eksplisit untuk mendefinisikan model LSTM (untuk workaround)
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import LSTM, Dense 
import numpy as np # Digunakan oleh TensorFlow dan Joblib
import pandas as pd # Seringkali digunakan oleh pustaka ML
from datetime import datetime, timedelta

# Inisialisasi aplikasi Flask Anda
app = Flask(__name__)

# Aktifkan CORS untuk mengizinkan permintaan dari frontend Next.js.
# Untuk keamanan yang lebih baik di produksi, pertimbangkan untuk membatasi `origins` ke URL frontend Anda yang sebenarnya.
CORS(app) 

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
        # Jika data tidak cukup, kembalikan semua yang ada
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
            model = joblib.load(model_path)
            predictions_raw = model.predict(n_periods=steps_ahead) # Gunakan nama berbeda agar tidak konflik
            
            # --- Perbaikan: Inverse Scaling untuk Prediksi ARIMA ---
            # Jika model ARIMA Anda dilatih dengan data yang diskalakan (misal, StandardScaler, MinMaxScaler),
            # Anda perlu memuat objek scaler yang sama dan menggunakannya untuk mengubah
            # prediksi kembali ke skala harga aslinya.
            # Jika tidak dilakukan, prediksi bisa menjadi 0 atau nilai yang sangat kecil/tidak realistis.
            scaler_path = os.path.join(MODEL_DIR, f"{commodity_type}_arima_scaler.joblib")
            if os.path.exists(scaler_path):
                print(f"[*] Loading ARIMA scaler from: {scaler_path}")
                scaler = joblib.load(scaler_path)
                # Skala balik prediksi. Asumsi: scaler.transform menerima dan mengembalikan array 2D (n_samples, n_features).
                # Jadi, prediksi harus di-reshape (misal, dari 1D menjadi 2D) sebelum inverse_transform.
                predictions_reshaped = np.array(predictions_raw).reshape(-1, 1)
                predicted_prices_original_scale = scaler.inverse_transform(predictions_reshaped).flatten().tolist()
                predicted_prices = predicted_prices_original_scale
                print("[*] ARIMA predictions inverse-scaled successfully.")
            else:
                # Jika tidak ada scaler file atau model tidak diskalakan, gunakan prediksi langsung.
                predicted_prices = predictions_raw.tolist()
                print("[*] ARIMA predictions used directly (no inverse scaling applied, no scaler file found).")
            # --- Akhir Perbaikan Scaling ---

            # Bulatkan hasil prediksi ke integer terdekat (harga biasanya bilangan bulat).
            predicted_prices = [round(p) for p in predicted_prices]
            print(f"[*] ARIMA prediction successful for {commodity_type} for {steps_ahead} steps. Predicted: {predicted_prices}")

        elif model_type == 'lstm':
            model_path = os.path.join(MODEL_DIR, f"{model_file_name_prefix}.h5")
            if not os.path.exists(model_path):
                print(f"[ERROR] LSTM model file not found at: {model_path}")
                return jsonify({'error': f'LSTM model file not found for {commodity_type}'}), 404
            
            print(f"[*] Attempting to load LSTM model/weights from: {model_path}")
            # --- Perbaikan: Mengatasi 'Unrecognized keyword arguments: ['batch_shape']' TANPA MELATIH ULANG ---
            # Error ini terjadi karena model `.h5` disimpan dengan versi TensorFlow yang berbeda/lebih baru
            # dari yang digunakan di Railway (TensorFlow 2.10.0).
            # Solusi di sini adalah MENDIFINISIKAN ULANG arsitektur model di kode, 
            # kemudian hanya memuat bobot (weights) dari file `.h5`.
            
            # PENTING: Anda HARUS mengganti arsitektur model di bawah ini agar SAMA PERSIS
            # dengan arsitektur model LSTM yang Anda latih dan simpan ke file .h5 Anda.
            # Jika tidak sama persis (misal: jumlah layer, jenis layer, unit, aktivasi, dll.),
            # ini akan menyebabkan error lain atau, lebih buruk, prediksi yang sangat salah.
            # look_back ini adalah `timesteps` dari input_shape model Anda.
            
            look_back = 10 # Default contoh. GANTI DENGAN NILAI AKTUAL DARI MODEL ANDA!

            # --- BEGIN CUSTOM LSTM MODEL ARCHITECTURE DEFINITION (WAJIB DISESUAIKAN!) ---
            try:
                # Contoh: Model Sequential dengan satu LSTM layer dan satu Dense output layer.
                # SESUAIKAN `units` (jumlah neuron LSTM) dan `activation` (fungsi aktivasi)
                # serta jumlah layer lain jika model Anda lebih kompleks.
                model = Sequential([
                    LSTM(units=50, activation='relu', input_shape=(look_back, 1)), 
                    Dense(1) # Output layer: 1 unit untuk prediksi harga tunggal
                ])
                # Kompilasi model (diperlukan sebelum memuat bobot pada beberapa versi Keras/TF)
                # Sesuaikan optimizer dan loss jika penting (tidak terlalu krusial untuk inferensi bobot saja)
                model.compile(optimizer='adam', loss='mse') 
                
                # Muat hanya bobot (weights) dari file .h5 ke dalam arsitektur yang sudah didefinisikan.
                model.load_weights(model_path)
                print(f"[*] LSTM model architecture defined and weights loaded successfully from {model_path}.")
            except Exception as model_def_error:
                print(f"[CRITICAL ERROR] Failed to define or load weights for LSTM model: {model_def_error}")
                traceback.print_exc()
                return jsonify({'error': f'Failed to load LSTM model architecture or weights: {str(model_def_error)}. Please ensure model architecture in main.py matches the saved model.'}), 500
            # --- AKHIR CUSTOM LSTM MODEL ARCHITECTURE DEFINITION ---
            
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
        # Tangani error tak terduga dan cetak traceback lengkap ke log server Railway.
        print(f"[CRITICAL ERROR] An unhandled exception occurred during prediction process for {commodity_type} with {model_type}: {e}")
        traceback.print_exc() # Mencetak tumpukan panggilan lengkap ke log
        return jsonify({'error': f'An internal server error occurred: {str(e)}. Please check server logs.'}), 500

# Blok ini hanya akan dieksekusi saat Anda menjalankan `python main.py` secara langsung dari terminal.
# Ini untuk lingkungan pengembangan lokal menggunakan server pengembangan Flask built-in.
# Gunicorn tidak akan menggunakan blok ini saat deploy ke Railway.
if __name__ == '__main__':
    print("[*] Running Flask development server locally...")
    app.run(debug=True, port=5000)