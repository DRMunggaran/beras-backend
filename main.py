import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS untuk mengizinkan permintaan dari frontend Next.js

# Path ke folder model dan data historis
MODEL_DIR = 'model'
DATA_HARGA_PATH = 'data_harga.json' # Pastikan path ini benar di root backend

# Cache untuk data historis
HISTORICAL_DATA_CACHE = None

def load_historical_data():
    """Memuat data historis dari data_harga.json."""
    global HISTORICAL_DATA_CACHE
    if HISTORICAL_DATA_CACHE is None:
        try:
            with open(DATA_HARGA_PATH, 'r') as f:
                HISTORICAL_DATA_CACHE = json.load(f)
            # Urutkan berdasarkan tanggal jika belum diurutkan
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
    
    # Ambil entri terakhir (asumsi data sudah terurut berdasarkan tanggal)
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

    # Pastikan commodity_type valid
    valid_commodity_types = ['medium_silinda', 'premium_silinda', 'medium_bapanas', 'premium_bapanas']
    if commodity_type not in valid_commodity_types:
        return jsonify({'error': 'Invalid commodity type'}), 400

    # Muat data historis dari cache
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
            
            # ARIMA model memerlukan data historis lengkap yang digunakan saat training atau window yang sama
            # Untuk prediksi ARIMA, kita akan menggunakan semua data historis yang tersedia sebagai input
            # atau setidaknya data yang cukup untuk memuaskan model.
            # Mengambil semua data untuk jenis beras tertentu
            series_data = [entry[commodity_type] for entry in historical_data if commodity_type in entry]
            
            # Konversi ke pandas Series dengan index tanggal
            dates = [entry['date'] for entry in historical_data if 'date' in entry]
            # Pastikan panjang dates dan series_data sama
            if len(dates) == len(series_data):
                date_index = pd.to_datetime(dates)
                series = pd.Series(series_data, index=date_index)
            else:
                # Fallback jika ada ketidakcocokan, atau bisa juga raise error
                series = pd.Series(series_data) # Tanpa indeks tanggal yang spesifik
            
            # Fit ulang model ARIMA jika diperlukan atau gunakan `update` method jika model mendukung.
            # Namun, untuk model yang sudah di-train, Anda bisa langsung menggunakan `predict` dengan data terakhir.
            # Implementasi `predict` dari statsmodels ARIMA biasanya bisa menerima `start` dan `end` index
            # atau `dynamic=True` untuk memprediksi ke depan.
            # Lebih aman adalah `model.forecast(steps=steps_ahead)` setelah model di-fit dengan data terbaru
            # atau menggunakan data asli yang digunakan saat training.
            
            # Karena model sudah diload, kita asumsikan model telah "belajar" dari data historis
            # dan kita hanya perlu melakukan forecast.
            # Jika model ARIMA disimpan *setelah* di-fit, Anda bisa langsung memanggil `predict` atau `forecast`.
            # Untuk kasus sederhana ini, kita akan asumsikan `forecast` dapat langsung digunakan.
            # Jika model ARIMA dari `pmdarima` maka `model.predict(n_periods=steps_ahead)`
            # Jika dari `statsmodels` maka `model.forecast(steps=steps_ahead)`
            
            # Penting: Jika model ARIMA Anda perlu diupdate dengan data terbaru setiap kali prediksi,
            # Anda harus menyimpan objek model ARIMA itu sendiri (bukan hanya parameter),
            # dan menggunakan method `update` atau `append` atau me-re-fit sebagian.
            # Untuk contoh ini, kita akan asumsikan model yang diload bisa langsung melakukan forecast.
            # Asumsi: model yang disimpan adalah objek `ARIMAResultsWrapper` atau sejenisnya
            # yang bisa langsung memanggil `forecast`.
            # Jika yang disimpan adalah `pmdarima.ARIMA` model:
            predictions = model.predict(n_periods=steps_ahead)
            predicted_prices = predictions.tolist()


        elif model_type == 'lstm':
            model_file = f"{model_path_prefix}.h5"
            if not os.path.exists(model_file):
                return jsonify({'error': f'LSTM model file not found for {commodity_type}'}), 404
            
            model = load_model(model_file)
            
            # LSTM memerlukan urutan data terakhir sebagai input
            # Asumsi: model LSTM Anda dilatih dengan window size (misal, 10 hari terakhir)
            # Anda perlu tahu `look_back` atau `window_size` yang digunakan saat melatih model LSTM Anda.
            # Untuk contoh, kita asumsikan 10 hari terakhir.
            look_back = 10 # Ganti dengan nilai look_back yang sebenarnya dari model Anda
            
            last_n_values = get_last_n_values(historical_data, commodity_type, look_back)
            if len(last_n_values) < look_back:
                return jsonify({'error': f'Not enough historical data for {commodity_type} to make LSTM prediction (need at least {look_back} days)'}), 400

            # Normalisasi data input jika model LSTM Anda dilatih dengan data yang dinormalisasi.
            # Jika model Anda tidak menggunakan normalisasi, lewati bagian ini.
            # Untuk contoh ini, kita *tidak* akan melakukan normalisasi di API,
            # asumsi model sudah dilatih dengan data asli atau normalisasi ditangani di dalam model (kurang umum).
            # Jika Anda menggunakan StandardScaler, Anda perlu menyimpannya dan memuatnya di sini.
            # scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib')) # Contoh memuat scaler
            # last_n_values_scaled = scaler.transform(np.array(last_n_values).reshape(-1, 1))

            # Reshape input untuk LSTM: (1, timesteps, features)
            # Asumsi: timesteps = look_back, features = 1
            input_sequence = np.array(last_n_values).reshape(1, look_back, 1) # Jika dinormalisasi, gunakan `last_n_values_scaled`

            predicted_prices = []
            current_input = input_sequence
            
            for _ in range(steps_ahead):
                prediction = model.predict(current_input)[0][0] # Ambil nilai prediksi
                predicted_prices.append(prediction)
                
                # Update input sequence untuk prediksi berikutnya (sliding window)
                new_value = prediction # Gunakan prediksi sebelumnya sebagai input untuk prediksi berikutnya
                current_input = np.append(current_input[:, 1:, :], [[[new_value]]], axis=1) # Geser window

            # Denormalisasi data output jika model LSTM Anda dinormalisasi.
            # predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten().tolist()
            
            # Bulatkan hasil prediksi jika diperlukan
            predicted_prices = [round(p) for p in predicted_prices]

        else:
            return jsonify({'error': 'Invalid model type. Choose "arima" or "lstm"'}), 400

        # Ambil harga terkini (harga hari ini) untuk perhitungan persentase perubahan jika diperlukan
        # Ini tidak wajib dikirim kembali, tapi bisa membantu jika frontend tidak punya data ini.
        # current_price = get_most_recent_data(historical_data, commodity_type)
        
        return jsonify({'predictions': predicted_prices}), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # Untuk local testing
    app.run(debug=True, port=5000)