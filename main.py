# beras-backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
import json # Untuk membaca file JSON
import os # Untuk memeriksa path file

# Konfigurasi logging untuk visibilitas di Railway logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Konfigurasi CORS ---
# Ganti dengan domain frontend Next.js Anda yang sebenarnya di Railway.
# Tambahkan juga http://localhost:3000 untuk pengembangan lokal.
origins = [
    "https://web-production-YOUR-FRONTEND-ID.up.railway.app", # <-- GANTI DENGAN URL ASLI FRONTEND RAILWAY-MU!
    "http://localhost:3000",                               # Untuk pengembangan lokal Next.js
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pemuatan Data Historis Global (dari data_harga.json di BACKEND) ---
# Data historis ini akan dimuat SEKALI saat aplikasi startup.
# Pastikan data_harga.json berada di lokasi yang dapat diakses oleh main.py.
# Misalnya, di root folder backend (`data_harga.json`) atau di folder `model/` (`model/data_harga.json`).
HISTORICAL_DATA_PATH_BACKEND = "data_harga.json" # <--- SESUAIKAN JIKA FILE DI TEMPAT LAIN!
historical_data_cache = {} # Cache untuk menyimpan data historis per kategori

try:
    logger.info(f"Memuat data historis dari {HISTORICAL_DATA_PATH_BACKEND}...")
    with open(HISTORICAL_DATA_PATH_BACKEND, 'r') as f:
        full_data_array_raw = json.load(f)
    full_data_df = pd.DataFrame(full_data_array_raw) # Menggunakan pandas untuk kemudahan akses

    historical_data_cache["medium_silinda"] = full_data_df['medium_silinda'].tolist()
    historical_data_cache["premium_silinda"] = full_data_df['premium_silinda'].tolist()
    historical_data_cache["medium_bapanas"] = full_data_df['medium_bapanas'].tolist()
    historical_data_cache["premium_bapanas"] = full_data_df['premium_bapanas'].tolist()
    logger.info("Data historis berhasil dimuat dan di-cache di backend.")
except FileNotFoundError:
    logger.error(f"File data historis TIDAK DITEMUKAN di backend: {HISTORICAL_DATA_PATH_BACKEND}. Pastikan file ada.")
    raise Exception(f"File data historis tidak ditemukan: {HISTORICAL_DATA_PATH_BACKEND}")
except Exception as e:
    logger.error(f"Gagal memuat data historis di backend: {e}")
    raise Exception(f"Gagal memuat data historis di backend: {e}")


# --- Pemuatan Model dan Scaler NYATA ---
# Model dan Scaler akan dimuat SEKALI saat aplikasi startup.
arima_models = {}
lstm_models = {}
scalers = {} # Untuk scaler yang digunakan LSTM

try:
    logger.info("Memuat model ARIMA...")
    arima_models["medium_silinda"] = load("model/medium_silinda_arima.joblib")
    arima_models["premium_silinda"] = load("model/premium_silinda_arima.joblib")
    arima_models["medium_bapanas"] = load("model/medium_bapanas_arima.joblib")
    arima_models["premium_bapanas"] = load("model/premium_bapanas_arima.joblib")
    logger.info("Model ARIMA berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat model ARIMA: {e}")
    raise Exception(f"Gagal memuat salah satu model ARIMA: {e}")

try:
    logger.info("Memuat model LSTM...")
    tf.config.set_visible_devices([], 'GPU') # Menonaktifkan penggunaan GPU jika tidak tersedia.
    lstm_models["medium_silinda"] = tf.keras.models.load_model("model/medium_silinda_lstm.h5")
    lstm_models["premium_silinda"] = tf.keras.models.load_model("model/premium_silinda_lstm.h5")
    lstm_models["medium_bapanas"] = tf.keras.models.load_model("model/medium_bapanas_lstm.h5")
    lstm_models["premium_bapanas"] = tf.keras.models.load_model("model/premium_bapanas_lstm.h5")
    logger.info("Model LSTM berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat model LSTM: {e}")
    raise Exception(f"Gagal memuat salah satu model LSTM: {e}")

try:
    logger.info("Memuat Scalers...")
    scalers["medium_silinda"] = load("model/scaler_medium_silinda.joblib")
    scalers["premium_silinda"] = load("model/scaler_premium_silinda.joblib")
    scalers["medium_bapanas"] = load("model/scaler_medium_bapanas.joblib")
    scalers["premium_bapanas"] = load("model/scaler_premium_bapanas.joblib")
    logger.info("Scalers berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat Scalers: {e}")
    raise Exception(f"Gagal memuat Scalers: {e}")


# --- Skema Request Prediksi ---
# `last_values` DIHAPUS karena backend akan mengambilnya sendiri dari cache
class PredictRequest(BaseModel):
    category: str
    steps_ahead: int = 1

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "API Prediksi Harga Beras - Berjalan Normal!"}

# --- Endpoint Prediksi ARIMA ---
@app.post("/predict/{category_name}")
async def predict_arima(category_name: str, req: PredictRequest):
    logger.info(f"Menerima request ARIMA untuk {category_name} dengan {req.steps_ahead} langkah.")

    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    model = arima_models.get(category_name.lower())
    historical_data = historical_data_cache.get(category_name.lower()) # Dapatkan data historis penuh dari cache

    if not model:
        raise HTTPException(status_code=404, detail=f"Model ARIMA untuk kategori '{category_name}' tidak ditemukan.")
    if not historical_data:
        raise HTTPException(status_code=404, detail=f"Data historis untuk kategori '{category_name}' tidak ditemukan di cache backend.")

    try:
        # PENTING UNTUK ARIMA (statsmodels):
        # Metode `model.forecast()` dari statsmodels biasanya memprediksi dari data terakhir
        # yang digunakan saat model itu di-FIT.
        # Untuk membuat ARIMA responsif terhadap data terbaru, Anda perlu:
        # 1. Menggunakan `model.append(new_data_series).forecast()` jika model mendukungnya,
        #    atau `model.apply(new_data_series).forecast()`
        # 2. Atau melatih ulang model dengan data historis terbaru (sangat mahal untuk API).
        #
        # Jika Anda tidak melakukan `append`/`apply`/refit, prediksi akan cenderung konvergen
        # ke rata-rata data pelatihan untuk jangka panjang.
        # Di sini, kita asumsikan model sudah dilatih dengan data historis penuh dan
        # `forecast()` akan memprediksi dari konteks training-nya.
        forecast = model.forecast(steps=req.steps_ahead)
        
        return {
            "model": "ARIMA",
            "category": category_name,
            "prediction": forecast.tolist()
        }
    except Exception as e:
        logger.error(f"Error saat prediksi ARIMA untuk {category_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses prediksi ARIMA: {e}")

# --- Endpoint Prediksi LSTM ---
@app.post("/predict_lstm/{category_name}")
async def predict_lstm(category_name: str, req: PredictRequest):
    logger.info(f"Menerima request LSTM untuk {category_name} dengan {req.steps_ahead} langkah.")

    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    model = lstm_models.get(category_name.lower())
    scaler = scalers.get(category_name.lower())
    historical_data = historical_data_cache.get(category_name.lower()) # Dapatkan data historis penuh dari cache

    if not model or not scaler:
        raise HTTPException(status_code=404, detail=f"Model atau Scaler LSTM untuk kategori '{category_name}' tidak ditemukan.")
    if not historical_data:
        raise HTTPException(status_code=404, detail=f"Data historis untuk kategori '{category_name}' tidak ditemukan di cache backend.")

    try:
        # Tentukan `time_steps` (sequence length) yang digunakan model LSTM saat dilatih
        # INI KRUSIAL: Sesuaikan dengan `sequence_length` atau `look_back` saat melatih model LSTM-mu!
        time_steps = 7 # <--- SESUAIKAN NILAI INI! (misal 7 atau 10, tergantung modelmu)

        if len(historical_data) < time_steps:
             raise HTTPException(status_code=400, detail=f"Data historis yang tersedia ({len(historical_data)}) tidak cukup untuk LSTM. Minimal {time_steps} diperlukan.")
             
        # Ambil `time_steps` nilai TERAKHIR dari data historis LENGKAP yang ada di cache backend.
        input_data_for_model = np.array(historical_data[-time_steps:]) # Slice N data terakhir
        
        # Step 1: Scaling Input
        scaled_input_data = scaler.transform(input_data_for_model.reshape(-1, 1)) # Reshape untuk scaler
        
        # Reshape untuk input model LSTM: (batch_size, time_steps, num_features) -> (1, time_steps, 1)
        input_sequence_for_model = scaled_input_data.reshape((1, time_steps, 1))

        preds_scaled = []
        # Lakukan prediksi berulang untuk `steps_ahead`
        for _ in range(req.steps_ahead):
            pred_scaled_value = model.predict(input_sequence_for_model, verbose=0)[0][0]
            preds_scaled.append(float(pred_scaled_value))

            # Update input sequence untuk prediksi multi-langkah (jika steps_ahead > 1)
            new_scaled_input_data = np.append(scaled_input_data[1:], [[pred_scaled_value]], axis=0)
            input_sequence_for_model = new_scaled_input_data.reshape((1, time_steps, 1))
        
        # Step 2: Inverse Scaling Output
        final_predictions_original_scale = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
        final_predictions_list = final_predictions_original_scale.flatten().tolist()

        return {
            "model": "LSTM",
            "category": category_name,
            "prediction": final_predictions_list
        }
    except Exception as e:
        logger.error(f"Error saat prediksi LSTM untuk {category_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses prediksi LSTM: {e}")