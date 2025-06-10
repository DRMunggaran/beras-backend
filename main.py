# beras_backend-main/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np
import tensorflow as tf
import os
import logging # Import modul logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Konfigurasi CORS (PENTING untuk komunikasi Frontend-Backend) ---
# Sesuaikan 'origins' dengan domain frontend Next.js Anda yang di-deploy di Railway.
# Tambahkan juga localhost untuk pengembangan lokal.
origins = [
    "https://web-production-7f140.up.railway.app", # Ganti dengan domain frontend Railway Anda yang sebenarnya
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Mengizinkan domain yang spesifik
    allow_credentials=True,      # Mengizinkan kredensial (misalnya cookies, header otorisasi)
    allow_methods=["*"],         # Mengizinkan semua metode HTTP (GET, POST, dll.)
    allow_headers=["*"],         # Mengizinkan semua header
)

# --- Pemuatan Model ---
# Disarankan untuk memuat model di luar endpoint atau fungsi,
# agar hanya dimuat sekali saat aplikasi startup.
# Pastikan jalur 'model/' sudah benar di lingkungan deployment Railway.
try:
    logger.info("Memuat model ARIMA...")
    arima_models = {
        "medium_silinda": load("model/medium_silinda_arima.joblib"),
        "premium_silinda": load("model/premium_silinda_arima.joblib"),
        "medium_bapanas": load("model/medium_bapanas_arima.joblib"),
        "premium_bapanas": load("model/premium_bapanas_arima.joblib")
    }
    logger.info("Model ARIMA berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat model ARIMA: {e}")
    # Jika model penting dan aplikasi tidak bisa jalan tanpanya, bisa exit atau raise error
    # raise e

try:
    logger.info("Memuat model LSTM...")
    # Pastikan TensorFlow tidak mencoba menggunakan GPU jika tidak tersedia
    tf.config.set_visible_devices([], 'GPU') # Menonaktifkan penggunaan GPU jika tidak ada
    lstm_models = {
        "medium_silinda": tf.keras.models.load_model("model/medium_silinda_lstm.h5"),
        "premium_silinda": tf.keras.models.load_model("model/premium_silinda_lstm.h5"),
        "medium_bapanas": tf.keras.models.load_model("model/medium_bapanas_lstm.h5"),
        "premium_bapanas": tf.keras.models.load_model("model/premium_bapanas_lstm.h5")
    }
    logger.info("Model LSTM berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat model LSTM: {e}")
    # raise e

# --- Skema Request Prediksi ---
class PredictRequest(BaseModel):
    # model_type: str  # Dihapus karena frontend akan memanggil endpoint spesifik
    category: str      # "medium_silinda", "premium_silinda", etc
    steps_ahead: int = 1 # Jumlah hari prediksi ke depan (diubah dari 'steps' ke 'steps_ahead' agar konsisten dengan frontend)
    last_values: list[float] = [] # Hanya untuk LSTM, nilai historis yang dibutuhkan

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "API Prediksi Harga Beras (ARIMA & LSTM) V1.0"}

# --- Endpoint Prediksi ARIMA ---
# Perubahan: Mengubah dari satu endpoint /predict menjadi endpoint spesifik
# Ini cocok dengan cara fungsi predictXxxxxArima Anda di frontend.
@app.post("/predict/{category_name}")
async def predict_arima(category_name: str, req: PredictRequest):
    # Pastikan category_name dari URL cocok dengan category di body request
    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    model_key = category_name.lower()
    model = arima_models.get(model_key)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model ARIMA untuk kategori '{model_key}' tidak ditemukan.")

    try:
        forecast = model.forecast(steps=req.steps_ahead)
        return {
            "model": "ARIMA",
            "category": model_key,
            "prediction": forecast.tolist() # Mengubah key menjadi 'prediction' agar sesuai dengan frontend
        }
    except Exception as e:
        logger.error(f"Error saat prediksi ARIMA untuk {model_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses prediksi ARIMA: {e}")

# --- Endpoint Prediksi LSTM ---
# Perubahan: Mengubah dari satu endpoint /predict_lstm menjadi endpoint spesifik
# Ini cocok dengan cara fungsi predictXxxxxLstm Anda di frontend.
@app.post("/predict_lstm/{category_name}")
async def predict_lstm(category_name: str, req: PredictRequest):
    # Pastikan category_name dari URL cocok dengan category di body request
    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    model_key = category_name.lower()
    model = lstm_models.get(model_key)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model LSTM untuk kategori '{model_key}' tidak ditemukan.")

    # Validasi input last_values untuk LSTM
    if not req.last_values or len(req.last_values) < 7:
        raise HTTPException(status_code=400, detail="Minimal 7 nilai terakhir diperlukan untuk prediksi LSTM.")

    try:
        # Siapkan input shape: (1, time_steps, features)
        input_seq = np.array(req.last_values[-7:]).reshape((1, 7, 1))
        preds = []
        for _ in range(req.steps_ahead): # Menggunakan steps_ahead
            pred = model.predict(input_seq, verbose=0)[0][0] # verbose=0 untuk menekan output predict
            preds.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        return {
            "model": "LSTM",
            "category": model_key,
            "prediction": preds # Mengubah key menjadi 'prediction' agar sesuai dengan frontend
        }
    except Exception as e:
        logger.error(f"Error saat prediksi LSTM untuk {model_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses prediksi LSTM: {e}")