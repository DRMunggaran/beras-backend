# beras-backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np
import tensorflow as tf
import logging
import pandas as pd # Tambahkan pandas untuk manipulasi data historis
# import statsmodels.tsa.arima.model as smt # Biasanya tidak perlu diimpor langsung jika model sudah diload

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Konfigurasi CORS ---
# Ganti dengan domain frontend Next.js-mu yang sebenarnya di Railway.
# Tambahkan juga localhost:3000 untuk pengembangan lokal.
origins = [
    "https://web-production-YOUR-FRONTEND-ID.up.railway.app", # Ganti dengan URL frontend Railway-mu!
    "http://localhost:3000",                               # Untuk pengembangan lokal Next.js
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pemuatan Model NYATA ---
# Model akan dimuat SEKALI saat aplikasi startup.
# Pastikan jalur 'model/' sudah benar di lingkungan deployment Railway.
# Pastikan versi library Python di requirements.txt cocok dengan saat model dilatih.
arima_models = {}
lstm_models = {}

try:
    logger.info("Memuat model ARIMA...")
    arima_models["medium_silinda"] = load("model/medium_silinda_arima.joblib")
    arima_models["premium_silinda"] = load("model/premium_silinda_arima.joblib")
    arima_models["medium_bapanas"] = load("model/medium_bapanas_arima.joblib")
    arima_models["premium_bapanas"] = load("model/premium_bapanas_arima.joblib")
    logger.info("Model ARIMA berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat model ARIMA: {e}")
    raise Exception(f"Gagal memuat salah satu model ARIMA: {e}") # Hentikan startup jika model penting tidak ada

try:
    logger.info("Memuat model LSTM...")
    # Penting: Pastikan TensorFlow tidak mencoba menggunakan GPU jika tidak tersedia.
    # Ini membantu menghindari error di lingkungan tanpa GPU.
    tf.config.set_visible_devices([], 'GPU')
    lstm_models["medium_silinda"] = tf.keras.models.load_model("model/medium_silinda_lstm.h5")
    lstm_models["premium_silinda"] = tf.keras.models.load_model("model/premium_silinda_lstm.h5")
    lstm_models["medium_bapanas"] = tf.keras.models.load_model("model/medium_bapanas_lstm.h5")
    lstm_models["premium_bapanas"] = tf.keras.models.load_model("model/premium_bapanas_lstm.h5")
    logger.info("Model LSTM berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat model LSTM: {e}")
    raise Exception(f"Gagal memuat salah satu model LSTM: {e}") # Hentikan startup jika model penting tidak ada


# --- Skema Request Prediksi ---
class PredictRequest(BaseModel):
    category: str
    steps_ahead: int = 1
    last_values: list[float] # Ini akan menerima 10 nilai terakhir dari frontend

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "API Prediksi Harga Beras - Siap!"}

# --- Endpoint Prediksi ARIMA ---
@app.post("/predict/{category_name}")
async def predict_arima(category_name: str, req: PredictRequest):
    logger.info(f"Menerima request ARIMA untuk {category_name} dengan {req.steps_ahead} langkah.")
    logger.info(f"ARIMA menerima {len(req.last_values)} last_values.")

    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    model = arima_models.get(category_name.lower())

    if not model:
        raise HTTPException(status_code=404, detail=f"Model ARIMA untuk kategori '{category_name}' tidak ditemukan.")

    try:
        # PENTING UNTUK ARIMA (statsmodels):
        # Model statsmodels.tsa.arima.model.ARIMAResultsWrapper yang di-load dari joblib
        # biasanya melakukan forecast dari END dari data yang digunakan saat model itu di-FIT.
        # `last_values` yang kamu kirimkan dari frontend TIDAK secara otomatis digunakan oleh `model.forecast()`.
        # Untuk ARIMA agar memprediksi berdasarkan data historis terbaru, ada dua pendekatan:
        # 1. Re-fit model dengan data terbaru (sangat mahal untuk setiap request).
        # 2. Menggunakan `model.apply(new_data_segment).forecast()` atau `model.predict(start=..., end=...)`
        #    jika modelmu dilatih untuk mengambil input `endog`.
        # Karena common use case `joblib` adalah load hasil `.fit()`, kita akan pakai `.forecast()`.
        # Prediksi akan konstan jika `steps_ahead` sangat besar karena konvergen ke mean.
        
        # Contoh jika model ARIMA kamu memerlukan data terbaru untuk `apply` atau `predict`:
        # latest_data_series = pd.Series(req.last_values)
        # updated_model_results = model.apply(latest_data_series)
        # forecast = updated_model_results.forecast(steps=req.steps_ahead)
        
        # Jika modelmu adalah hasil dari `fit()` dan bisa langsung `forecast()`:
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
    logger.info(f"LSTM menerima {len(req.last_values)} last_values.")

    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    model = lstm_models.get(category_name.lower())

    if not model:
        raise HTTPException(status_code=404, detail=f"Model LSTM untuk kategori '{category_name}' tidak ditemukan.")

    # Validasi panjang `last_values`: kita harapkan 10 nilai dari frontend
    if not req.last_values or len(req.last_values) < 10:
        raise HTTPException(status_code=400, detail="Minimal 10 nilai historis diperlukan untuk prediksi LSTM.")

    try:
        # Untuk LSTM, kita akan memproses `last_values` sesuai dengan input shape modelmu.
        # Asumsi model LSTM-mu dilatih dengan 7 time steps (sequence length 7).
        # Jadi kita ambil 7 nilai terakhir dari 10 yang dikirim frontend.
        time_steps = 7 # Sesuaikan dengan time_steps model LSTM-mu saat dilatih
        if len(req.last_values) < time_steps:
             raise HTTPException(status_code=400, detail=f"LSTM memerlukan {time_steps} nilai, hanya menerima {len(req.last_values)}.")
             
        # Ambil `time_steps` nilai terakhir dan reshape ke (1, time_steps, 1)
        input_sequence_for_model = np.array(req.last_values[-time_steps:]).reshape((1, time_steps, 1))

        preds = []
        # Lakukan prediksi berulang untuk `steps_ahead`
        for _ in range(req.steps_ahead):
            # predict() dari TensorFlow Keras models:
            # verbose=0 untuk menekan output log selama prediksi
            # [0][0] untuk mendapatkan nilai skalar dari output prediksi model
            pred = model.predict(input_sequence_for_model, verbose=0)[0][0]
            preds.append(float(pred)) # Konversi ke float standar Python

            # Update input sequence untuk prediksi multi-langkah (jika steps_ahead > 1)
            # Ini mensimulasikan feeding the new prediction back into the input sequence
            input_sequence_for_model = np.append(input_sequence_for_model[:, 1:, :], [[[pred]]], axis=1)

        return {
            "model": "LSTM",
            "category": category_name,
            "prediction": preds
        }
    except Exception as e:
        logger.error(f"Error saat prediksi LSTM untuk {category_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses prediksi LSTM: {e}")