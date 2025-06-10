# beras-backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import random # Hanya untuk simulasi prediksi
import time # Untuk simulasi delay loading

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Konfigurasi CORS ---
# Ganti dengan domain frontend Next.js Anda yang sebenarnya di Railway.
# Tambahkan juga localhost:3000 untuk pengembangan lokal.
origins = [
    "https://your-frontend-id.up.railway.app", # Ganti dengan URL frontend Railway Anda
    "http://localhost:3000",                   # Untuk pengembangan lokal Next.js
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Contoh Pemuatan Model (Simulasi) ---
# Di sini, Anda akan memuat model ML Anda yang sebenarnya.
# Untuk contoh ini, kita hanya akan mensimulasikan model yang selalu mengembalikan angka acak.
# Pastikan 'model/' adalah direktori yang benar jika Anda menyertakan model Anda.
# Contoh:
# try:
#     logger.info("Memuat model...")
#     dummy_arima_model = {"predict": lambda steps: [random.uniform(10000, 15000) for _ in range(steps)]}
#     dummy_lstm_model = {"predict": lambda input_seq: [random.uniform(10000, 15000)]}
#     logger.info("Model dummy berhasil dimuat.")
# except Exception as e:
#     logger.error(f"Gagal memuat model dummy: {e}")
#     # Dalam aplikasi nyata, Anda mungkin ingin raise error agar aplikasi tidak startup
#     # raise e

# --- Skema Request Prediksi ---
class PredictRequest(BaseModel):
    category: str      # Contoh: "medium_silinda"
    steps_ahead: int = 1 # Jumlah hari prediksi ke depan
    last_values: list[float] = [] # Nilai historis terakhir untuk model seperti LSTM

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "API Prediksi Harga Beras (FastAPI Contoh)"}

# --- Endpoint Prediksi (Simulasi ARIMA) ---
@app.post("/predict/{category_name}")
async def predict_arima(category_name: str, req: PredictRequest):
    logger.info(f"Menerima request ARIMA untuk {category_name} dengan {req.steps_ahead} langkah.")
    
    # Simulasikan delay jaringan/komputasi model
    time.sleep(1) 

    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    # Simulasi logika prediksi ARIMA
    # Di sini Anda akan memanggil model.forecast(steps=req.steps_ahead) yang sesungguhnya
    simulated_prediction = [round(random.uniform(11000, 13000), 2) for _ in range(req.steps_ahead)]

    return {
        "model": "ARIMA",
        "category": category_name,
        "prediction": simulated_prediction # Sesuai dengan `res.prediction` di frontend
    }

# --- Endpoint Prediksi (Simulasi LSTM) ---
@app.post("/predict_lstm/{category_name}")
async def predict_lstm(category_name: str, req: PredictRequest):
    logger.info(f"Menerima request LSTM untuk {category_name} dengan {req.steps_ahead} langkah dan {len(req.last_values)} last_values.")

    # Simulasikan delay jaringan/komputasi model
    time.sleep(1.5)

    if req.category.lower() != category_name.lower():
        raise HTTPException(status_code=400, detail="Category di URL dan body tidak cocok.")

    if not req.last_values or len(req.last_values) < 7:
        raise HTTPException(status_code=400, detail="Minimal 7 nilai terakhir diperlukan untuk prediksi LSTM (simulasi).")

    # Simulasi logika prediksi LSTM
    # Di sini Anda akan memanggil model.predict(input_seq) yang sesungguhnya
    simulated_prediction = [round(random.uniform(10500, 12500), 2) for _ in range(req.steps_ahead)]

    return {
        "model": "LSTM",
        "category": category_name,
        "prediction": simulated_prediction # Sesuai dengan `res.prediction` di frontend
    }