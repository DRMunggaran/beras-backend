from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# Load ARIMA models
arima_models = {
    "medium_silinda": load("model/medium_silinda_arima.joblib"),
    "premium_silinda": load("model/premium_silinda_arima.joblib"),
    "medium_bapanas": load("model/medium_bapanas_arima.joblib"),
    "premium_bapanas": load("model/premium_bapanas_arima.joblib")
}

# Load LSTM models
lstm_models = {
    "medium_silinda": tf.keras.models.load_model("model/medium_silinda_lstm.h5"),
    "premium_silinda": tf.keras.models.load_model("model/premium_silinda_lstm.h5"),
    "medium_bapanas": tf.keras.models.load_model("model/medium_bapanas_lstm.h5"),
    "premium_bapanas": tf.keras.models.load_model("model/premium_bapanas_lstm.h5")
}

class PredictRequest(BaseModel):
    model_type: str  # "arima" atau "lstm"
    category: str    # "medium_silinda", "premium_silinda", etc
    steps: int = 1   # jumlah hari prediksi ke depan
    last_values: list[float] = []  # hanya untuk LSTM

@app.get("/")
def root():
    return {"message": "API Prediksi Harga Beras (ARIMA & LSTM)"}

@app.post("/predict")
def predict(req: PredictRequest):
    model_key = req.category.lower()

    if req.model_type == "arima":
        model = arima_models.get(model_key)
        if not model:
            return {"error": f"Model ARIMA untuk kategori '{model_key}' tidak ditemukan"}
        forecast = model.forecast(steps=req.steps)
        return {
            "model": "ARIMA",
            "category": model_key,
            "predicted_prices": forecast.tolist()
        }

    elif req.model_type == "lstm":
        model = lstm_models.get(model_key)
        if not model:
            return {"error": f"Model LSTM untuk kategori '{model_key}' tidak ditemukan"}
        if not req.last_values or len(req.last_values) < 7:
            return {"error": "Minimal 7 nilai terakhir diperlukan untuk prediksi LSTM"}

        # Siapkan input shape: (1, time_steps, features)
        input_seq = np.array(req.last_values[-7:]).reshape((1, 7, 1))
        preds = []
        for _ in range(req.steps):
            pred = model.predict(input_seq)[0][0]
            preds.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        return {
            "model": "LSTM",
            "category": model_key,
            "predicted_prices": preds
        }

    else:
        return {"error": "Model type harus 'arima' atau 'lstm'"}
