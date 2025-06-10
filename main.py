from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import pickle
import joblib
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import model libraries
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(
    title="API Prediksi Harga Beras",
    description="API untuk prediksi harga beras menggunakan model ARIMA dan LSTM",
    version="1.0.0"
)

# CORS middleware untuk mengizinkan akses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dalam produksi, ganti dengan domain frontend Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class PredictionRequest(BaseModel):
    category: str
    steps_ahead: int

# Global variables untuk caching data dan model
cached_data = None
cached_models = {}
cached_scalers = {}
data_cache_timestamp = None
CACHE_DURATION = 3600  # 1 jam dalam detik

# Kategori yang didukung
SUPPORTED_CATEGORIES = [
    'medium_silinda', 'premium_silinda', 
    'medium_bapanas', 'premium_bapanas'
]

def load_historical_data():
    """Memuat data historis dari file JSON lokal"""
    global cached_data, data_cache_timestamp
    
    # Cek apakah data masih valid (cache belum expired)
    if (cached_data is not None and 
        data_cache_timestamp is not None and 
        (datetime.now() - data_cache_timestamp).seconds < CACHE_DURATION):
        return cached_data
    
    try:
        # Coba baca dari file lokal
        if os.path.exists('data_harga.json'):
            with open('data_harga.json', 'r') as f:
                data = json.load(f)
        else:
            # Jika file tidak ada, buat data dummy untuk testing
            data = generate_dummy_data()
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Cache data
        cached_data = df
        data_cache_timestamp = datetime.now()
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback ke data dummy
        return pd.DataFrame(generate_dummy_data())

def generate_dummy_data():
    """Generate data dummy untuk testing jika file tidak ada"""
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    data = []
    base_prices = {
        'medium_silinda': 12000,
        'premium_silinda': 13000,
        'medium_bapanas': 12500,
        'premium_bapanas': 13500
    }
    
    for i, date in enumerate(dates):
        # Simulasi fluktuasi harga dengan trend dan noise
        trend_factor = 1 + (i * 0.0001)  # Trend naik perlahan
        noise_factor = 1 + np.random.normal(0, 0.02)  # Noise ±2%
        
        data.append({
            'id': i + 1,
            'date': date,
            'medium_silinda': base_prices['medium_silinda'] * trend_factor * noise_factor,
            'premium_silinda': base_prices['premium_silinda'] * trend_factor * noise_factor,
            'medium_bapanas': base_prices['medium_bapanas'] * trend_factor * noise_factor,
            'premium_bapanas': base_prices['premium_bapanas'] * trend_factor * noise_factor
        })
    
    return data

def prepare_lstm_data(data_series, look_back=10):
    """Persiapkan data untuk model LSTM"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(look_back=10):
    """Buat model LSTM sederhana"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_or_load_arima_model(data_series, category):
    """Train atau load model ARIMA"""
    model_key = f"arima_{category}"
    
    if model_key in cached_models:
        return cached_models[model_key]
    
    try:
        # Coba berbagai parameter ARIMA
        best_aic = float('inf')
        best_model = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(data_series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            # Fallback ke model sederhana
            model = ARIMA(data_series, order=(1, 1, 1))
            best_model = model.fit()
        
        cached_models[model_key] = best_model
        return best_model
    
    except Exception as e:
        print(f"Error training ARIMA model for {category}: {e}")
        # Fallback ke model sederhana
        model = ARIMA(data_series, order=(1, 1, 1))
        fitted_model = model.fit()
        cached_models[model_key] = fitted_model
        return fitted_model

def train_or_load_lstm_model(data_series, category):
    """Train atau load model LSTM"""
    model_key = f"lstm_{category}"
    scaler_key = f"scaler_{category}"
    
    if model_key in cached_models and scaler_key in cached_scalers:
        return cached_models[model_key], cached_scalers[scaler_key]
    
    try:
        look_back = 10
        X, y, scaler = prepare_lstm_data(data_series, look_back)
        
        if len(X) < 20:  # Tidak cukup data untuk training
            raise ValueError("Insufficient data for LSTM training")
        
        # Reshape untuk LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Buat dan train model
        model = create_lstm_model(look_back)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        cached_models[model_key] = model
        cached_scalers[scaler_key] = scaler
        
        return model, scaler
    
    except Exception as e:
        print(f"Error training LSTM model for {category}: {e}")
        # Fallback ke model dummy yang mengembalikan nilai rata-rata
        class DummyLSTMModel:
            def __init__(self, mean_value):
                self.mean_value = mean_value
            
            def predict(self, X):
                return np.array([[self.mean_value]] * len(X))
        
        class DummyScaler:
            def __init__(self, data_series):
                self.mean = data_series.mean()
                self.std = data_series.std()
            
            def inverse_transform(self, data):
                return data * self.std + self.mean
            
            def transform(self, data):
                return (data - self.mean) / self.std
        
        dummy_model = DummyLSTMModel(data_series.mean())
        dummy_scaler = DummyScaler(data_series)
        
        cached_models[model_key] = dummy_model
        cached_scalers[scaler_key] = dummy_scaler
        
        return dummy_model, dummy_scaler

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "API Prediksi Harga Beras",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "supported_categories": SUPPORTED_CATEGORIES
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        df = load_historical_data()
        return {
            "status": "healthy",
            "data_records": len(df),
            "latest_date": df['date'].max().isoformat() if not df.empty else None,
            "categories": SUPPORTED_CATEGORIES
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict/{category}")
async def predict_arima(category: str, request: PredictionRequest):
    """Prediksi menggunakan model ARIMA"""
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=400, 
            detail=f"Category tidak didukung. Pilih dari: {SUPPORTED_CATEGORIES}"
        )
    
    try:
        # Load data
        df = load_historical_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data historis tidak tersedia")
        
        # Ambil data untuk kategori yang diminta
        data_series = df[category].dropna()
        if len(data_series) < 10:
            raise HTTPException(status_code=400, detail="Data tidak cukup untuk prediksi")
        
        # Train atau load model
        model = train_or_load_arima_model(data_series, category)
        
        # Lakukan prediksi
        forecast = model.forecast(steps=request.steps_ahead)
        
        # Pastikan hasil prediksi masuk akal (tidak negatif, tidak terlalu ekstrem)
        forecast = np.maximum(forecast, data_series.mean() * 0.5)  # Minimum 50% dari rata-rata
        forecast = np.minimum(forecast, data_series.mean() * 2.0)  # Maximum 200% dari rata-rata
        
        return {
            "category": category,
            "model": "ARIMA",
            "steps_ahead": request.steps_ahead,
            "prediction": forecast.tolist(),
            "last_actual_price": float(data_series.iloc[-1]),
            "prediction_date": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error in ARIMA prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error dalam prediksi: {str(e)}")

@app.post("/predict_lstm/{category}")
async def predict_lstm(category: str, request: PredictionRequest):
    """Prediksi menggunakan model LSTM"""
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=400, 
            detail=f"Category tidak didukung. Pilih dari: {SUPPORTED_CATEGORIES}"
        )
    
    try:
        # Load data
        df = load_historical_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data historis tidak tersedia")
        
        # Ambil data untuk kategori yang diminta
        data_series = df[category].dropna()
        if len(data_series) < 20:
            raise HTTPException(status_code=400, detail="Data tidak cukup untuk prediksi LSTM")
        
        # Train atau load model
        model, scaler = train_or_load_lstm_model(data_series, category)
        
        # Persiapkan data untuk prediksi
        look_back = 10
        last_values = data_series.tail(look_back).values
        scaled_last_values = scaler.transform(last_values.reshape(-1, 1))
        
        predictions = []
        current_input = scaled_last_values.flatten()
        
        # Lakukan prediksi iteratif
        for _ in range(request.steps_ahead):
            # Reshape untuk input LSTM
            input_data = current_input[-look_back:].reshape(1, look_back, 1)
            
            # Prediksi step berikutnya
            next_pred = model.predict(input_data, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update input untuk prediksi berikutnya
            current_input = np.append(current_input, next_pred[0, 0])
        
        # Inverse transform untuk mendapatkan nilai asli
        predictions_array = np.array(predictions).reshape(-1, 1)
        actual_predictions = scaler.inverse_transform(predictions_array).flatten()
        
        # Pastikan hasil prediksi masuk akal
        actual_predictions = np.maximum(actual_predictions, data_series.mean() * 0.5)
        actual_predictions = np.minimum(actual_predictions, data_series.mean() * 2.0)
        
        return {
            "category": category,
            "model": "LSTM",
            "steps_ahead": request.steps_ahead,
            "prediction": actual_predictions.tolist(),
            "last_actual_price": float(data_series.iloc[-1]),
            "prediction_date": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error in LSTM prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error dalam prediksi: {str(e)}")

@app.get("/data/latest")
async def get_latest_data():
    """Ambil data harga terbaru"""
    try:
        df = load_historical_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data tidak tersedia")
        
        latest_data = df.tail(1).iloc[0]
        return {
            "date": latest_data['date'].isoformat(),
            "prices": {
                "medium_silinda": float(latest_data['medium_silinda']),
                "premium_silinda": float(latest_data['premium_silinda']),
                "medium_bapanas": float(latest_data['medium_bapanas']),
                "premium_bapanas": float(latest_data['premium_bapanas'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mengambil data: {str(e)}")

@app.get("/data/history")
async def get_historical_data(days: Optional[int] = 30):
    """Ambil data historis dalam jumlah hari tertentu"""
    try:
        df = load_historical_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data tidak tersedia")
        
        # Ambil data beberapa hari terakhir
        recent_data = df.tail(days)
        
        return {
            "data": recent_data.to_dict('records'),
            "total_records": len(recent_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mengambil data historis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)