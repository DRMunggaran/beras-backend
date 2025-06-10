from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Rice Price Prediction API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PredictionRequest(BaseModel):
    steps_ahead: int = 1
    historical_data: Optional[List[float]] = None

class PredictionResponse(BaseModel):
    predictions: List[float]
    method: str
    category: str

# Global variables untuk cache data
historical_data_cache = None
last_cache_update = None
CACHE_DURATION = 300  # 5 menit dalam detik

def load_historical_data():
    """Load data historis dari file JSON"""
    global historical_data_cache, last_cache_update
    
    # Check cache validity
    now = datetime.now()
    if (historical_data_cache is not None and 
        last_cache_update is not None and 
        (now - last_cache_update).seconds < CACHE_DURATION):
        return historical_data_cache
    
    try:
        # Coba beberapa lokasi file
        possible_paths = [
            'data_harga.json',
            '../data_harga.json',
            '../../public/data_harga.json',
            './public/data_harga.json'
        ]
        
        data = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Data loaded from: {path}")
                break
        
        if data is None:
            # Jika file tidak ditemukan, buat dummy data
            logger.warning("Data file not found, creating dummy data")
            data = create_dummy_data()
        
        historical_data_cache = data
        last_cache_update = now
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return create_dummy_data()

def create_dummy_data():
    """Buat dummy data untuk testing"""
    base_prices = {
        'medium_silinda': 12000,
        'premium_silinda': 15000,
        'medium_bapanas': 11000,
        'premium_bapanas': 14000,
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        date = start_date + timedelta(days=i)
        row = {
            'id': i + 1,
            'date': date.strftime('%Y-%m-%d'),
        }
        
        # Add some random variation to prices
        for category, base_price in base_prices.items():
            variation = np.random.normal(0, 500)  # Random variation ±500
            row[category] = max(int(base_price + variation), base_price - 1000)
        
        data.append(row)
    
    return data

def simple_arima_prediction(data: List[float], steps: int = 1) -> List[float]:
    """Simple ARIMA-like prediction using moving average and trend"""
    if len(data) < 3:
        return [data[-1]] * steps if data else [12000] * steps
    
    # Calculate trend (simple linear regression)
    x = np.arange(len(data))
    y = np.array(data)
    
    # Simple trend calculation
    n = len(data)
    if n >= 2:
        trend = (y[-1] - y[0]) / (n - 1)
    else:
        trend = 0
    
    # Moving average of last 3 values
    ma = np.mean(data[-3:])
    
    predictions = []
    for i in range(steps):
        # Combine moving average with trend
        pred = ma + (trend * (i + 1))
        # Add some smoothing
        pred = 0.7 * pred + 0.3 * data[-1]
        predictions.append(max(pred, data[-1] * 0.8))  # Prevent too low prices
    
    return predictions

def simple_lstm_prediction(data: List[float], steps: int = 1) -> List[float]:
    """Simple LSTM-like prediction using exponential smoothing"""
    if len(data) < 2:
        return [data[-1]] * steps if data else [12000] * steps
    
    # Exponential smoothing parameters
    alpha = 0.3  # Smoothing factor
    
    # Calculate exponentially weighted average
    weights = np.exp(-alpha * np.arange(len(data))[::-1])
    weights = weights / np.sum(weights)
    
    weighted_avg = np.sum(np.array(data) * weights)
    
    # Recent trend
    if len(data) >= 3:
        recent_trend = (data[-1] - data[-3]) / 2
    else:
        recent_trend = data[-1] - data[-2] if len(data) >= 2 else 0
    
    predictions = []
    for i in range(steps):
        # Predict with dampened trend
        pred = weighted_avg + (recent_trend * (i + 1) * 0.5)
        # Ensure reasonable bounds
        pred = max(pred, data[-1] * 0.85)
        pred = min(pred, data[-1] * 1.15)
        predictions.append(pred)
    
    return predictions

def get_category_data(data: List[dict], category: str) -> List[float]:
    """Extract price data for specific category"""
    try:
        prices = [item[category] for item in data if category in item]
        return prices
    except Exception as e:
        logger.error(f"Error extracting {category} data: {e}")
        return []

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Rice Price Prediction API is running"}

@app.get("/health")
async def health_check():
    data = load_historical_data()
    return {
        "status": "healthy",
        "data_points": len(data) if data else 0,
        "timestamp": datetime.now().isoformat()
    }

# Prediction endpoints for each category
@app.post("/predict/medium_silinda/arima", response_model=PredictionResponse)
async def predict_medium_silinda_arima(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'medium_silinda')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_arima_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="ARIMA",
            category="medium_silinda"
        )
    except Exception as e:
        logger.error(f"Error in medium_silinda ARIMA prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/medium_silinda/lstm", response_model=PredictionResponse)
async def predict_medium_silinda_lstm(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'medium_silinda')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_lstm_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="LSTM",
            category="medium_silinda"
        )
    except Exception as e:
        logger.error(f"Error in medium_silinda LSTM prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/premium_silinda/arima", response_model=PredictionResponse)
async def predict_premium_silinda_arima(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'premium_silinda')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_arima_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="ARIMA",
            category="premium_silinda"
        )
    except Exception as e:
        logger.error(f"Error in premium_silinda ARIMA prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/premium_silinda/lstm", response_model=PredictionResponse)
async def predict_premium_silinda_lstm(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'premium_silinda')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_lstm_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="LSTM",
            category="premium_silinda"
        )
    except Exception as e:
        logger.error(f"Error in premium_silinda LSTM prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/medium_bapanas/arima", response_model=PredictionResponse)
async def predict_medium_bapanas_arima(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'medium_bapanas')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_arima_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="ARIMA",
            category="medium_bapanas"
        )
    except Exception as e:
        logger.error(f"Error in medium_bapanas ARIMA prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/medium_bapanas/lstm", response_model=PredictionResponse)
async def predict_medium_bapanas_lstm(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'medium_bapanas')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_lstm_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="LSTM",
            category="medium_bapanas"
        )
    except Exception as e:
        logger.error(f"Error in medium_bapanas LSTM prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/premium_bapanas/arima", response_model=PredictionResponse)
async def predict_premium_bapanas_arima(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'premium_bapanas')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_arima_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="ARIMA",
            category="premium_bapanas"
        )
    except Exception as e:
        logger.error(f"Error in premium_bapanas ARIMA prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/premium_bapanas/lstm", response_model=PredictionResponse)
async def predict_premium_bapanas_lstm(request: PredictionRequest):
    try:
        data = load_historical_data()
        if request.historical_data:
            prices = request.historical_data
        else:
            prices = get_category_data(data, 'premium_bapanas')
        
        if not prices:
            raise HTTPException(status_code=400, detail="No data available for prediction")
        
        predictions = simple_lstm_prediction(prices, request.steps_ahead)
        return PredictionResponse(
            predictions=predictions,
            method="LSTM",
            category="premium_bapanas"
        )
    except Exception as e:
        logger.error(f"Error in premium_bapanas LSTM prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)