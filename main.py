from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)
CORS(app)

# Global variables untuk caching data
cached_data = None
cached_timestamp = None
CACHE_DURATION = 3600  # 1 hour in seconds

def load_historical_data():
    """Load dan cache data historis"""
    global cached_data, cached_timestamp
    
    current_time = datetime.now().timestamp()
    
    # Check if we need to reload cache
    if cached_data is None or (current_time - cached_timestamp) > CACHE_DURATION:
        try:
            with open('data_harga.json', 'r') as f:
                data = json.load(f)
            
            cached_data = data
            cached_timestamp = current_time
            print(f"Data loaded and cached: {len(data)} records")
            
        except FileNotFoundError:
            print("Warning: data_harga.json not found")
            cached_data = []
            cached_timestamp = current_time
    
    return cached_data

def prepare_lstm_data(data, column_name, sequence_length=10):
    """Prepare data for LSTM prediction"""
    if len(data) < sequence_length:
        raise ValueError(f"Insufficient data. Need at least {sequence_length} records")
    
    # Extract values for the specified column
    values = [float(record[column_name]) for record in data[-sequence_length:]]
    
    # Normalize data
    scaler = MinMaxScaler()
    values_normalized = scaler.fit_transform(np.array(values).reshape(-1, 1))
    
    return values_normalized.flatten(), scaler

def predict_with_arima(model_path, steps_ahead=1):
    """Make prediction using ARIMA model"""
    try:
        # Load ARIMA model
        arima_model = joblib.load(model_path)
        
        # Make prediction
        forecast = arima_model.forecast(steps=steps_ahead)
        
        # Convert to list if it's a single value
        if isinstance(forecast, (int, float)):
            forecast = [forecast]
        elif hasattr(forecast, 'tolist'):
            forecast = forecast.tolist()
        
        return forecast
        
    except Exception as e:
        print(f"Error in ARIMA prediction: {str(e)}")
        raise

def predict_with_lstm(model_path, data, column_name, steps_ahead=1, sequence_length=10):
    """Make prediction using LSTM model"""
    try:
        # Load LSTM model
        lstm_model = load_model(model_path)
        
        # Prepare data
        last_values, scaler = prepare_lstm_data(data, column_name, sequence_length)
        
        predictions = []
        current_sequence = last_values.copy()
        
        for _ in range(steps_ahead):
            # Reshape for prediction
            input_data = current_sequence.reshape(1, sequence_length, 1)
            
            # Make prediction
            prediction = lstm_model.predict(input_data, verbose=0)
            
            # Denormalize prediction
            prediction_denorm = scaler.inverse_transform(prediction)[0][0]
            predictions.append(float(prediction_denorm))
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = prediction[0][0]
        
        return predictions
        
    except Exception as e:
        print(f"Error in LSTM prediction: {str(e)}")
        raise

# API Routes for ARIMA models
@app.route('/predict/medium_silinda/arima', methods=['POST'])
def predict_medium_silinda_arima():
    try:
        data = request.get_json()
        steps_ahead = data.get('steps_ahead', 1)
        
        predictions = predict_with_arima('model/medium_silinda_arima.joblib', steps_ahead)
        
        return jsonify({
            'predictions': predictions,
            'model': 'ARIMA',
            'rice_type': 'medium_silinda',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/premium_silinda/arima', methods=['POST'])
def predict_premium_silinda_arima():
    try:
        data = request.get_json()
        steps_ahead = data.get('steps_ahead', 1)
        
        predictions = predict_with_arima('model/premium_silinda_arima.joblib', steps_ahead)
        
        return jsonify({
            'predictions': predictions,
            'model': 'ARIMA',
            'rice_type': 'premium_silinda',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/medium_bapanas/arima', methods=['POST'])
def predict_medium_bapanas_arima():
    try:
        data = request.get_json()
        steps_ahead = data.get('steps_ahead', 1)
        
        predictions = predict_with_arima('model/medium_bapanas_arima.joblib', steps_ahead)
        
        return jsonify({
            'predictions': predictions,
            'model': 'ARIMA',
            'rice_type': 'medium_bapanas',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/premium_bapanas/arima', methods=['POST'])
def predict_premium_bapanas_arima():
    try:
        data = request.get_json()
        steps_ahead = data.get('steps_ahead', 1)
        
        predictions = predict_with_arima('model/premium_bapanas_arima.joblib', steps_ahead)
        
        return jsonify({
            'predictions': predictions,
            'model': 'ARIMA',
            'rice_type': 'premium_bapanas',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API Routes for LSTM models
@app.route('/predict/medium_silinda/lstm', methods=['POST'])
def predict_medium_silinda_lstm():
    try:
        request_data = request.get_json()
        steps_ahead = request_data.get('steps_ahead', 1)
        
        # Load historical data
        historical_data = load_historical_data()
        
        predictions = predict_with_lstm(
            'model/medium_silinda_lstm.h5', 
            historical_data, 
            'medium_silinda', 
            steps_ahead
        )
        
        return jsonify({
            'predictions': predictions,
            'model': 'LSTM',
            'rice_type': 'medium_silinda',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/premium_silinda/lstm', methods=['POST'])
def predict_premium_silinda_lstm():
    try:
        request_data = request.get_json()
        steps_ahead = request_data.get('steps_ahead', 1)
        
        # Load historical data
        historical_data = load_historical_data()
        
        predictions = predict_with_lstm(
            'model/premium_silinda_lstm.h5', 
            historical_data, 
            'premium_silinda', 
            steps_ahead
        )
        
        return jsonify({
            'predictions': predictions,
            'model': 'LSTM',
            'rice_type': 'premium_silinda',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/medium_bapanas/lstm', methods=['POST'])
def predict_medium_bapanas_lstm():
    try:
        request_data = request.get_json()
        steps_ahead = request_data.get('steps_ahead', 1)
        
        # Load historical data
        historical_data = load_historical_data()
        
        predictions = predict_with_lstm(
            'model/medium_bapanas_lstm.h5', 
            historical_data, 
            'medium_bapanas', 
            steps_ahead
        )
        
        return jsonify({
            'predictions': predictions,
            'model': 'LSTM',
            'rice_type': 'medium_bapanas',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/premium_bapanas/lstm', methods=['POST'])
def predict_premium_bapanas_lstm():
    try:
        request_data = request.get_json()
        steps_ahead = request_data.get('steps_ahead', 1)
        
        # Load historical data
        historical_data = load_historical_data()
        
        predictions = predict_with_lstm(
            'model/premium_bapanas_lstm.h5', 
            historical_data, 
            'premium_bapanas', 
            steps_ahead
        )
        
        return jsonify({
            'predictions': predictions,
            'model': 'LSTM',
            'rice_type': 'premium_bapanas',
            'steps_ahead': steps_ahead
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cached_records': len(cached_data) if cached_data else 0
    })

# Get latest prices endpoint
@app.route('/latest-prices', methods=['GET'])
def get_latest_prices():
    try:
        historical_data = load_historical_data()
        
        if not historical_data:
            return jsonify({'error': 'No historical data available'}), 404
        
        latest_record = historical_data[-1]
        
        return jsonify({
            'date': latest_record['date'],
            'prices': {
                'medium_silinda': latest_record['medium_silinda'],
                'premium_silinda': latest_record['premium_silinda'],
                'medium_bapanas': latest_record['medium_bapanas'],
                'premium_bapanas': latest_record['premium_bapanas']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load data on startup
    load_historical_data()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)