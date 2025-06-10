from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from tensorflow import keras
import pandas as pd

app = Flask(__name__)

# Load models (modify sesuai nama file Anda)
models = {
    'medium_bapanas_arima': None,
    'medium_bapanas_lstm': None,
    'medium_silinda_arima': None,
    'medium_silinda_lstm': None,
    'premium_bapanas_arima': None,
    'premium_bapanas_lstm': None,
    'premium_silinda_arima': None,
    'premium_silinda_lstm': None,
}

def load_models():
    """Load all models at startup"""
    models_dir = 'models'
    
    # Load ARIMA models (joblib)
    arima_models = ['medium_bapanas_arima', 'medium_silinda_arima', 
                   'premium_bapanas_arima', 'premium_silinda_arima']
    
    for model_name in arima_models:
        try:
            file_path = os.path.join(models_dir, f"{model_name}.joblib")
            if os.path.exists(file_path):
                models[model_name] = joblib.load(file_path)
                print(f"Loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    # Load LSTM models (h5)
    lstm_models = ['medium_bapanas_lstm', 'medium_silinda_lstm',
                  'premium_bapanas_lstm', 'premium_silinda_lstm']
    
    for model_name in lstm_models:
        try:
            file_path = os.path.join(models_dir, f"{model_name}.h5")
            if os.path.exists(file_path):
                models[model_name] = keras.models.load_model(file_path)
                print(f"Loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running", "models_loaded": len([m for m in models.values() if m is not None])})

@app.route('/predict/<model_type>/<product_type>', methods=['POST'])
def predict(model_type, product_type):
    """
    Predict using specified model
    model_type: arima or lstm
    product_type: medium_bapanas, medium_silinda, premium_bapanas, premium_silinda
    """
    try:
        data = request.get_json()
        
        # Construct model name
        model_name = f"{product_type}_{model_type}"
        
        if model_name not in models or models[model_name] is None:
            return jsonify({"error": f"Model {model_name} not found or not loaded"}), 404
        
        model = models[model_name]
        
        if model_type == 'arima':
            # ARIMA prediction
            steps = data.get('steps', 5)  # default 5 steps
            forecast = model.forecast(steps=steps)
            
            return jsonify({
                "model": model_name,
                "prediction": forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)
            })
            
        elif model_type == 'lstm':
            # LSTM prediction
            input_data = np.array(data.get('input_data', []))
            
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1, 1)
            elif len(input_data.shape) == 2:
                input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
            
            prediction = model.predict(input_data)
            
            return jsonify({
                "model": model_name,
                "prediction": prediction.tolist()
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    available_models = [name for name, model in models.items() if model is not None]
    return jsonify({"available_models": available_models})

if __name__ == '__main__':
    # Load models at startup
    load_models()
    
    # Get port from environment variable (Fly.io sets this)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)