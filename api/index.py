from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

def lambda_handler(event, context):
    # Serverless function untuk predict
    try:
        # Parse input
        body = event.get('body', '{}')
        data = json.loads(body) if isinstance(body, str) else body
        
        model_name = event.get('pathParameters', {}).get('model')
        
        # Load model
        model_path = f'models/{model_name}.joblib'
        model = joblib.load(model_path)
        
        # Predict
        result = model.predict([data['input']])
        
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': result.tolist()})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }