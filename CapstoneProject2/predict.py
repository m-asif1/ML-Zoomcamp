#!/usr/bin/env python3
"""
predict.py

This script loads the final trained model from 'final_model.pkl' and
serves a prediction endpoint via a Flask web service.

Example usage:
    Start the server:
        python predict.py

    Send a POST request to the /predict endpoint with a JSON payload:
    {
      "data": [
          {
              "DC_POWER": 1176.39,
              "IRRADIATION": 0.122339,
              "AMBIENT_TEMPERATURE": 24.305631,
              "MODULE_TEMPERATURE": 26.727956,
              "hour": 7,
              "day_of_week": 4,
              "month": 6,
              "AC_POWER_lag1": 1148.31
          }
      ]
    }

The response will be a JSON object containing the predicted AC_POWER values.
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from file
MODEL_FILE = 'final_model.pkl'
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_FILE}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the expected features in the correct order
FEATURES = [
    'DC_POWER', 
    'IRRADIATION', 
    'AMBIENT_TEMPERATURE', 
    'MODULE_TEMPERATURE', 
    'hour', 
    'day_of_week', 
    'month', 
    'AC_POWER_lag1'
]

@app.route('/')
def home():
    return "Welcome to the Solar Power Generation Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Validate that the key "data" is present
    if 'data' not in data:
        return jsonify({"error": "Invalid input. Expected JSON key 'data'."}), 400
    
    input_data = data['data']
    
    # Ensure the input data is in list format
    if not isinstance(input_data, list):
        return jsonify({"error": "Input 'data' should be a list of records."}), 400
    
    try:
        # Convert the list of dictionaries into a DataFrame
        df_input = pd.DataFrame(input_data)
        # Ensure that DataFrame contains all required features in the correct order
        df_input = df_input[FEATURES]
    except Exception as e:
        return jsonify({"error": f"Error processing input data: {e}"}), 400

    # Make predictions using the loaded model
    try:
        predictions = model.predict(df_input)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

    # Return the predictions as a JSON response
    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    # Run the Flask app on port 5000 by default.
    app.run(debug=True)
