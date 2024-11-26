# Import Required Libraries
import pickle
import pandas as pd
from flask import Flask, request, jsonify
import bentoml

# Load the Model
# Load model using pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Create a Flask App

# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Convert JSON to DataFrame
    input_data = pd.DataFrame([data])

    # Perform prediction
    prediction = model.predict(input_data)
    
    # Return the result as JSON
    return jsonify({'prediction': prediction[0]})

# Run the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
