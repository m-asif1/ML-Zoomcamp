import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the vectorizer and model
with open('dv.bin', 'rb') as f_dv, open('model1.bin', 'rb') as f_model:
    dv = pickle.load(f_dv)
    model = pickle.load(f_model)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.json
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    return jsonify({'probability': y_pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
