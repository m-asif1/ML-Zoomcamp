import requests

url = "http://localhost:5000/predict"  # Adjust this if deploying elsewhere
client = {"job": "student", "duration": 280, "poutcome": "failure"}
response = requests.post(url, json=client)
probability = response.json()['probability']

print("Probability of subscription:", probability)
