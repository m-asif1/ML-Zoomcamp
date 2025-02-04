## **Solar Energy Prediction Service**

### 📌 Description of the Problem
This project aims to predicts **AC power generation** based on deploy a machine learning model for predicting solar energy output based on weather and environmental data. The goal is to build a machine learning model to forecast power output, helping optimize energy management. The trained model is served using a Flask web service and deployed locally via Docker. Users can send requests to the API to obtain predictions based on input features.

We use data from two CSV files:
- `Plant_2_Generation_Data.csv`: Contains power generation details (DC/AC Power, Daily & Total Yield).
- `Plant_2_Weather_Sensor_Data.csv`: Includes weather parameters (Ambient Temperature, Module Temperature, and Irradiation).

The project involves:   
✅ **Data Cleaning & Feature Engineering** (Handling missing values, creating lag features, etc.)     
✅ **Exploratory Data Analysis (EDA)** (Visualizing trends & correlations)         
✅ **Model Training & Hyperparameter Tuning** (Linear Regression, Random Forest, Gradient Boosting, XGBoost)     
✅ **Deploying as a Web Service** (Using Flask & Docker)

### Project Structure
📂 Solar-Prediction-Service          
│── 📄 predict.py          # Flask app for serving predictions     
│── 📄 notebook.ipynb      # jupyter configuration file      
│── 📄 Dockerfile          # Docker configuration file     
│── 📄 requirements.txt    # List of dependencies         
│── 📄 final_model.pkl     # Trained machine learning model        
│── 📄 README.md           # Project documentation


## Installation and Usage
### 1. Prerequisites

Ensure you have the following installed:
- Python 3.10 or higher (if running locally)
- Docker (for containerized deployment)

### 2.  How to Run the Project

#### a) Running Locally (Without Docker)

#### 1. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
#### 2. Install dependencies:
```
pip install -r requirements.txt
```
### 3. Train the Model
Run `train.py` to preprocess the data, train the model, and save it:
```
python train.py
```
This will generate a file `final_model.pkl` that stores the trained model.
### 4. Run the Flask Web service:
Start the Flask server using:
```
python predict.py
```
The API will be available at:
👉 http://127.0.0.1:5000/predict

### 5. Make Predictions
You can send a POST request with JSON input like:
```
{
    "DC_POWER": 1200.5,
    "IRRADIATION": 0.25,
    "AMBIENT_TEMPERATURE": 30.5,
    "MODULE_TEMPERATURE": 35.2,
    "hour": 14,
    "day_of_week": 3,
    "month": 6
}
```
Use Postman or curl to test:
```
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{
    "DC_POWER": 1200.5,
    "IRRADIATION": 0.25,
    "AMBIENT_TEMPERATURE": 30.5,
    "MODULE_TEMPERATURE": 35.2,
    "hour": 14,
    "day_of_week": 3,
    "month": 6
}'
```

## **🚢 Deploy with Docker**
### 1. Build the Docker Image
```
docker build -t solar-service .
```
### 2. Run the Docker Container
```
docker run -p 5000:5000 solar-service
```
The service will be available at:
👉 http://localhost:5000/predict

## **☁️ Deploying to the Cloud**

### **Deploying on AWS EC2**

1. Launch an EC2 instance with Ubuntu.
2.  Install Docker.

```
sudo apt update && sudo apt install docker.io -y
```
3. Copy files to the server & build the Docker image

```
scp -i your-key.pem -r /your-project-directory ubuntu@your-ec2-ip:~
ssh -i your-key.pem ubuntu@your-ec2-ip
cd your-project-directory
docker build -t solar-service .
docker run -p 80:5000 solar-service

```
4. Now, your service will be accessible via your EC2 Public IP:
```
👉 http://your-ec2-ip/predict 
```

#### 🚀 Live Deployment
```
🌍 Live API URL: http://your-cloud-url/predict  
```
### 🏆 Results & Best Model

- **Best Model:** `XGBoost`
- **Test MAE:** `X.XX`
- **Test MSE:** `X.XX`
- **Feature Importance:** `IRRADIATION, DC_POWER, MODULE_TEMPERATURE`
