## **Household Power Consumption Prediction**

### **Description of the Problem**
The goal of this project is to predict power consumption based on historical household power usage data. Using the [Household Power Consumption Dataset on UCI](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) , the project explores how features like global active power, voltage, and other electricity-related metrics can be used to forecast future consumption patterns. This information can be valuable for energy management, optimizing electricity usage, and reducing energy costs.

The dataset contains over 2 million observations collected between December 2006 and November 2010, including:

- `Date` and `Time`: Timestamp of data recording.   
- `Global_active_power`: The total active power consumed by the household.
- `Voltage`: The average voltage in the household.  
- `Global_intensity`: The total current intensity.
- Other relevant power metrics.

By building a machine learning model, we aim to predict household power consumption, which can help utilities and consumers optimize energy usage and planning.


## **Project Overview**
- **Data preprocessing**: Cleaning the data and handling missing values.
- **Exploratory Data Analysis (EDA)**: Understanding the dataset and feature importance.
- **Model training**: Selecting the best model and tuning hyperparameters.
- **Model deployment**: Deploying a prediction service using a Flask API and containerizing it with Docker.

## **Instructions on How to Run the Project**

**Clone the Repository**  
First, clone the repository to your local machine:
```
git clone https://github.com/your-username/ML-Zoomcamp-Project.git
cd ML-Zoomcamp-Project
```
**Set up the environment**  
You have several options to set up the environment. Choose one:

**Using pip**
1. Make sure you have Python 3.8+ installed.
2. Install the required dependencies.
    > pip install -r requirements.txt    
    > python3 -m venv env    
    source env/bin/activate       # On Windows use `env\Scripts\activate`

**Using Pipenv**
1. Make sure you have Pipenv installed.
    > pip install pipenv
2. Create a virtual environment and install dependencies.
    > pipenv install  
    > pipenv shell

**Using Conda**
1. Make sure you have Conda installed.
2. Create an environment using the provided `environment.yml`
    > conda env create -f environment.yml
3. Activate the environment
    > conda activate ml-zoomcamp-env

**Download the Dataset**:   
Download the dataset from the UCI Machine Learning Repository

- [Household Power Consumption Dataset on UCI](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
- Place the dataset file (e.g., `household_power_consumption`) inside the data/ folder.
  
**Data Preparation and Cleaning**

1. Open `notebook.ipynb` to explore data cleaning, EDA, and model training.
2. Alternatively, train the model directly using `train.py`
    > python train.py
    
    This will generate a `model.pkl` file containing the trained model.

### **Run the Prediction Service**
1. Start the Flask prediction service
    > python predict.py
2. By default, the service will run at `http://localhost:5000`. You can use Postman or `curl` to interact with the service.
    > curl -X POST -H "Content-Type: application/json" -d '{"input": [0.418, 234.840, 18.400, 0.000, 1.000, 17.000]}' http://localhost:5000/predict

### **Dockerize the Project (Optional)**
If you prefer to run the service using Docker   

1. Build the Docker image 
    > docker build -t household-power-service.
2. Run the Docker container:
    > docker run -d -p 5000:5000 household-power-service
3. Access the prediction service at `http://localhost:5000`.

### **Deployment (Optional)**
Deploy the Dockerized service to your preferred cloud platform (e.g., Heroku, AWS, GCP). Follow the deployment instructions specific to the platform you choose.

### **Dependencies**

Dependencies for this project are managed via `requirements.txt`. To install them, use the following command   
> pip install -r requirements.txt


`requirements.txt`
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

