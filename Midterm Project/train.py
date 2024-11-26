import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import bentoml

# Load and Prepare the dataset
data = pd.read_csv('data/household_power_consumption.txt', delimiter=';', 
                   parse_dates=[['Date', 'Time']], infer_datetime_format=True, 
                   na_values='?', low_memory=False)

# Convert Date_Time to datetime and handle missing values
data['Date_Time'] = pd.to_datetime(data['Date_Time'])
data.fillna(data.median(), inplace=True)

# Feature engineering
data['Day_of_Week'] = data['Date_Time'].dt.dayofweek
data['Hour'] = data['Date_Time'].dt.hour

# Features and target
X = data[['Day_of_Week', 'Hour']]
y = data['Global_active_power']
 
# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save model to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved to model.pkl")