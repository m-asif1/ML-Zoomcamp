### **Description of the Problem**
The goal of this project is to predict household power consumption based on historical data collected by smart meters in homes. We will use various machine learning techniques to process and predict the energy usage. The prediction can help optimize energy distribution, forecast demand, and promote energy efficiency in households.

**Set up the environment**    
Create a virtual environment:   
> python3 -m venv env    
source env/bin/activate   # On Windows use `env\Scripts\activate`

**Install the required dependencies**
> pip install -r requirements.txt


**Download the Dataset**:   
The dataset can be downloaded from the UCI repository:

- [Household Power Consumption Dataset on UCI](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  

Place the dataset file (e.g., `household_power_consumption`.csv) inside the data/ folder.

**Run the notebook**   
Open the notebook (notebook.ipynb) using Jupyter
> jupyter notebook notebook.ipynb

`notebook.ipynb`   
**Data Preparation and Cleaning**
>import pandas as pd

># Load data
>data = pd.read_csv('data/household_power_consumption.csv', delimiter=';', 
                   parse_dates=[['Date', 'Time']], infer_datetime_format=True, 
                   na_values='?', low_memory=False)

**Data Cleaning**    
- Convert `Date_Time` to a `datetime` object.    
- Handle missing values.
```
# Convert to datetime format
data['Date_Time'] = pd.to_datetime(data['Date_Time'])

# Handle missing values (replace with median or drop rows)
data.fillna(data.median(), inplace=True)
```

**Exploratory Data Analysis (EDA)**   
Visualize Power Consumption
```
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(data['Date_Time'], data['Global_active_power'])
plt.title("Global Active Power Over Time")
plt.xlabel("Time")
plt.ylabel("Global Active Power (kilowatts)")
plt.show()
```

**Feature Engineering**  
Extract additional features (e.g., day of the week, hour of the day).
```
data['Day_of_Week'] = data['Date_Time'].dt.dayofweek
data['Hour'] = data['Date_Time'].dt.hour
```

**Feature Importance Analysis**  
use techniques like correlation matrix or feature importance from models such as Random Forest to identify significant features.
```
import seaborn as sns
corr = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```
**Model Selection and Parameter Tuning**
- **Train-Test Split**  
Split the data into training and testing sets.
    ```
    from sklearn.model_selection import train_test_split

    X = data[['Day_of_Week', 'Hour']]  # Features
    y = data['Global_active_power']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
- **Model Selection**   
You can use models like Random Forest or Linear Regression for this task.
    ```
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ```
- **Parameter Tuning**  
 Use GridSearchCV for hyperparameter tuning.
    ```
    from sklearn.model_selection import GridSearchCV

    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    ```
**Model Evaluation**   
Evaluate the model performance using metrics like RMSE or MAE.
    
```
from sklearn.metrics import mean_absolute_error

y_pred = grid_search.best_estimator_.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
```

`requirements.txt`
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Run the `train.py` script:
> python train.py

After running the script:
- If you are using **pickle**, a `model.pkl` file will be created, storing your trained model.    
- If you are using **BentoML**, the model will be saved and registered, and you can check your local BentoML model store

### **predict.py**
1. Import Required Libraries
2. Load the Model
3. Create a Flask App
4. Run the Flask App
