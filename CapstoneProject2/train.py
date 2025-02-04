#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_prepare_data():
    # --- Load the generation data ---
    gen_df = pd.read_csv('Plant_2_Generation_Data.csv', sep=',')
    print("Columns in generation data:", gen_df.columns.tolist())
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
    # Drop SOURCE_KEY and keep only required columns
    gen_df = gen_df[['DATE_TIME', 'PLANT_ID', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']]

    # --- Load the weather sensor data ---
    weather_df = pd.read_csv('Plant_2_Weather_Sensor_Data.csv', sep=',')
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
    weather_df = weather_df[['DATE_TIME', 'PLANT_ID', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]

    # --- Merge the two datasets ---
    merged_df = pd.merge_asof(
        gen_df.sort_values('DATE_TIME'),
        weather_df.sort_values('DATE_TIME'),
        on='DATE_TIME',
        by='PLANT_ID',
        direction='nearest',
        tolerance=pd.Timedelta('15min')
    )

    print("Merged DataFrame head:")
    print(merged_df.head())
    print(merged_df.info())

    # Ensure DATE_TIME is datetime and set as the index
    merged_df['DATE_TIME'] = pd.to_datetime(merged_df['DATE_TIME'])
    merged_df.set_index('DATE_TIME', inplace=True)

    # --- Data Cleaning ---
    # Interpolate missing values using time-based interpolation, then back-fill if needed.
    merged_df.interpolate(method='time', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)

    # --- Feature Engineering ---
    # Create time-based features: hour, day_of_week, and month.
    merged_df['hour'] = merged_df.index.hour
    merged_df['day_of_week'] = merged_df.index.dayofweek
    merged_df['month'] = merged_df.index.month

    # Create a lag feature for AC_POWER (previous observation)
    merged_df['AC_POWER_lag1'] = merged_df['AC_POWER'].shift(1)

    # Drop rows with missing values (created by the lag)
    merged_df.dropna(inplace=True)

    return merged_df


def train_models(merged_df):
    # --- Define Features and Target ---
    features = ['DC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 
                'MODULE_TEMPERATURE', 'hour', 'day_of_week', 'month', 'AC_POWER_lag1']
    target = 'AC_POWER'
    X = merged_df[features]
    y = merged_df[target]

    # --- Split the Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Define Hyperparameter Grids for Models ---
    param_grid_lr = {}  # LinearRegression does not require tuning
    param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    param_grid_gb = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
    param_grid_xgb = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}

    # --- Define Candidate Models ---
    models = {
        "Linear Regression": (LinearRegression(), param_grid_lr),
        "Random Forest": (RandomForestRegressor(random_state=42), param_grid_rf),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=42), param_grid_gb),
        "XGBoost": (xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid_xgb)
    }

    results = {}
    best_model = None
    best_mae = float("inf")
    best_model_name = None

    # --- Loop Through Models and Tune ---
    for name, (model, grid) in models.items():
        print("Tuning model:", name)
        if grid:  # Use GridSearchCV if hyperparameter grid is provided
            grid_search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = -grid_search.best_score_
            print(f"Best params for {name}: {best_params}")
        else:
            # For models with no hyperparameter tuning (e.g., Linear Regression)
            model.fit(X_train, y_train)
            best_estimator = model
            best_params = {}
            preds = best_estimator.predict(X_train)
            cv_score = mean_absolute_error(y_train, preds)
            print(f"{name} training MAE: {cv_score:.2f}")

        # Evaluate on the test set
        y_pred = best_estimator.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)

        results[name] = {
            "best_estimator": best_estimator,
            "best_params": best_params,
            "cv_score": cv_score,
            "test_mae": test_mae,
            "test_mse": test_mse
        }
        print(f"{name} Test MAE: {test_mae:.2f}, Test MSE: {test_mse:.2f}\n")

        # Track the best performing model
        if test_mae < best_mae:
            best_mae = test_mae
            best_model = best_estimator
            best_model_name = name

    # --- Summarize Model Performances ---
    results_df = pd.DataFrame({name: {"CV MAE": np.round(info["cv_score"], 2),
                                      "Test MAE": np.round(info["test_mae"], 2),
                                      "Test MSE": np.round(info["test_mse"], 2)}
                                for name, info in results.items()}).T
    print("----- Model Performance Summary -----")
    print(results_df)
    print(f"\nBest model based on Test MAE: {best_model_name} with a Test MAE of {best_mae:.2f}")

    # --- Plot Predictions vs Actual for Best Model ---
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.index, y_test, label='Actual AC_POWER', lw=2)
    plt.plot(y_test.index, best_model.predict(X_test), label='Predicted AC_POWER', lw=2, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('AC Power')
    plt.title(f"Actual vs. Predicted AC Power Generation ({best_model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig('best_model_predictions.png')
    plt.close()

    return best_model


def save_model(model, filename='final_model.pkl'):
    # Save the final best model to a file using pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


if __name__ == '__main__':
    # Load and prepare data
    merged_df = load_and_prepare_data()

    # Train multiple models and select the best one
    best_model = train_models(merged_df)

    # Save the final trained model to a file
    save_model(best_model)
