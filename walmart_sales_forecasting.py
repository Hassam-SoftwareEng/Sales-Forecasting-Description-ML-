# walmart_sales_forecasting.py
# Sales Forecasting using Walmart Dataset
# Time series regression with Linear Regression, Random Forest, and XGBoost
# Fully updated with fixes, one-hot encoding, debug prints, and subset testing

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Try importing seasonal_decompose, skip if not installed
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    statsmodels_installed = True
except ImportError:
    statsmodels_installed = False
    print("statsmodels not installed. Seasonal decomposition will be skipped.")

# -------------------------------
# 1. Load Dataset
# -------------------------------
print("Step 1: Loading datasets...")

train_path = r'D:\Remote Internship\Task 3\train.csv'
features_path = r'D:\Remote Internship\Task 3\features.csv'
stores_path = r'D:\Remote Internship\Task 3\stores.csv'

# Check if files exist
for path in [train_path, features_path, stores_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please check dataset location.")

train = pd.read_csv(train_path, parse_dates=['Date'])
features = pd.read_csv(features_path, parse_dates=['Date'])
stores = pd.read_csv(stores_path)

print("Datasets loaded successfully!")
print("Train shape:", train.shape)
print("Features shape:", features.shape)
print("Stores shape:", stores.shape)

# Merge datasets
data = pd.merge(train, features, on=['Store', 'Date'])
data = pd.merge(data, stores, on='Store')

# Sort by date
data = data.sort_values(by='Date')

# -------------------------------
# 2. Feature Engineering
# -------------------------------
print("Step 2: Feature engineering...")

# Time-based features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Lag features
data['Lag_1'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
data['Lag_2'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)

# Rolling average (fixed using transform)
data['Rolling_4'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(4).mean())

# Fill missing values (use bfill)
data.fillna(method='bfill', inplace=True)

# Optional: Use only a subset for faster testing
data = data.head(2000)  # comment out to use full dataset

print("Feature engineering done. Data shape:", data.shape)

# -------------------------------
# 3. Prepare Data for Modeling
# -------------------------------
print("Step 3: Preparing data for modeling...")

features_list = ['Store', 'Dept', 'Year', 'Month', 'Week', 'DayOfWeek', 'Lag_1', 'Lag_2', 'Rolling_4', 
                 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size']

# Separate target
y = data['Weekly_Sales']

# One-hot encode categorical columns (Type)
X = pd.get_dummies(data[features_list], columns=['Type'], drop_first=True)

# Train-test split (time series aware)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Data split done. Training size:", X_train.shape[0], "Testing size:", X_test.shape[0])

# -------------------------------
# 4. Train Models
# -------------------------------
print("Step 4: Training models...")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost
xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)

print("Models trained successfully!")

# -------------------------------
# 5. Evaluate Models
# -------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("\nStep 5: Evaluating models...")
print("Linear Regression RMSE:", rmse(y_test, y_pred_lr))
print("Random Forest RMSE:", rmse(y_test, y_pred_rf))
print("XGBoost RMSE:", rmse(y_test, y_pred_xg))

# -------------------------------
# 6. Visualize Actual vs Predicted
# -------------------------------
print("Step 6: Plotting actual vs predicted...")

plt.figure(figsize=(12,6))
plt.plot(data['Date'].iloc[-len(y_test):], y_test, label='Actual')
plt.plot(data['Date'].iloc[-len(y_test):], y_pred_xg, label='Predicted (XGBoost)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Actual vs Predicted Weekly Sales')
plt.legend()
plt.show()

# -------------------------------
# 7. Seasonal Decomposition (Bonus)
# -------------------------------
if statsmodels_installed:
    print("Step 7: Performing seasonal decomposition...")
    result = seasonal_decompose(data['Weekly_Sales'], model='additive', period=52)
    result.plot()
    plt.show()
else:
    print("Skipping seasonal decomposition (statsmodels not installed).")

print("Script finished successfully!")
