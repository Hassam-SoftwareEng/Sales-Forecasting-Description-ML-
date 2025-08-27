# Walmart Sales Forecasting

This project demonstrates **time series sales forecasting** using the Walmart dataset. The goal is to predict weekly sales for different stores and departments based on historical sales data and external factors. Multiple regression models are used, including **Linear Regression**, **Random Forest**, and **XGBoost**.

---

## Dataset

The project uses the **Walmart Store Sales Forecasting dataset**, which consists of three CSV files:

1. `train.csv` – Historical weekly sales for each store and department.
2. `features.csv` – Store-level features like temperature, fuel price, CPI, unemployment, and holiday indicators.
3. `stores.csv` – Store information including store type and size.

> Dataset source: [Kaggle Walmart Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)

---

## Features

The following features are engineered and used for modeling:

- **Time-based features:** Year, Month, Week, DayOfWeek  
- **Lag features:** Previous week sales (`Lag_1`, `Lag_2`)  
- **Rolling average:** 4-week rolling average of sales (`Rolling_4`)  
- **External factors:** Temperature, Fuel_Price, CPI, Unemployment  
- **Store information:** Type (categorical), Size  

Categorical variables are one-hot encoded before training.

---

## Models Used

- **Linear Regression** – baseline regression model  
- **Random Forest Regressor** – ensemble model for non-linear relationships  
- **XGBoost Regressor** – gradient boosting model for improved accuracy  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/walmart-sales-forecasting.git
cd walmart-sales-forecasting
Install required libraries:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
statsmodels is optional but required for seasonal decomposition plots.

Usage
Place the datasets (train.csv, features.csv, stores.csv) in the project folder.

Update file paths in walmart_sales_forecasting.py if necessary.

Run the script:

bash
Copy code
python walmart_sales_forecasting.py
The script will:

Load and merge datasets

Perform feature engineering

Split data into training and testing sets

Train Linear Regression, Random Forest, and XGBoost models

Evaluate models using RMSE

Plot Actual vs Predicted weekly sales

Optionally perform seasonal decomposition (if statsmodels installed)

Output
RMSE for each model will be printed to the console.

Line plot showing Actual vs Predicted sales.

Seasonal decomposition plots (optional) showing trend, seasonality, and residuals.

Notes
For faster testing, the script currently uses a subset of 2000 rows. To use the full dataset, comment out the data.head(2000) line.

Ensure categorical columns (Type) are properly one-hot encoded to avoid errors during model training.

License
This project is open-source and available under the MIT License.

Acknowledgments
Kaggle Walmart Store Sales Forecasting Competition for providing the dataset.

Python libraries: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Statsmodels.
