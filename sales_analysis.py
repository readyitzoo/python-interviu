# ProductID , 
# ProductName , 
# Category , 
# Sales , 
# DateSold

import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Using a function to preprocess the dataframes for shorter and clearer code
def load_and_preprocess_data(filepath):

    # Load the dataset
    df = pd.read_csv(filepath)

    # Rename columns if necessary
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'DateSold'}, inplace=True)
        # Convert 'DateSold' to datetime
        df['DateSold'] = pd.to_datetime(df['DateSold'], errors='coerce')

    if 'Item Code' in df.columns:
        df.rename(columns={'Item Code': 'ProductID'}, inplace=True)

    if 'Item Name' in df.columns:
        df.rename(columns={'Item Name': 'ProductName'}, inplace=True)

    # Drop rows with missing data to avoid introducing potential outliers from default values
    df.dropna(inplace=True)
    return df


def merge_data(df1, df2, merge_on='ProductID'):

    merged_df = pd.merge(df2, df1, on=merge_on, how='left')

    return merged_df


# I.  Data Loading & Preprocessing
# Raw file path (replace with your actual raw file path)
raw_file_path = r'D:\Documents\Mihai\Python_test'

# File paths
sales_data_filepath = 'sales_data.csv'
product_info_filepath = 'product_info.csv'
wholesale_prices_filepath = 'wholesale_prices.csv'  # Replace with the actual path

# Load and preprocess the product information and sales data
product_df = load_and_preprocess_data(product_info_filepath)
sales_df = load_and_preprocess_data(sales_data_filepath)
wholesale_prices_df = load_and_preprocess_data(wholesale_prices_filepath)

# II. Data Integration
merged_data = merge_data(product_df, sales_df)

# Merge sales data with wholesale prices, using a backward fill for dates
# Now we have a final Data Frame with all information needed
final_merged_df = pd.merge_asof(
    merged_data.sort_values('DateSold'),
    wholesale_prices_df.sort_values('DateSold'),
    on='DateSold',
    by='ProductID',
    direction='backward'
)

# Save the merged and processed dataset to a new CSV file
final_merged_df.to_csv(os.path.join(raw_file_path, 'merged_sales_data1.csv'), index=False)


# III. Data Analysis

# 1. Calculate the Sales for each transaction
final_merged_df['Sales'] = final_merged_df['Unit Selling Price (RMB/kg)'] * final_merged_df['Quantity Sold (kilo)']

# 2. Calculate the Total Cost for each transaction
final_merged_df['Total Cost'] = final_merged_df['Wholesale Price (RMB/kg)'] * final_merged_df['Quantity Sold (kilo)']

# 3. Calculate the Profit for each transaction
final_merged_df['Profit'] = final_merged_df['Sales'] - final_merged_df['Total Cost']

# 4. Calculate the total sales and profit for each product
total_sales_profit_per_product = final_merged_df.groupby('ProductID').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

total_sales_profit_per_product.rename(columns={'Sales': 'TotalSales', 'Profit': 'TotalProfit'}, inplace=True)

# 5. Identify the top 5 best-selling products
top_5_products = total_sales_profit_per_product.nlargest(5, 'TotalSales')['ProductID']

# 6. Add a boolean column for Top 5 Best Sellers
total_sales_profit_per_product['Top5BestSeller'] = total_sales_profit_per_product['ProductID'].isin(top_5_products)


# 10. Identify sales trends over time :: vara si iarna, dar si heat map cu pret--total sales !!
# a. Monthly Sales Trends
final_merged_df['MonthYear'] = final_merged_df['DateSold'].dt.to_period('M')
monthly_sales = final_merged_df.groupby('MonthYear')['Sales'].sum().reset_index()
print("Monthly Sales Trend:")
print(monthly_sales)

# b. Quarterly Sales Trends
final_merged_df['Quarter'] = final_merged_df['DateSold'].dt.to_period('Q')
quarterly_sales = final_merged_df.groupby('Quarter')['Sales'].sum().reset_index()
print("Quarterly Sales Trend:")
print(quarterly_sales)

# V. Automation & Reporting
# 7. Merge with product information to get ProductName, drop duplicates to have only one item of each in the table
final_report = pd.merge(total_sales_profit_per_product, final_merged_df[['ProductID', 'ProductName']].drop_duplicates(), on='ProductID', how='left')

# 8. Reorder columns to match the required report format
final_report = final_report[['ProductID', 'ProductName', 'TotalSales', 'TotalProfit', 'Top5BestSeller']]

# 9. Save the summary report to a CSV file
final_report.to_csv('sales_summary.csv', index=False)

print("Summary report saved as sales_summary.csv")

# VI. Predicting Future Sales using Linear Regression

# daily_sales_df = final_merged_df.groupby('DateSold').agg({'Sales': 'sum'}).reset_index()
#
# # Convert 'DateSold' to ordinal
# daily_sales_df['DateSold_ordinal'] = daily_sales_df['DateSold'].apply(lambda x: x.toordinal())
#
# print(daily_sales_df.head())
#
# daily_sales_df['SalesRolling7'] = daily_sales_df['Sales'].rolling(window=7).mean()
#
# # Drop NaNs resulting from the rolling calculation
# daily_sales_df.dropna(inplace=True)
#
# # Define features and target
# X = daily_sales_df[['SalesRolling7', 'DateSold_ordinal']]
# y = daily_sales_df['Sales']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initialize and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"Rolling Averages with Daily Sales - Mean Squared Error (MSE): {mse}")
# print(f"Rolling Averages with Daily Sales - R-squared (R^2) Score: {r2}")
#
# # Save the predictions along with the actual values to a new DataFrame
# predictions_df = X_test.copy()
# predictions_df['ActualSales'] = y_test
# predictions_df['PredictedSales'] = y_pred
#
# # Optionally, save to a CSV file
# predictions_df.to_csv('daily_sales_rolling_predictions.csv', index=False)
#
# print("Predictions saved as daily_sales_rolling_predictions.csv")

# Assuming final_merged_df is your DataFrame with a 'DateSold' and 'Sales' column

# Step 1: Data Preparation
final_merged_df['DateSold'] = pd.to_datetime(final_merged_df['DateSold'], format='%d-%m-%y', errors='coerce')
final_merged_df.dropna(subset=['DateSold'], inplace=True)

# Ensure that the data is sorted by date
final_merged_df = final_merged_df.sort_values(by='DateSold')

# Aggregate sales by day
daily_sales_df = final_merged_df.groupby('DateSold').agg({'Sales': 'sum'}).reset_index()

# Step 2: Feature Engineering
# Add features for the day of the week, month, and rolling averages
daily_sales_df['DayOfWeek'] = daily_sales_df['DateSold'].dt.dayofweek
daily_sales_df['Month'] = daily_sales_df['DateSold'].dt.month
daily_sales_df['Year'] = daily_sales_df['DateSold'].dt.year

# Calculate rolling averages (e.g., 7-day rolling average)
daily_sales_df['SalesRolling7'] = daily_sales_df['Sales'].rolling(window=7, min_periods=1).mean()

# Convert 'DateSold' to ordinal
daily_sales_df['DateSold_ordinal'] = daily_sales_df['DateSold'].apply(lambda x: x.toordinal())

# Drop any potential NaN values from rolling averages (although min_periods=1 should handle this)
daily_sales_df.dropna(inplace=True)

# Define features and target
X = daily_sales_df[['DateSold_ordinal', 'DayOfWeek', 'Month', 'Year', 'SalesRolling7']]
y = daily_sales_df['Sales']

# Step 3: Model Training with Cross-Validation
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model using cross-validation
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cross_val_scores}")
print(f"Mean Cross-Validation R² Score: {np.mean(cross_val_scores)}")

# Fit the model on the full training data
model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Mean Squared Error (MSE): {mse}")
print(f"Test Set R-squared (R²) Score: {r2}")

# Residual Analysis
residuals = y_test - y_pred
print(f"Residuals: \n{residuals.describe()}")

# Optional: Plot residuals
import matplotlib.pyplot as plt

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# Optional: Save the predictions to a CSV file
predictions_df = X_test.copy()
predictions_df['ActualSales'] = y_test
predictions_df['PredictedSales'] = y_pred
predictions_df.to_csv('enhanced_daily_sales_predictions.csv', index=False)

print("Predictions saved as enhanced_daily_sales_predictions.csv")
