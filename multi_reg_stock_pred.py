import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the stock data (assuming the file is in CSV format)
# Replace 'Stock_data.csv' with the correct file path
stock_data = pd.read_csv('Stock_data.csv')

# Define the independent variables (interest_rate, unemployment_rate) and dependent variable (stock_index_price)
X = stock_data[['Interest_rate', 'Unemployment_rate']]  # Independent variables
y = stock_data['Stock_index_price']  # Dependent variable

# Create the linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Define the input for prediction (interest rate = 3 and unemployment rate = 5.7)
interest_rate = 3
unemployment_rate = 5.7
predicted_stock_index_price = model.predict([[interest_rate, unemployment_rate]])

# Print the predicted stock index price
print(f"Predicted stock index price: {predicted_stock_index_price[0]}")