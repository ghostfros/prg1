import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the car data (assuming the file is in CSV format)
# Replace 'Car_data.csv' with the correct file path
car_data = pd.read_csv('Car_data.csv')

# Define the independent variables (Volume, Engine weight) and dependent variable (CO2 emission level)
X = car_data[['Volume', 'Weight']]  # Independent variables: Volume and Engine weight
y = car_data['CO2']  # Dependent variable: CO2 emission level

# Create the linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Define the input for prediction (Volume = 1300 and Engine weight = 3300 Kg)
volume = 1300
engine_weight = 3300
predicted_CO2 = model.predict([[volume, engine_weight]])

# Print the predicted CO2 emission level
print(f"Predicted CO2 emission level: {predicted_CO2[0]} g/km")