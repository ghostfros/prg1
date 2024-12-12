import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the bank data (assuming the file is in CSV format)
# Replace 'Bank_data_set_1.csv' with the correct file path
bank_data = pd.read_csv('Bank_data_set_1.csv')

# Data preprocessing
# Assuming the target column is 'subscribed' and features are columns like 'age', 'balance', 'duration', etc.
# You need to adjust based on the actual dataset structure
# Convert categorical data to dummy/indicator variables
bank_data = pd.get_dummies(bank_data, drop_first=True)

# Define the independent variables (features) and dependent variable (target: subscribed)
X = bank_data.drop('subscribed', axis=1)  # Features
y = bank_data['subscribed']  # Target variable

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the accuracy and classification report
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)