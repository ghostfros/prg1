import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming the file is in CSV format)
# Replace 'Social_Network_Advertise_Data2.csv' with the correct file path
data = pd.read_csv('Social_Network_Ads2.csv')

# Display the first few rows of the dataset
print(data.head())

# Assuming the dataset has columns 'Age', 'EstimatedSalary' as features and 'Purchased' as the target
# Define independent variables (features) and dependent variable (target)
X = data[['Age', 'EstimatedSalary']]  # Features (independent variables)
y = data['Purchased']  # Target variable (dependent variable, 0 or 1 indicating purchase decision)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for Naive Bayes since it doesn't assume a particular scale for features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the Gaussian Naive Bayes model
model = GaussianNB()

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