# Required imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample data: Income, Savings, Loan Granted (1 = Yes, 0 = No)
data = {
    'Income': [40000, 50000, 60000, 30000, 100000, 20000, 70000, 120000],
    'Savings': [5000, 20000, 10000, 3000, 30000, 2000, 40000, 50000],
    'Loan Granted': [1, 0, 1, 1, 0, 1, 0, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Separate features (Income, Savings) and the target (Loan Granted)
X = df[['Income', 'Savings']]  # Features
y = df['Loan Granted']          # Target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Output the predictions
print("Predictions:", y_pred)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example prediction for new data
test_income = 75000
test_savings = 25000
user_data = [[test_income, test_savings]]

# Predict loan status for new data
user_prediction = model.predict(user_data)
if user_prediction[0] == 1:
    print("Loan Granted")
else:
    print("Loan Rejected")
