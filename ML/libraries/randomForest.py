# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Hardcoded data for Income, Savings, and Loan Granted (1 = Yes, 0 = No)
data = {
    'Income': [30000, 45000, 50000, 60000, 20000, 70000, 80000, 55000, 40000, 75000],
    'Savings': [10000, 20000, 15000, 25000, 8000, 35000, 40000, 20000, 12000, 38000],
    'Loan Granted': [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
}

# Create a DataFrame from the hardcoded data
df = pd.DataFrame(data)

# Extract features (Income and Savings) and target label (Loan Granted)
X = df[['Income', 'Savings']]  # Features
y = df['Loan Granted']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(cr)

# Take user input to test the model
test_income = float(input("Enter income: "))
test_savings = float(input("Enter savings: "))
test_data = [[test_income, test_savings]]

# Predict if loan should be granted or not
loan_pred = model.predict(test_data)
if loan_pred == 1:
    print("Loan will be Granted!")
else:
    print("Loan will NOT be Granted!")
