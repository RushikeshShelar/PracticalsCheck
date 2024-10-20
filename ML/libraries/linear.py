# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample Data
data = {'Experience': [1, 2, 3, 4, 5], 'Salary': [40000, 42000, 49000, 58000, 63000]}
df = pd.DataFrame(data)

# Features and labels
X = df[['Experience']]
y = df['Salary']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Visualize results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.show()
