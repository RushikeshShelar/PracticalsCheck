import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Sample data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 
                'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 
                    'Mild', 'Mild', 'Hot', 'Mild', 'Mild', 'Hot', 
                    'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 
                 'Normal', 'High', 'Normal', 'Normal', 'Normal', 
                 'High', 'Normal', 'Normal'],
    'Windy': [False, True, False, False, False, True, True, 
              False, False, False, True, True, False, True],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert categorical variables to numerical values
df['Outlook'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
df['Temperature'] = df['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
df['Humidity'] = df['Humidity'].map({'High': 0, 'Normal': 1})
df['Windy'] = df['Windy'].astype(int)
df['Play'] = df['Play'].map({'No': 0, 'Yes': 1})

# Define features and target variable
X = df.drop('Play', axis=1)
y = df['Play']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the ID3 decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')  # Use 'entropy' for ID3
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Output the predictions
print("Predictions:", y_pred)

# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title("Decision Tree Visualization")
plt.show()
