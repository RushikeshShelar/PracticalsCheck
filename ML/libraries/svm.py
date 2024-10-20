import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Generate synthetic dataset
X, y = datasets.make_classification(
    n_samples=100,         # Number of samples
    n_features=2,         # Number of features
    n_informative=2,      # Number of informative features
    n_redundant=0,        # Number of redundant features
    n_clusters_per_class=1, # Number of clusters per class
    random_state=42       # For reproducibility
)

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the SVM model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 5: Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function to visualize decision boundaries
def plot_decision_boundaries(X, y, model):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundaries')
    plt.show()

# Step 6: Plot decision boundaries
plot_decision_boundaries(X, y, svm_classifier)
