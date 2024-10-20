import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

# Step 1: Generate synthetic data using make_blobs
n_samples = 500
n_features = 2
n_clusters = 3
X, _ = datasets.make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# Step 2: Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)  # Adjust eps and min_samples as needed
labels = dbscan.fit_predict(X)

# Step 3: Visualize the results
plt.figure(figsize=(10, 6))

# Plot clustered data points
unique_labels = set(labels)
for label in unique_labels:
    if label == -1:
        # Noise points
        plt.scatter(X[labels == label][:, 0], X[labels == label][:, 1], color='k', marker='x', label='Noise')
    else:
        plt.scatter(X[labels == label][:, 0], X[labels == label][:, 1], label=f'Cluster {label}')

plt.title('DBSCAN Clustering Result with Blobs')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
