# Write a program to Demonstrate for the following data, which specify classifications for nine combinations of VAR1 and VAR2 predict a classification for a case where VAR1=0.906 and VAR2=0.606, using the result of k-means clustering with 3 means (i.e., 3centroids)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Offline data
data = {
    'VAR1': [1.0, 1.1, 0.9, 0.8, 0.7, 1.2, 0.6, 0.5, 0.4],
    'VAR2': [0.8, 0.9, 0.7, 0.6, 0.5, 0.8, 0.4, 0.3, 0.2],
    'Classification': ['A', 'A', 'A', 'B', 'B', 'A', 'B', 'C', 'C']
}

# Create DataFrame
df = pd.DataFrame(data)
X = df[['VAR1', 'VAR2']]

# Apply k-means clustering with 3 centroids
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predict the cluster for a new point
new_point = np.array([[0.906, 0.606]])
predicted_cluster = kmeans.predict(new_point)

# Visualize the clusters and the new point
plt.scatter(X['VAR1'], X['VAR2'], c=kmeans.labels_, cmap='viridis', label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.scatter(new_point[:, 0], new_point[:, 1], s=100, c='blue', label='New Point', marker='X')
plt.xlabel('VAR1')
plt.ylabel('VAR2')
plt.title('K-Means Clustering (3 Centroids)')
plt.legend()
plt.show()

# Output predicted cluster for the new point
print("Predicted cluster for VAR1=0.906 and VAR2=0.606:", predicted_cluster[0])
