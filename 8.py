# 8.Write a program to Demonstrate for the following data, which specify classifications for nine combinations of VAR1 and VAR2 predict a classification for a case where VAR1=0.906 and VAR2=0.606, using the result of k-means clustering with 3 means (i.e., 3centroids)
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample data: 9 combinations of VAR1 and VAR2
data = {
    'VAR1': [0.123, 0.345, 0.567, 0.234, 0.456, 0.789, 0.987, 0.876, 0.654],
    'VAR2': [0.234, 0.678, 0.890, 0.345, 0.567, 0.456, 0.123, 0.234, 0.789],
    'Classification': ['Class A', 'Class B', 'Class A', 'Class C', 'Class B', 'Class C', 'Class A', 'Class C', 'Class B']
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
X = df[['VAR1', 'VAR2']].values

# Apply K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

# Majority voting to map cluster to class
cluster_to_class = {c: df[df['Cluster'] == c]['Classification'].mode()[0] for c in range(3)}
df['Predicted_Class'] = df['Cluster'].map(cluster_to_class)

# Compute accuracy and confusion matrix
accuracy = accuracy_score(df['Classification'], df['Predicted_Class'])
conf_matrix = confusion_matrix(df['Classification'], df['Predicted_Class'], labels=['Class A', 'Class B', 'Class C'])

# Predict for new data point
new_point = np.array([[0.906, 0.606]])
cluster = kmeans.predict(new_point)[0]
centroid = kmeans.cluster_centers_[cluster]

# Print results in the requested format
print(f"Predicted Cluster: {cluster}")
print(f"Accuracy of K-means clustering: {accuracy:.2f}")
print(f"Predicted Cluster for VAR1=0.906, VAR2=0.606: Cluster {cluster}")
print(f"Centroid of this Cluster: {centroid}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot clusters and centroids
plt.scatter(df['VAR1'], df['VAR2'], c=kmeans.labels_, cmap='viridis', label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.scatter(new_point[0][0], new_point[0][1], c='blue', marker='o', s=100, label='New Point')
plt.xlabel('VAR1')
plt.ylabel('VAR2')
plt.legend()
plt.title('K-means Clustering with 3 Centroids')
plt.show()
