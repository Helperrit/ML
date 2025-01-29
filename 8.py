from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Example data
data = np.array([[0.1, 0.6], [0.15, 0.71], [0.08, 0.9], [0.16, 0.85],
                 [0.2, 0.3], [0.25, 0.5], [0.24, 0.1], [0.3, 0.2]])

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(data)

# Predict for VAR1=0.906 and VAR2=0.606
predicted_cluster = kmeans.predict([[0.906, 0.606]])
print("Predicted Cluster:", predicted_cluster[0])

# Sample data: 9 combinations of VAR1 and VAR2
data = {
    'VAR1': [0.123, 0.345, 0.567, 0.234, 0.456, 0.789, 0.987, 0.876, 0.654],
    'VAR2': [0.234, 0.678, 0.890, 0.345, 0.567, 0.456, 0.123, 0.234, 0.789],
    'Classification': ['Class A', 'Class B', 'Class A', 'Class C', 'Class B', 'Class C', 'Class A', 'Class C', 'Class B']
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Features (VAR1, VAR2)
X = df[['VAR1', 'VAR2']].values

# Apply K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X)

# Assign clusters to actual classifications
cluster_labels = kmeans.labels_
df['Cluster'] = cluster_labels

# Majority voting to map cluster to class
cluster_to_class = {}
for cluster in range(3):
    mask = df['Cluster'] == cluster
    most_common_class = df[mask]['Classification'].mode()[0]
    cluster_to_class[cluster] = most_common_class

df['Predicted_Class'] = df['Cluster'].map(cluster_to_class)

# Compute accuracy (informally)
accuracy = accuracy_score(df['Classification'], df['Predicted_Class'])
print(f"Accuracy of K-means clustering: {accuracy:.2f}")

# New data point: VAR1 = 0.906, VAR2 = 0.606
new_point = np.array([[0.906, 0.606]])

# Predict the cluster for the new data point
cluster = kmeans.predict(new_point)

# Find the centroid for the predicted cluster
centroid = kmeans.cluster_centers_[cluster]

# Output the predicted cluster and centroid
print(f"Predicted Cluster for VAR1=0.906, VAR2=0.606: Cluster {cluster[0]}")
print(f"Centroid of this Cluster: {centroid}")

# Optional: Plot the clusters and centroids
plt.scatter(df['VAR1'], df['VAR2'], c=kmeans.labels_, cmap='viridis', label='Data Points')
plt.scatter(centroid[0][0], centroid[0][1], c='red', marker='X', s=200, label='Centroid')
plt.scatter(new_point[0][0], new_point[0][1], c='blue', marker='o', s=100, label='New Point (VAR1=0.906, VAR2=0.606)')
plt.xlabel('VAR1')
plt.ylabel('VAR2')
plt.legend()
plt.title('K-means Clustering with 3 Centroids')
plt.show()
