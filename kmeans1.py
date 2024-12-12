import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset (assuming the file is in CSV format)
# Replace 'Countries_data_set_1.csv' with the correct file path
data = pd.read_csv('Countries_data_set_1.csv')

# Assuming the dataset has columns named 'Longitude' and 'Latitude'
X = data[['Longitude', 'Latitude']]

# Create the K-Means model with a predefined number of clusters (K)
K = 3  # You can adjust the number of clusters based on your requirement
kmeans = KMeans(n_clusters=K)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the original data for analysis
data['Cluster'] = labels

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='rainbow', s=50)
plt.title(f"K-Means Clustering of Countries Based on Longitude and Latitude (K={K})")
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Plot the cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centroids')

plt.legend()
plt.show()

# Print the resulting clusters and centroids
print("Cluster Centers (Centroids):")
print(centroids)