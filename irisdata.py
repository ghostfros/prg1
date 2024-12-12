import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import random

# Load the iris dataset
iris = load_iris()
data = iris['data']

# Number of clusters
K = 4

# Randomly initialize the centroids (cluster means) by selecting K random points from the dataset
np.random.seed(42)
initial_centroids = data[np.random.choice(data.shape[0], K, replace=False)]

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Function to assign each point to the nearest cluster
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# Function to update centroids based on cluster assignment
def update_centroids(data, clusters, K):
    new_centroids = np.zeros((K, data.shape[1]))
    for k in range(K):
        cluster_points = data[clusters == k]
        if len(cluster_points) > 0:
            new_centroids[k] = np.mean(cluster_points, axis=0)
    return new_centroids

# Perform K-means for 10 iterations
centroids = initial_centroids
for i in range(10):
    clusters = assign_clusters(data, centroids)
    centroids = update_centroids(data, clusters, K)

# Print the final cluster means
print("Final cluster means (centroids) after 10 iterations:")
for idx, centroid in enumerate(centroids):
    print(f"Cluster {idx+1} mean: {centroid}")