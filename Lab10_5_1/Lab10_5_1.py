from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print kmeans.cluster_centers_