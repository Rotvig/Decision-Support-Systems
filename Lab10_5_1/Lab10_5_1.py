from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot
from sklearn.datasets.samples_generator import make_blobs

k=2

X, y = make_blobs(n_samples=50, centers=k)

kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(k):
    # select only data observations with cluster label == i
    ds = X[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()