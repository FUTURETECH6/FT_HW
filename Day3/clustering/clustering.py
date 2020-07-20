import numpy as np
import scipy.io as sio
from plot import plot
from todo import kmeans
from todo import spectral
from todo import knn_graph

cluster_data = sio.loadmat('cluster_data.mat')
X = cluster_data['X']

idx = kmeans(X, 2)
plot(X, idx, "Clustering-kmeans")

W = knn_graph(X, 10, 1.0)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_10_1.0")

W = knn_graph(X, 15, 1.0)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_15_1.0")

W = knn_graph(X, 20, 1.0)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_20_1.0")

W = knn_graph(X, 10, 1.45)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_10_1.45")

W = knn_graph(X, 15, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_15_1.45")

W = knn_graph(X, 20, 1.45)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_20_1.45")

W = knn_graph(X, 10, 2.0)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_10_2.0")

W = knn_graph(X, 15, 2.0)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_15_2.0")

W = knn_graph(X, 20, 2.0)
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral_20_2.0")