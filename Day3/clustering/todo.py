import numpy as np
from scipy.spatial.distance import cdist


def kmeans(X, k):
    '''
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    OUTPUT:  idx: cluster label, N-by-1 vector
    '''

    N, P = X.shape
    idx = np.zeros(N)
    # YOUR CODE HERE
    # ----------------
    # ANSWER BEGIN
    # ----------------

    # Normal Distribution
    centers = np.random.randn(k, P) * np.std(X, axis=0) + np.mean(X, axis=0)

    # Random
    # rand_array = np.arange(0, N)
    # np.random.shuffle(rand_array)
    # centers = np.zeros((k, P))
    # centers = X[rand_array[:k], :]

    pre_centers = np.zeros((k, P))
    centers_move = 1.0
    min_move = 0.1E-9
    dist = np.zeros((N, k))  # N points, each has k dist to k centers

    while centers_move > min_move:  # Convergence is fast enough
        for iCenter in range(k):
            dist[:, iCenter] = np.linalg.norm(X-centers[iCenter], axis=1)

        idx = np.argmin(dist, axis=1)  # In Dimension of k centers

        pre_centers = centers.copy()   # Very important... Don't forget the shallow copy

        for iCenter in range(k):
            # In Dimension of N points
            centers[iCenter] = np.mean(X[idx == iCenter], axis=0)

        centers_move = np.sum(np.linalg.norm(
            centers[i] - pre_centers[i]) for i in range(k))

    # ----------------
    # ANSWER END
    # ----------------
    return idx


def spectral(W, k):
    '''
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    '''
    N = W.shape[0]
    idx = np.zeros((N, 1))
    # YOUR CODE HERE
    # ----------------
    # ANSWER BEGIN
    # ----------------

    DegMat = np.diag(np.sum(W, axis=1))

    LapMat = DegMat - W

    eigValues, eigVectors = np.linalg.eig(
        np.dot(np.linalg.inv(DegMat), LapMat))  # invD(D-W)

    dim = len(eigValues)
    dictEigValues = dict(zip(eigValues, range(dim)))
    ix = [dictEigValues[i] for i in np.sort(eigValues)[0:k]]
    X = eigVectors[:, ix]

    # ----------------
    # ANSWER END
    # ----------------
    X = X.astype(float)
    idx = kmeans(X, k)
    return idx


def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')
    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]  # aj[i,i] = 0
        W[i, index] = 1
    W[aj >= threshold] = 0
    return W
