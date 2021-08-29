import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, k):
        data, origin_label = make_blobs(n_samples=k * 100, centers=k)
        self.k = k
        self.data = data
        self.labels = np.array(list(np.arange(0, k)) * 100)
        self.centers = np.zeros([k, data.shape[1]], dtype=np.float64)
        self.distance = np.zeros([data.shape[0], k], dtype=np.float64)
        # pre-calculate centers
        for j in range(k):
            centers[j] = np.mean(data[labels == j], axis=0)

    def update_step(self):
        # calculate distance
        for j, c in enumerate(centers):
            self.distance[:, j] = np.linalg.norm(self.data - c, axis=1)
        # assign labels
        self.labels = np.argmin(self.distance, axis=1)
        # re-calculate centers
        for j in range(k):
            self.centers[j] = np.mean(self.data[self.labels == j], axis=0)
