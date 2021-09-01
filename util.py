import numpy as np
from sklearn.datasets import make_blobs


class KMeans:
    def __init__(self, k, samples=0):
        if samples == 0:
            samples = k * 100
        data, origin_label = make_blobs(n_samples=samples, centers=k)
        self.k = k
        self.data = data
        q = int((samples - 1) / k + 1) # ceiling of (samples/k)
        self.labels = np.array(list(np.arange(0, k)) * q)[:samples]
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
