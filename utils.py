import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from platform import system
from mlhub.pkg import mlcat, mlpreview, get_cmd_cwd
from mlhub.utils import get_package_dir
from pathlib import Path
import os


def join_path(filename):
    # under linux: join the file path with package directory(~/.mlhub/kmeans)
    # under window: join the file path with cwd, for convince of debug
    dirpath = get_package_dir("kmeans") if is_linux() else get_cmd_cwd()
    return os.path.abspath( os.path.join(dirpath, filename))

def prepare(filepath):
    # mkdir if the file path directory is not exist
    directory = os.path.dirname(filepath)
    Path(directory).mkdir(parents=True,exist_ok=True)

def save_animation(ani, filepath):
    # save animation to the filepath, no matter whether the place exist
    prepare(filepath)
    ani.save(filepath)
    print(f"Save animation to {filepath}")

def is_linux():
    return system() == "Linux"

def view(filename, previewer = None):
    mlpreview(filename, begin = "Opening file... ",msg="Close the graphic window using Ctrl-W. ", previewer=previewer)

class KMeans:
    # Class for K-means algorithm, with k as the clustering target
    def __init__(self, k, samples=0, n_features=2, centers=0, input_data=None, repeat_times=3, cluster_std=1.0):
        print("Initializing data...")
        if centers == 0:
            centers = k
        if samples == 0:
            samples = k * 100
        # using sklearn make_blobs method to generate data
        data, origin_label = make_blobs(
            n_samples=samples, n_features=n_features, centers=centers, center_box=(-10.0, 10.0), cluster_std=cluster_std)
        if input_data is not None:
            self.data = input_data
        else:
            self.data = data
        self.k = k
        # first time assign labels, centers and distance
        q = int((samples - 1) / k + 1)  # ceiling of (samples/k)
        #self.labels = np.array(list(np.arange(0, k)) * q)[:samples]
        self.labels = np.zeros(data.shape[0], dtype=np.float64)
        self.centers = np.zeros([k, data.shape[1]], dtype=np.float64)
        self.distance = np.zeros([data.shape[0], k], dtype=np.float64)
        # self.calculate_centers()
        self.select_centers()
        self.converge = -10 # the frames would still remain after the algorithm converge
        self.converge_times = 0 # count the times the algorithm already converged
        self.repeat_times = repeat_times # the times the algorithm will repeat
        self.init_cm()

    def calculate_labels(self):
        # calculate distance
        for j, c in enumerate(self.centers):
            self.distance[:, j] = np.linalg.norm(self.data - c, axis=1)
        # assign labels
        new_labels = np.argmin(self.distance, axis=1)
        if np.array_equal(new_labels, self.labels):
            self.converge += 1
        self.labels = new_labels

    def calculate_centers(self):
        # re-calculate centers
        for j in range(self.k):
            self.centers[j] = np.mean(self.data[self.labels == j], axis=0)

    def select_centers(self):
        # random select centers from data, without repeat
        mask = np.random.choice(self.data.shape[0], size=self.k, replace=False)
        self.centers = self.data[mask]
        self.converge = -10
        self.calculate_labels()

    def pca(self):
        # return the 2-d PCA result, mainly for animation plotting
        self.pca_model = PCA(n_components=2)
        print("applying PCA to reduce dimensions to 2")
        return self.pca_model.fit_transform(self.data)

    def init_cm(self):
        # Initializing the colormap for plotting
        cm = plt.cm.get_cmap("viridis")
        my_cmap = cm(np.linspace(0, 1, self.k))
        # Set alpha
        my_cmap[:, -1] = 0.5
        # Create new colormap
        self.my_cmap = ListedColormap(my_cmap)

    def set_ax(self, ax):
        self.ax = ax
        if self.data.shape[1] > 2:
            self.printdata = self.pca()
        else:
            self.printdata = self.data
        self.scatter_0 = self.ax.scatter(self.printdata[:, 0], self.printdata[:, 1],
                                         c=self.labels, cmap=self.my_cmap, marker='.', lw=0.5)
        self.scatter_1 = self.ax.scatter(self.centers[:, 0], self.centers[:, 1], c=range(
            self.k), cmap="viridis",  marker='x', lw=3, s=60)

    def plot_centers(self):
        if self.data.shape[1] > 2:
            center = self.pca_model.transform(self.centers)
        else:
            center = self.centers
        self.scatter_1.set_offsets(center)

    def itergenetor(self):
        # genetor for frame iteration, exhausted after converge serval times
        if self.converge_times == 0:
            i = 0
        while self.converge_times < self.repeat_times:
            if self.converge > 0:
                self.converge_times += 1
                if self.repeat_times-self.converge > 0:
                    self.select_centers()
                    i = 0
            yield i
            i += 1

#todo: convex hull


def update(i, kmeans: KMeans):
    # use as animation.FuncAnimation(fig, animate, farg = (kmeans,), interval=...)
    if i == 0:
        kmeans.scatter_0.set_array(kmeans.labels)
        kmeans.plot_centers()
    if i % 2 == 0:  # when i is even, update labels
        kmeans.scatter_0.set_array(kmeans.labels)
        kmeans.calculate_labels()
    else:  # when i is odd, update centers
        kmeans.calculate_centers()
        kmeans.plot_centers()
    kmeans.ax.set_title(f'Step: {int(i/2)}')
    print(f"rendering frame {i}", end='\r')
    return (kmeans.scatter_0, kmeans.scatter_1)
