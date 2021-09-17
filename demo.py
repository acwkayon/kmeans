# import library
print("import requiring libraries")

from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

print("generating dataset")

# dataset
k = 5
max_steps = 50
data, origin_label = make_blobs(n_samples=k * 100, centers=k)

cm = plt.cm.get_cmap("viridis")
my_cmap = cm(np.linspace(0, 1, k))
# Set alpha
my_cmap[:, -1] = 0.3
# Create new colormap
my_cmap = ListedColormap(my_cmap)


def update_frame(i):
    # calculate distance
    for j, c in enumerate(centers):
        distance[:, j] = np.linalg.norm(data - c, axis=1)
    # assign labels
    labels = np.argmin(distance, axis=1)
    # re-calculate centers
    for j in range(k):
        centers[j] = np.mean(data[labels == j], axis=0)
    print(f"Calculating {i}%", end='\r')
    scatter_0.set_array(labels)
    scatter_1.set_offsets(centers)
    ax.set_title(f'Step: {i}')
    return scatter_0, scatter_1


if __name__ == '__main__':
    fig, ax = plt.subplots()
    labels = np.zeros([data.shape[0]], dtype=np.float64)
    distance = np.zeros([data.shape[0], k], dtype=np.float64)
    centers = np.zeros([k, data.shape[1]], dtype=np.float64)
    '''# random choose center
    centers = data[np.random.randint(data.shape[0], size=k)]
    # calculate distance
    for j, c in enumerate(centers):
        distance[:, j] = np.linalg.norm(data - c, axis=1)'''
    # assign labels
    labels = np.array(list(np.arange(0, k)) * 100)
    # re-calculate centers
    for j in range(k):
        centers[j] = np.mean(data[labels == j], axis=0)

    scatter_0 = ax.scatter(data[:, 0], data[:, 1],
                           c=labels, cmap=my_cmap, marker='.', lw=0.5)
    scatter_1 = ax.scatter(centers[:, 0], centers[:, 1], c=range(
        k), cmap="viridis",  marker='x', lw=3, s=60)
    ani = animation.FuncAnimation(
        fig, update_frame, frames=int(max_steps), interval=200)
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("examples/demo_movie.mp4")
    print("Output to cache directory")
