# import library
print("import requiring libraries")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils import update, KMeans, view, prepare, join_path, save_animation, join_path
from iris import plot_origin
import os
from mlhub.pkg import mlcat, mlask

if __name__ == '__main__':
    introduction = '''k-means is an unsupervised clustering algorithm, which does not require any pre-labeled knowledge. The algorithm groups data into k clusters, each is represent by its cluster centroid, which is chosen randomly in our showcase, as shown in the following figure.\n
    '''
    mlcat(title="K-means algorithm showcase demo", text=introduction, delim="=", begin="", end="")
    fig, ax = plt.subplots()
    # The first showcase
    kmeans = KMeans(k=3, samples=3*100,centers = [[1,1],[1,-1],[-1,-1]], repeat_times=1)
    ax.set_xlabel("age")
    ax.set_ylabel("income")
    kmeans.set_ax(ax)
    fig.savefig(join_path("examples/Initializing.png"))
    mlask(begin= "Will show the randomly initialized centroids.\n",end="Now ")
    view(os.path.abspath("examples/Initializing.png"))
    mlask(end = '\r')
    introduction_steps = '''The algorithm iteratively applies the 2 steps:\n\n 1. Assign all data points to its nearest centroid;\n\n 2. Update the centroids position as the mean of its clusters.\n\nThe algorithm would stop at an optimal point where repeating the 2 steps would not increase any performance, or the centers would stop moving.\n
    '''
    mlcat(title="K-means algorithm showcase demo", text=introduction_steps, delim="=", begin="", end="")
    ani = animation.FuncAnimation(fig, update, frames=50,fargs=(kmeans,),interval=500)
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    save_animation(ani, join_path("examples/movie1.mp4"))
    view(join_path("examples/movie1.mp4"))
    mlask(end = '\r') # wait until any input

    # The second showcase
    fig, ax = plt.subplots()
    mlcat(title="K-means algorithm showcase demo", text="Here is another example dataset for clustering, which has more separate data points: \n", delim="=", begin="", end="\n")
    kmeans = KMeans(k=3, samples=3*100, cluster_std = 0.8,centers = [[0,5],[4,-3],[-4,-3]], repeat_times=1)
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(fig, update, frames=40,fargs=(kmeans,),interval=500)
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    save_animation(ani, join_path("examples/movie2.mp4"))
    view(join_path("examples/movie2.mp4"))
    mlask(end = '\r') # wait until any input

    # The iris showcase
    print("Loading datasets", end = '\r')
    from sklearn import datasets
    iris = datasets.load_iris()
    data = iris.data
    iris_labels = iris.target
    fig, ax = plt.subplots()
    mlcat(title="K-means algorithm showcase demo", text="Applying the algorithm on the well known iris dataset: \n", delim="=", begin="", end="\n")
    kmeans = KMeans(3, samples=150, input_data=data)
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(
        fig, update, frames=50, fargs=(kmeans,), interval=500 )
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("examples/iris.mp4")
    save_animation(ani, join_path("examples/iris.mp4"))
    view(join_path("examples/iris.mp4"))
    mlask(end = '\r')
    plot_origin(iris_labels,kmeans, join_path("examples/iris.png"))
    view(join_path("examples/iris.png"))
    mlask(prompt = "Press Enter to Exit")
