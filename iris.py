from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import KMeans, update

if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data
    iris_labels = iris.target
    fig = plt.figure()
    ax = fig.add_subplot()
    kmeans = KMeans(3,samples=150, input_data=data)
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(fig, update, frames=kmeans.itergenetor(),fargs=(kmeans,),interval=200)
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("examples/iris.mp4")
