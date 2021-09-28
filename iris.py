from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import KMeans, update, save_animation, prepare, join_path


def plot_origin(labels, kmeans: KMeans, filepath):
    ani_ax = kmeans.ax
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlim(ani_ax.get_xlim())
    ax1.set_ylim(ani_ax.get_ylim())
    ax1.scatter(kmeans.printdata[:, 0], kmeans.printdata[:, 1],
                c=labels, cmap="viridis", marker='.', lw=0.5)
    ax1.set_title("The original labels of iris dataset")
    prepare(filepath)
    fig1.savefig(filepath)
    print(f"Save origin labels plot to {filepath}")


if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data
    iris_labels = iris.target
    fig = plt.figure()
    ax = fig.add_subplot()
    kmeans = KMeans(3, samples=150, input_data=data)
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(
        fig, update, frames=kmeans.itergenetor(), fargs=(kmeans,), interval=200, save_count=150)
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("examples/iris.mp4")
    save_animation(ani, join_path("examples/iris.mp4"))
    plot_origin(iris_labels,kmeans, join_path("examples/iris.png"))
