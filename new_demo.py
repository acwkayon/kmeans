# import library
print("import requiring libraries")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils import update, KMeans

if __name__ == '__main__':
    fig, ax = plt.subplots()
    kmeans = KMeans(3)
    ax.set_xlabel("age")
    ax.set_ylabel("income")
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(fig, update, frames=kmeans.itergenetor(),fargs=(kmeans,),interval=200)
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("examples/movie.mp4")
