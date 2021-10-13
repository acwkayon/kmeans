print("import requiring libraries")
import argparse
import sys
import pandas as pd
from utils import KMeans, join_path, update, join_path, save_animation, view
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mlhub.pkg import mlask, mlcat
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("k", type=int, help="the number of clusters for the algorithm")
parser.add_argument("path", nargs="*", help="file(s), folder, or url to csv")
args = parser.parse_args()

k = args.k
if len(args.path) == 0:
    print("Reading input csv from standard input.")
    input_list = [sys.stdin]
else:
    print("Reading input csv from command line arguments")
    input_list = args.path
for file in args.path:
    print(f"Reading data from {file}")
    df = pd.read_csv(file)
    data = df.to_numpy()
    fig, ax = plt.subplots()
    kmeans = KMeans(k,input_data=data,repeat_times=1)
    kmeans.farest_center()
    fig, ax = plt.subplots()
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(
        fig, update, frames=50, fargs=(kmeans,), interval=500 )
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("examples/iris.mp4")
    save_animation(ani, join_path("examples/train.mp4"))
    view(join_path("examples/train.mp4"))
    mlask(end = '\r')
    center_csv = ""
    for c in kmeans.centers:
        line = np.array2string(c, separator = ",")
        line = line[1:-1]+'\n'
        center_csv = center_csv + line
    mlcat(title = f"The centers of clusters in csv format",text = "",delim="=",end='')
    print(center_csv)
