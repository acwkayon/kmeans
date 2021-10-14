# print("import requiring libraries")
# For the format of output, all the print, mlask, mlcat and such output will be turn off
import argparse
import sys
import pandas as pd
from utils import KMeans, join_path, update, join_path, save_animation, view
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mlhub.pkg import mlask, mlcat, mlpreview
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("k", type=int, help="the number of clusters for the algorithm")
parser.add_argument("path", nargs="*", help="file(s), folder, or url to csv")
parser.add_argument("-o", "--output", nargs="*",help="filepath for output, or will output to stdout")
args = parser.parse_args()

k = args.k

if not bool(args.output) :
    args.output = []

if len(args.path) == 0:
    #print("Reading input csv from standard input.")
    input_list = [sys.stdin]
else:
    #print("Reading input csv from command line arguments")
    input_list = args.path
origin_out = sys.stdout
for i,file in enumerate(args.path):
    #print(f"Reading data from {file}")
    # redirect the output
    sys.stdout = origin_out
    if i < len(args.output):
        f = open(args.output[i],'w')
        sys.stdout = f
    df = pd.read_csv(file)
    data = df.to_numpy()
    fig, ax = plt.subplots()
    kmeans = KMeans(k,input_data=data,repeat_times=1,slience=True)
    kmeans.farest_center()
    fig, ax = plt.subplots()
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(
        fig, update, frames=50, fargs=(kmeans,False,), interval=500 )
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("examples/iris.mp4")
    save_animation(ani, join_path("examples/train.mp4"), output=False)
    # view(join_path("examples/train.mp4"))
    df["labels"] = kmeans.labels
    header = ','.join(df.columns)
    mlpreview(join_path("examples/train.mp4"),begin="",msg=header) # the mlpreview will call print(begin+prompt), which includes a '\n'
    # mlask(end = '\r')
    print(df.to_csv(header=False, index=False),end="")
    center_labels = [f"center {i}" for i in range(k)]
    center_df = pd.DataFrame(kmeans.centers)
    center_df["labels"] = center_labels
    print(center_df.to_csv(header=False, index=False))
