# print("import requiring libraries")
# For the format of output, all the print, mlask, mlcat and such output will be turn off
import argparse
import sys
import pathlib
import os
import pandas as pd
from utils import KMeans, join_path, update, join_path, save_animation, view
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mlhub.pkg import mlask, mlcat, mlpreview
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "k", type=int, help="the number of clusters for the algorithm")
parser.add_argument("path", nargs="?", help="file(s), folder, or url to input, in the format of csv",
                    type=argparse.FileType('r'), default=sys.stdin)
parser.add_argument("-o", "--output", nargs="?", help="filepath for output, or will output to stdout",
                    type=argparse.FileType('w'), default=sys.stdout)
parser.add_argument(
    "-v", "--view", help="with this option, will popout an video for visualisation", action="store_true")
parser.add_argument("-s", "--savemovie", help="the path for movie save path",
                    type=pathlib.Path, default=join_path("examples/train.mp4"))
args = parser.parse_args()

k = args.k
save_movie = os.path.abspath(args.savemovie)
origin_out = sys.stdout

# redirect the output
sys.stdout = args.output

df = pd.read_csv(args.path)
data = df.to_numpy()
fig, ax = plt.subplots()
kmeans = KMeans(k, input_data=data, repeat_times=1, slience=True)
kmeans.farest_center()
fig, ax = plt.subplots()
kmeans.set_ax(ax)
ani = animation.FuncAnimation(
    fig, update, frames=50, fargs=(kmeans, False,), interval=500)
writer = animation.FFMpegWriter(
    fps=30, metadata=dict(artist='Me'), bitrate=1800)
save_animation(ani, save_movie, output=False)
df["label"] = kmeans.labels
header = ','.join(df.columns)
if args.view:
    # the mlpreview will call print(begin+prompt), which includes a '\n'
    mlpreview(save_movie, begin="", msg=header)
else:
    print(header)
center_labels = [f"center {i}" for i in range(k)]
center_df = pd.DataFrame(kmeans.centers)
center_df["label"] = center_labels
print(center_df.to_csv(header=False, index=False, float_format='%.3f'))
