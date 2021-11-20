# -*- coding: utf-8 -*-
#
# Time-stamp: <Saturday 2021-11-20 20:36:46 AEDT Graham Williams>
#
# Copyright (c) Gefei Shan, Graham Williams. All rights reserved.
# Licensed under GPLv3
# Authors: Gefei Chen, Graham.Williams@togaware.com
#
# MLHub demonstrator and toolkit for kmeans.

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import update, KMeans, view, save_animation, plot_origin

from mlhub.pkg import mlcat, mlask


def main():

    introduction = """K-means is an unsupervised clustering algorithm which does not
    require any pre-labeled data to build a model. The algorithm groups
    data into k clusters, each represented by its cluster centroid. The
    user needs to provide the value of k (the number of clusters).

    Our first example will build a clustering for a random dataset (a
    different one each time) consisting of two variables, age and
    income, for each person. The task begins by randomly choosing k
    (3) centroids (shown as X's in the graphic). Each point is also
    coloured according to its nearest centroid.

    The plot that we will display will show these three randonly
    chosen starting points as the X. The dots in the plot represent
    each individual person (age and income) and are coloured according
    to the colour of the nearest X.
"""
    mlcat(title="K-Means Algorithm Showcase", text=introduction)

    ################################################################
    # First Showcase

    kmeans = KMeans(k=3,
                    samples=3 * 100,
                    centers=[[1, 1], [1, -1], [-1, -1]],
                    repeat_times=1)

    fig, ax = plt.subplots()
    ax.set_xlabel("age")
    ax.set_ylabel("income")
    kmeans.set_ax(ax)

    fig_path = "examples/Initializing.png"

    fig.savefig(fig_path)

    mlask(begin="\n", end="\n",
          prompt="Press Enter to display the initial plot")

    view(fig_path)

    introduction_steps = """The algorithm iteratively applies the 2 steps:

    1. Assign all data points to its nearest centroid;

    2. Update the centroids position as the mean of its clusters.

    The algorithm stops at an optimal point where repeating the two
    above steps would not increase any measure of goodness of the
    model or the centers would stop moving.

    The algorithm moves the centroids (X) each step according to a
    measure which aims to maximise the between cluster sum of squares
    (distances) and to minimise the within cluster sum of squares.

    After moving the centroid the dots are recoloured according to
    their nearest centroid. The process stops when no improvement can
    be made to the measure.

    A movie is generated to show the change to the centroids each
    step.  When no further improvement can be made the centroids stop
    moving.

    We will now generate and then display the movie.
"""
    mlcat(title="K-Means Algorithm - The Movie", text=introduction_steps)

    # Animation.

    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=50,
                                  fargs=(kmeans,),
                                  interval=500)

    path = "examples/movie1.mp4"
    save_animation(ani, path)
    mlask(begin="\n", end="\n", prompt="Press Enter to view the movie")
    view(path)

    ################################################################
    # Second Showcase

    message = """Another example dataset illustrates the algorithm with data points
    that are more clearly seperate as 3 clusters.

    Again we will illustrate the iterations of the algorithm as the
    final set of best centroids are fit to the data.
"""
    mlcat(title="K-Means Algorithm Second Demo", text=message)

    kmeans = KMeans(k=3, samples=3 * 100, cluster_std=0.8,
                    centers=[[0, 5], [4, -3], [-4, -3]], repeat_times=1)

    fig, ax = plt.subplots()
    kmeans.set_ax(ax)

    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=40,
                                  fargs=(kmeans,),
                                  interval=500)

    print("")
    path = "examples/movie2.mp4"
    save_animation(ani, path)
    mlask(begin="\n", end="\n", prompt="Press Enter to view the movie")
    view(path)

    ################################################################
    # Iris Showcase

    message = """Cluster the common iris dataset. To visualise the data we first do
    a principle component analysis to map to the two most important
    components, to suit a 2D plot which we display. The points are
    coloured according the the iris species.

    Because this dataset consists of 4 variables for each iris plant
    (sepal width and length, and petal width and length), and we want
    to visualise the data in 2 dimensions (x and y), a principle
    component analysis (PCA) is undertaken, and the two most
    significant components are chaosen to be displayed in the plot.
"""

    mlcat(title="K-Means Iris Clustering", text=message)

    mlcat(text="Loading iris dataset from the sklearn package of datasets.\n")

    from sklearn import datasets
    iris = datasets.load_iris()
    data = iris.data
    iris_labels = iris.target
    fig, ax = plt.subplots()

    kmeans = KMeans(3, samples=150, input_data=data)
    print("")
    kmeans.set_ax(ax)
    print("")

    path = "examples/iris.png"
    plot_origin(iris_labels, kmeans, path)
    mlask(begin="\n", end="\n",
          prompt="Press Enter to display the iris dataset")
    view(path)

    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=50,
                                  fargs=(kmeans,),
                                  interval=500)

    path = "examples/iris.mp4"
    save_animation(ani, path)
    print("")
    view(path, ask=False)

    mlask(begin="\n", end="\n", prompt="Press Enter to Exit")


if __name__ == '__main__':
    main()
