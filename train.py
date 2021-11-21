# -*- coding: utf-8 -*-
#
# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Sunday 2021-11-21 08:48:30 AEDT Graham Williams>
#
# Authors: Gefei Shan, Graham.Williams@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Gefei Shan, Graham Williams. All rights reserved.


import sys
import click

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mlhub.pkg import mlpreview

from utils import KMeans, update, save_animation


# Command line argument and options.

@click.command()
@click.argument("k", type=click.IntRange(2))
@click.option("-i", "--input",
              default=sys.stdin,
              type=click.File('r'),
              help="Filename of the CSV file to cluster, or from STDIN.")
@click.option("-o", "--output",
              default=sys.stdout,
              type=click.File('w'),
              help="Filename of the CSV file to save model, or to STDOUT.")
@click.option("-m", "--movie",
              type=click.Path(),
              help="Filename of the movie file to save if desired.")
@click.option("--view",
              is_flag=True,
              default=False,
              help="Popup a movie viewer to visualise the algorithm.")
def cli(k, input, output, movie, view):
    """Train a k-means cluster model, output as centers and labels."""

    # Construct a suitably structured dataset from iunput CSV file.

    try:
        df = pd.read_csv(input)
    except pd.errors.EmptyDataError:
        click.echo("Exiting model training as no data is available.")
        sys.exit(1)
    data = df.to_numpy()

    # Build the k-means model and animation.

    kmeans = KMeans(k, input_data=data, repeat_times=1, slience=True)
    kmeans.farest_center()
    fig, ax = plt.subplots()
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(
        fig, update, frames=50, fargs=(kmeans, False,), interval=500)
    if view:
        save_animation(ani, movie, verbose=False)

    df["label"] = kmeans.labels
    header = ','.join(df.columns)

    if view:
        mlpreview(movie, begin="", msg=header)

    # Output the model as a center per label, CSV format.

    labels = range(k)
    centers = pd.DataFrame(kmeans.centers)
    centers["label"] = labels
    model = centers.to_csv(header=False, index=False, float_format='%.2f')
    click.echo(header)
    click.echo(model.strip())


if __name__ == "__main__":
    cli(prog_name="train")
