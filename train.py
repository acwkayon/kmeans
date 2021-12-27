# -*- coding: utf-8 -*-
#
# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Saturday 2021-12-18 15:20:50 AEDT Graham Williams>
#
# Authors: Gefei Shan, Graham.Williams@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Gefei Shan, Graham Williams. All rights reserved.

import os
import sys
import click
import tempfile

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mlhub.pkg import mlpreview, get_cmd_cwd

from utils import KMeans, update, save_animation


# Ensure paths are relative to the user's cwd.

os.chdir(get_cmd_cwd())

# Command line argument and options.

@click.command()
@click.argument("k",
                type=click.IntRange(2))
@click.argument("filename",
                default=sys.stdin,
                type=click.File('r'))  # Path(exists=True, readable=True))
@click.option("-o", "--output",
              default=sys.stdout,
              type=click.File('w'),
              help="Filename of the CSV file to save model, or to STDOUT.")
@click.option("-m", "--movie",
              type=click.Path(),
              help="Filename of the movie file to save if desired.")
@click.option("-v", "--view",
              is_flag=True,
              default=False,
              help="Popup a movie viewer to visualise the algorithm.")
def cli(k, filename, output, movie, view):
    """Train a k-means cluster model, output as centers and labels."""

    # Construct a suitably structured dataset from iunput CSV file.

    try:
        df = pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        click.echo("Exiting model training as no data is available.")
        sys.exit(1)
    data = df.to_numpy()

    # Build the k-means model

    kmeans = KMeans(k, input_data=data, repeat_times=1, slience=True)
    kmeans.farest_center()
    df["label"] = kmeans.labels
    header = ','.join(df.columns)

    # Build the animation, view it and/or save it.

    if movie is None:
        save_movie = False
        with tempfile.TemporaryDirectory() as tmp:
            movie = os.path.join(tmp, 'mlhub_kmeans_movie.mp4')
    else:
        save_movie = True

    if view or save_movie:
        fig, ax = plt.subplots()
        kmeans.set_ax(ax)
        ani = animation.FuncAnimation(
            fig, update, frames=50, fargs=(kmeans, False,), interval=500)
        save_animation(ani, movie, verbose=False)
        if view:
            mlpreview(movie, begin="", msg=header)

    # Output the model as a center per label, CSV format.

    labels = range(k)
    centers = pd.DataFrame(kmeans.centers)
    centers["label"] = labels
    model = centers.to_csv(header=False, index=False, float_format='%.2f')
    click.echo(header, output)
    click.echo(model.strip(), output)


if __name__ == "__main__":
    cli(prog_name="train")
