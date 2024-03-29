# -*- coding: utf-8 -*-
#
# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Wed 2022-08-17 18:52:32 -0500 Anita Williams>
#
# Authors: Gefei Shan, Graham.Williams@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Gefei Shan, Graham Williams. All rights reserved.

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

# Import the required libraries.

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


# -----------------------------------------------------------------------
# Command line argument and options
# -----------------------------------------------------------------------

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
    """Train a K-means cluster model from observataions in the DATAFILE.

K must be specified as the number of clusters to train.

FILENAME is optional and if not supplied the data is obtained from
STDIN. It is a CSV file of named numeric columns, generally expected
to be NORMALISEd.

The output to STDOUT is a k-means cluster model represented as a CSV
file with each of the k-means (the centers or centroids of each
cluster) on a single line, together with a uique label (0..k-1) to
identify the cluster.

    """

    # Construct a suitably structured dataset from iunput CSV file, quit if file does not meet format requirements.

    try:
        df = pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        click.echo("Exiting model training as no data is available.")
        sys.exit(1)
        
    if not all(df.apply(lambda col: pd.to_numeric(col, errors='coerce').notnull().all())) == True:
        click.echo("Exiting model training as data looks malformed. Please check that all values are numeric.")
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
            mlpreview(movie, begin="", msg="")

    # Output the model as a center per label, CSV format.

    labels = range(k)
    centers = pd.DataFrame(kmeans.centers)
    centers["label"] = labels
    model = centers.to_csv(header=False, index=False, float_format='%.2f')
    click.echo(header, output)
    click.echo(model.strip(), output)


if __name__ == "__main__":
    cli(prog_name="train")
