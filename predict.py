# -*- coding: utf-8 -*-
#
# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Wed 2021-12-29 14:59:54 -0500 Anita Williams>
#
# Authors: Gefei Shan, Graham.Williams@togaware.com, Anita@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Gefei Shan, Graham Williams. All rights reserved.

import os
import sys
import click

import pandas as pd
import numpy as np

from mlhub.pkg import get_cmd_cwd

# Ensure paths are relative to the user's cwd.

os.chdir(get_cmd_cwd())

# Command line argument and options.


@click.command()
@click.argument('csvfile',
                type=click.File('r'),
                default=sys.stdin) 
# help="the input data file, in csv format"
@click.argument('modelfile',
                type=click.File('r'),
                default=sys.stdin)  
# help="model file for predictions, in csv format"

def cli(modelfile, csvfile, output):

    # Ensure we look for the files in the user's cwd
    os.chdir(get_cmd_cwd())

    # Quit the program if there is not any data to read
    try:
        df = pd.read_csv(csvfile)
    except pd.errors.EmptyDataError:
        click.echo("Quitting as CSV file has no data available.")
        sys.exit(1)
    data = df.to_numpy()

    # modelfile, the centers of clusters
    df_centers = pd.read_csv(modelfile)
    if "labels" in df_centers.columns:
        df_centers["label"] = df_centers["labels"]
        df_centers = df_centers.drop(columns="labels")

    df_centers.sort_values(by="label", inplace=True)
    centers = df_centers.drop(columns="label").to_numpy()
    label_index = df_centers["label"]

    k, m = centers.shape
    data = df.to_numpy()
    n = data.shape[0]
    distance = np.zeros([n, k])
    # calculate distance and assign labels
    for j, c in enumerate(centers): 
        distance[:, j] = np.linalg.norm(data - c, axis=1)
    new_labels = np.argmin(distance, axis=1)
    # map labels from 0,...,k-1 to the provided label string
    df_labels = pd.Series(data=new_labels).map(label_index.to_dict())

    df["label"] = df_labels

    origin_out = sys.stdout
    #  redirect the output
    sys.stdout = output

    print(df.to_csv(index=False).strip())


if __name__ == "__main__":
    cli(prog_name="predict")
