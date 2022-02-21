# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Monday 2022-01-03 09:02:34 +1100 Graham Williams>
#
# Authors: Gefei Shan, Anita@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Gefei Shan, Anita Williams. All rights reserved.

import os
import sys
import click

import numpy as np
import pandas as pd

from mlhub.pkg import get_cmd_cwd

# Ensure paths are relative to the user's cwd.

os.chdir(get_cmd_cwd())


# Command line interface, arguments and options.


@click.command()
@click.argument('datafile',
                type=click.File('r'))
@click.argument('modelfile',
                type=click.File('r'),
                default=sys.stdin)
def cli(modelfile, datafile):
    """Apply MODELFILE to DATAFILE.

DATAFILE is required. It is a CSV file with named numeric columns that
match those of the MODELFILE, except for the absence of a label
column.

MODELFILE is optional and is STDIN if not provided. It is a CSV file
of the same named numeric columns, with the addition of a label
column. Each row in the MODELFILE represents a cluster centroid for
the K clusters output by the train command.

If no MODELFILE is provided then the model is read from stdin,
allowing the output of the TRAIN command to be piped to the input of
this predict command.

The output to STDOUT is the same DATAFILE file with an additional
column, named label, as the cluster whose centroid from the MODELFILE
is closest in distance to the row's observations.

    """

    # The datafile for predicting from is required.

    df = pd.read_csv(datafile)
    data = df.to_numpy()

    # The modelfile is required and records the centers of clusters. The
    # label field reports the cluster number.

    try:
        centers = pd.read_csv(modelfile)
    except pd.errors.EmptyDataError:
        click.echo("A model is required: consists of centers,label.")
        sys.exit(1)

    # Prepare data.

    centers.sort_values(by="label", inplace=True)
    labels = centers["label"]
    centers = centers.drop(columns="label").to_numpy()

    k, m = centers.shape  # k clusters and m variables.
    n = data.shape[0]  # n observations to be classified.
    distance = np.zeros([n, k])  # Initialise a distance matrix.

    # Calculate distance and assign labels.

    for j, c in enumerate(centers):
        distance[:, j] = np.linalg.norm(data - c, axis=1)
    new_labels = np.argmin(distance, axis=1)

    # Map labels from 0,...,k-1 to the provided label string

    df["label"] = pd.Series(data=new_labels).map(labels.to_dict())

    print(df.to_csv(index=False).strip())


if __name__ == "__main__":
    cli(prog_name="predict")
