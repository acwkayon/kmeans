# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Thu 2021-12-30 11:14:53 -0500 Anita Williams>
#
# Authors: Anita@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Anita Williams. All rights reserved.

import os
import sys
import click

import pandas as pd

from mlhub.pkg import get_cmd_cwd

# Constants

ROUND = 3  # Number of decimal points to round to.

# Ensure paths are relative to the user's cwd.

os.chdir(get_cmd_cwd())


@click.command()
@click.argument('csvfile',
                type=click.File('r'),
                default=sys.stdin)
# help="the input data file, in csv format"
def cli(csvfile):

    # Quit if no data to read.

    try:
        df = pd.read_csv(csvfile)
    except pd.errors.EmptyDataError:
        click.echo("No data available.")
        sys.exit(1)

    # Apply the z-score method
    # https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475

    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    click.echo(df.round(ROUND).to_csv(index=False))


if __name__ == "__main__":
    cli(prog_name="normalise")
