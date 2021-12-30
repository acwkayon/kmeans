# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Thursday 2021-12-30 16:44:08 +1100 Graham Williams>
#
# Authors: Anita@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Anita Williams. All rights reserved.

import os
import sys
import click

import pandas as pd
import numpy as np

from mlhub.pkg import get_cmd_cwd

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

    df_std = df.copy()

    # Apply the z-score method.

    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    
    click.echo(df_std.to_csv(index=False))


if __name__ == "__main__":
    cli(prog_name="normalise")
