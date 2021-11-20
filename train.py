import sys
import click

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import KMeans, update, save_animation

from mlhub.pkg import mlpreview


@click.command()
@click.argument("k", type=click.IntRange(2))
@click.option("-i", "--input", default=sys.stdin, type=click.File('r'),
              help="Filename of the CSV file to cluster, or from STDIN.")
@click.option("-o", "--output", default=sys.stdout, type=click.File('w'),
              help="Filename of the CSV file to save model, or to STDOUT.")
@click.option("-m", "--movie", type=click.Path(),
              default="tmppp.mp4",  # tempfile.TemporaryFile(suffix=".mp4"),
              help="Filename of the movie file to save if desired.")
@click.option("--view", is_flag=True, default=False,
              help="Popup a movie viewer to visualise the algorithm.")
def cli(k, input, output, movie, view):
    """Train a kmeans cluster model."""

    try:
        df = pd.read_csv(input)
    except pd.errors.EmptyDataError:
        click.echo("Exiting model training as no data is available.")
        sys.exit(1)
    data = df.to_numpy()

    kmeans = KMeans(k, input_data=data, repeat_times=1, slience=True)
    kmeans.farest_center()
    fig, ax = plt.subplots()
    kmeans.set_ax(ax)
    ani = animation.FuncAnimation(
        fig, update, frames=50, fargs=(kmeans, False,), interval=500)
    # writer = animation.FFMpegWriter(
    #     fps=30, metadata=dict(artist='Me'), bitrate=1800)

    if view:
        save_animation(ani, movie, verbose=False)

    df["label"] = kmeans.labels
    header = ','.join(df.columns)

    if view:
        # the mlpreview will call print(begin+prompt), which includes a '\n'
        mlpreview(movie, begin="", msg=header)
    else:
        print(header)

    # Output the model as centers per label as CSV format.

    labels = range(k)
    centers = pd.DataFrame(kmeans.centers)
    centers["label"] = labels
    click.echo(centers.to_csv(header=False, index=False, float_format='%.2f'))


if __name__ == "__main__":
    cli(prog_name="train")
