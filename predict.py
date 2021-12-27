import os
import sys
import click

import pandas as pd
import numpy as np

from mlhub.pkg import get_cmd_cwd

os.chdir(get_cmd_cwd())

@click.command()
@click.argument('modelfile', 
                type=click.File('r'), 
                default=sys.stdin)
# removed help="the model file for predictions, in csv format",
@click.argument('csvfile', 
                type=click.File('r'), 
                default=sys.stdin)
# removed  help="the input data file, in csv format",
@click.option('-o', '--output', 
            default=sys.stdout,type=click.File('w'), 
            help="Save the output predictions to file.")
def cli(modelfile, csvfile, output):
   

    try:
        df = pd.read_csv(csvfile)
    except pd.errors.EmptyDataError:
        click.echo("Quitting as CSV file has no data available.")
        sys.exit(1)
    data = df.to_numpy()


    df_centers = pd.read_csv(modelfile, header=1) # modelfile, the centers of clusters
    #df_data = pd.read_csv(csvfile) # might not need to read twice
    
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
    df_labels = pd.Series(data=new_labels).map(label_index.to_dict()) # map labels from 0,...,k-1 to the provided label string

    df["label"] = df_labels

    origin_out = sys.stdout
    # redirect the output
    sys.stdout = output

    print(df.to_csv(index = False).strip())


# copying from train.py, not sure about this yet:
if __name__ == "__main__":
    cli(prog_name="predict")
