# MLHub demonstrator and toolkit for kmeans.
#
# Time-stamp: <Sat 2022-02-12 13:13:41 -0800 Anita Williams>
#
# Authors: Anita@togaware.com
# License: General Public License v3 GPLv3
# License: https://www.gnu.org/licenses/gpl-3.0.en.html
# Copyright: (c) Anita Williams. All rights reserved.

import os
import sys
import click

import pandas as pd

from plotnine import ggplot, aes, geom_point, labs, ggsave
from sklearn.decomposition import PCA
from mlhub.pkg import get_cmd_cwd

# Ensure paths are relative to the user's cwd

os.chdir(get_cmd_cwd())


@click.command()
@click.argument('csvfile',
                type=click.File('r'),
                default=sys.stdin)
@click.option("-l", "--label",
              default="label",
              help="identify the header for the column that contains the label")
@click.option("-o", "--output",
              default= "output not given",
              help="Filename of the PNG file to save the plot. If none specificed, plot will print on screen.")


def cli(csvfile, label, output):

      
      # Quit if there is no data to read
      try:
        df = pd.read_csv(csvfile)
      except pd.errors.EmptyDataError:
        click.echo("No data available.")
        sys.exit(1)

      df.rename(columns={label: "label"}, inplace = True)
      
      # Line of dots if data is one dimension
      
      if len(df.columns) == 2:
            # Find the non-label column
            header_1_col = list(df.columns.values)
            header_1_col.remove('label')
            vis_plot = (ggplot(df) +
            aes(x=header_1_col[0], y='label', color="factor(label)")
            + labs(color="Cluster") + geom_point())

            
      
      # Scatter plot if data is two dimensions
      elif len(df.columns) == 3:
            
            # Find the non-label columns
            header_2_col = list(df.columns.values)
            header_2_col.remove('label')
            vis_plot = (ggplot(df) +
            aes(x=header_2_col[0], y=header_2_col[1], color="factor(label)") +
            labs(color="Cluster") + geom_point())
      
      # Perform PCA if there are 2+ dimensions and visualise the 2 most
      # significant components 
      # to do: add some context about PCA for end user
      else:
            pca = PCA(n_components=2)
            
            # Remove the label column for PCA
            df_for_pca = df.drop(['label'], axis=1)
            principal_components = pca.fit_transform(df_for_pca)
            principal_components
            principal_df = pd.DataFrame(data=principal_components
                            , columns=['PC 1', 'PC 2']
                            )
            principal_label_df = pd.concat([principal_df,
                                           df[['label']]],
                                           axis=1)
            vis_plot = (ggplot(principal_label_df) +
            aes(x="PC 1", y="PC 2", color="factor(label)") 
            + labs(color="Cluster")+ geom_point())
      
      
      if output == "output not given":
            print(vis_plot)
      else: 
            ggsave(plot = vis_plot, filename = output + ".png")

if __name__ == "__main__":
    cli(prog_name="visualise")
