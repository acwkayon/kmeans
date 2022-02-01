import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point
from sklearn.decomposition import PCA

def visualise(df):
      if len(df.columns) == 2:
            header_1_col = list(df.columns.values)
            return (ggplot(df) +
            aes(x=header_1_col[0], y="label", color="factor(label)") +
            geom_point())
      if len(df.columns) == 3:
            header_2_col = list(df.columns.values)
            return (ggplot(df) +
            aes(x=header_2_col[0], y=header_2_col[1], color="factor(label)") +
            geom_point())
      else:
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(df)
            principal_df = pd.DataFrame(data=principal_components
                            , columns=['PC 1', 'PC 2']
                            )
            principal_label_df = pd.concat([principal_df, df[['label']]], axis=1)

            return (ggplot(principal_label_df) +
            aes(x="PC 1", y="PC 2", color="factor(label)") +
            geom_point())

# examples for demo-ing

# output of preditc.py using iris.csv

df_multi_col = pd.read_csv('labelled.csv')

# a csv file with one column + a label column

df_1_col = pd.read_csv('labelled_col_1.csv')

# a csv file with two columns + a label column
df_2_col = pd.read_csv('labelled_col_2.csv')

print(visualise(df_1_col))
print(visualise(df_2_col))
print(visualise(df_multi_col))

