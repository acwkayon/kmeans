

from plotnine import ggplot, aes, geom_point
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Given a CSV file which has the label added to it, as in the output from PREDICT for example, generate a plot of the data with the clusters.

# Perhaps utilise ggplot from plotnine

# Special cases for 1 dimension, 2 dimensions and >2 diemnsions (do PCA and choose two most significant components).

df = pd.read_csv('labelled.csv')


# Use a scatter plot for 2 dimensions 
print(ggplot(df) + aes(x="sepal_length", y="sepal_width", color="factor(label)") + geom_point())

# Line of dots for 1 dimension
print(ggplot(df) + aes(x="sepal_length", y="label", color="factor(label)") + geom_point())

# Do PCA to find the two most significant components for > 2 dimensions

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df)
principal_df = pd.DataFrame(data = principal_components, columns = ['PC 1', 'PC 2'])


principal_label_df = pd.concat([principal_df, df[['label']]], axis = 1)

print(ggplot(principal_label_df) + aes(x="PC 1", y="PC 2", color="factor(label)") + geom_point())