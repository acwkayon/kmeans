from sklearn import datasets
import matplotlib as plt
from utils import KMeans

if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data
    iris_labels = iris.target
    fig = plt.figure()
    ax = fig.add_subplot()
    kmeans = KMeans(3, input_data=data)
