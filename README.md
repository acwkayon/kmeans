# showcase-demo
A animation demo for k-means, idea from [mlhub's model](https://github.com/mlhubber/mlhub/issues/140).

### Usage
* Suitable for Ubuntu system, to install mlhub python package:

```
$ pip3 install mlhub
```

* To install and configure the package(if have previous version, please `ml remove` first):

```
$ ml install	davecatmeow/showcase-demo
$ ml configure	kmeans
```

### Demonstration

```
$ ml demo	kmeans
```

### Train model on given data

```
$ ml train kmeans [k] [input.csv] -o [output.csv]
```
or
```
$ ml train kmeans [k] [input.csv] > [output.csv]
```


With input dataset `n` lines and `f` features, the output will be:
 * `n+k` lines where the addition `k` lines represents the `k` centers of clusters
 * `f+1` columns where the addition column represents labels of data

### Predict data on given model

```
ml predict kmeans [options] [modelfile] <csvfile>
     -o <file.csv>   --output=<file.csv>       Save the output predictions to file.
```

