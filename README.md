[![GitHub](https://img.shields.io/github/license/acwkayon/kmeans?label=License&style=plastic)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/acwkayon/kmeans/Build?label=Build&style=plastic)](https://github.com/acwkayon/kmeans/actions/workflows/build.yml)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/acwkayon/kmeans?label=Tag&style=plastic)](https://github.com/acwkayon/kmeans/releases/tag/v1.0.0)

# K-Means Cluster Analysis

Visit <https://survivor.togaware.com/mlhub/kmeans.html>. The k-means
package is documented in detail in the MLHub Survival Guide.

This MLHub (https://mlhub.ai) package provides a demonstration and
command line tools for kmeans cluster analysis. Kmeans will identify
"natural" groups in a population dataset. This package provides the
*demo* command to demonstrate k-means in action and provides command
line tools to perform a cluster analysis, including animations.

Source code: <https://github.com/acwkayon/kmeans>.

Initial implementation :https://github.com/davecatmeow/showcase-demo

Initial implementation by Gefei Shan with new commands and MLHub
conformation by Anita Williams.

## Quick Start

Some simple examples:

```bash
wget https://raw.githubusercontent.com/acwkayon/kmeans/master/iris.csv
ml train kmeans 3 iris.csv
ml train kmeans 3 iris.csv --view
ml train kmeans 3 --view --movie iris.mp4 iris.csv | ml predict kmeans iris.csv | ml visualise kmeans
```
