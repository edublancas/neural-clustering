# Neural clustering project

Neural Clustering project.

Machine Learning with Probabilistic Programming, Columbia University, Fall 2017.

## Structure

* notebooks/ - Jupyter notebooks
* src/ - Source code for the package that contains the utility functions
* yass_config/ - Configuration files for the [YASS](https://github.com/paninski-lab/yass) package
* config.yaml - Project configuration file
* count.sh - Script for counting the length of the notebooks

## Installing requirements

```shell
# clone repo
git clone https://github.com/edublancas/neural-clustering

# install command line tools and other functions
cd neural-noise
pip install .

# install notebooks/ dependencies
pip install -r requirements.txt
```

## Step 1: Run YASS pipeline

The first step is to run YASS to generate the necessary
files for the clustering and visualization notebooks.

```shell
# run yass pipeline to process neural data
run_yass yass_config/local_7ch.yaml config.yaml

# this will process the neural recordings to generate
# the training data for the clustering algorithm
```

## Step 2: Run notebooks

Once input files are generated, start `jupyter` and take a look at the
files located in  `notebooks/`

```shell
jupyter notebook
```


## Project repository template

* https://github.com/akucukelbir/probprog-finalproject
