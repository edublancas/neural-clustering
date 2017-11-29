# Neural clustering project

Neural Clustering project.

Machine Learning with Probabilistic Programming, Columbia University, Fall 2017.

## Structure

* examples/ - Some files I used to implement models and learn about Edward
* experiments/ - Files I used to experiment with different models and settings
* notebooks/ - Jupyter notebooks
* src/ - Source code for the package that implements the utility functions
* yass_config/ - Configuration files for the [YASS](https://github.com/paninski-lab/yass) package
* config.yaml and server_config.yaml - Project configuration file
* count - Script for counting the length of the notebooks

## Step 1: Install requirements

```shell
# clone repo
git clone https://github.com/edublancas/neural-clustering

# install package
cd neural-noise
pip install .

# install other dependencies
pip install -r requirements.txt
```

## Step 2: Code linting and notebooks length

Checking code passes flake8:

```shell
pytest --flake8
```

Counting jupyter notebooks length:

```
./count
```


## Step 3: YASS setup


## Step 4: Run YASS pipeline

Run YASS to generate the necessary files for the clustering and visualization notebooks.

```shell
# run yass pipeline to process neural data
run_yass yass_config/local_7ch.yaml config.yaml

# this will process the neural recordings to generate
# the training data for the clustering algorithm
```

## Step 5: Run notebooks

Once input files are generated, start `jupyter` and take a look at the
files located in  `notebooks/`

```shell
jupyter notebook
```


## Project repository template

* https://github.com/akucukelbir/probprog-finalproject
