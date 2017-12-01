# Neural clustering

Machine Learning with Probabilistic Programming, Columbia University, Fall 2017.

## Structure

* `examples/` - Some files I used to implement models and learn about Edward
* `examples/dpmm.py` - Truncated DPMM implementation in Edward, many issues arised when implementing it, see docstring at the top of the file for details
* `experiments/` - Files I used to experiment with different models and settings
* `notebooks/` - Jupyter notebooks
* `src/` - Source code for the package that implements the utility functions
* `yass_config/` - Configuration files for the [YASS](https://github.com/paninski-lab/yass) package
* `config.yaml` and server_config.yaml - Project configuration file
* `count` - Script for counting the length of the notebooks
* `test` - Script to check that everything runs smoothly
* `export_presentation` - Exports notebooks for presentation

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


## Step 3: Configuration file

Update `root` in `config.yaml` to a folder in the computer, put the provided `sessions/` folder inside `root`, this is the folder where the output from the pipeline will be stored.

## Step 4: YASS setup

Get raw data files: `7ch.bin` and `geometry.txt`.

Open `yass_config/demo.yaml` and update `root` with the absolute path where `7ch.bin` and `geometry.txt` are located so YASS can load and process the data.

## Step 5: Run YASS pipeline

Run YASS to generate the necessary files for the clustering and visualization notebooks.

```shell
# run yass pipeline to process neural data, this will the entire pipeline
# since there is no way to just run the detection step, but we
# will only use the detected spikes for clustering
run_yass yass_config/demo.yaml config.yaml

# when the command is done there is going to be a folder in {root}/yass
# where root is the root folder in the config.yaml file
```

## Step 6: Run notebooks

Once input files are generated, start `jupyter notebook` and take a look at the
files located in  `notebooks/`

Notebooks overview:

1. Intro - Brief introduction to spike sorting
2. Data loading - Explanation of the data we are working with
3. Model fit and experiments listing - Example of how to fit models, it also includes a listing of the current experiments
4. Model criticism (there is one notebook for GMM and another for DPMM) - Checking convergence (only for GMM) and PPC plots
5. Clustering evaluation - Plots to evaluate the clustering results
6. Conclusions

`notebooks/experiments/`  contains the output for notebooks 4 and 5 for some experiments.

`notebooks/presentation/` contains the notebooks that will be used for the presentation.

## Resources

* [Project template](https://github.com/akucukelbir/probprog-finalproject)
