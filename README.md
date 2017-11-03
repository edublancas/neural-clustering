# neural-noise

Neural noise project

## Structure

* notebooks/ - Jupyter notebooks
* src/ - Source code for the package that implementsall the functions
* yass_config/ - Confgiguration files for the YASS package
* config.yaml - Project configuration file

## Installing requirements

```shell
# clone repo
git clone https://github.com/edublancas/neural-noise

# install command line tools and other functions
cd neural-noise
pip install .

# install notebooks/ dependencies
pip install -r requirements.txt
```

## Running pipeline

```shell
# run yass pipeline to process neural data
run_yass yass_config/local_100k.yaml config.yaml
```


## Visualizing results

```shell
jupyter notebook
```


## Other requirements

tensorflow: https://github.com/lakshayg/tensorflow-build
graphviz

## Project repository template

https://github.com/akucukelbir/probprog-finalproject
