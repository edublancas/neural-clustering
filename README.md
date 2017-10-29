# neural-noise

Neural noise project

## Installing requirements

```shell
# clone repo
git clone https://github.com/edublancas/neural-noise

# install command line tools
cd neural-noise
pip install .

# install notebooks dependencies
pip install -r requirements.txt
```

## Running pipeline

```shell
# extract waveforms
get-waveforms config_sample.yaml

# extract noise
get-noise config_sample.yaml

# fit noise model using edward
fit-noise config_sample.yaml
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