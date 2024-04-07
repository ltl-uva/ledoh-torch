# Toy dispersion experiment

## Usage
Pytorch and cuda stuff are not included in the ```requirements.txt``` file (it was used with ```torch 2.2.1+cu118``` and cuda 11.8).

An example configuration file is provided in the repository. Note that Gaussian distribution is N(x; 10, 1). The offset 10 is hardcoded in ```experiment.py```

```
python3 experiment.py -c config.yaml
```

## Wandb output
The min dist per point is uploaded to wandb after dispersion. It is saved as the table `dmin_hist` (as an artefact). See example.ipynb.

The cvar, mind dist are logged as plot (and table). Suppose x1, .., xn are hyperparams, then it creates plots where n - 1 params are fixed and the remaining param can vary.
