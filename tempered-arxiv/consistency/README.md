# Tempered Overfitting README

## Dependencies

Packages can be installed via `requirements.txt` as:
```
pip install -r requirements.txt
```
into the host system or a virtualenv.

We use Python 3 for all experiments.

## Training and testing

We use YACS for configurations. Default configs are sourced from `config.py` and custom configurations
can be be provided at the command line or as a custom config yaml file. An example of a custom
config yaml file is given in `configs/`.

To run the code from the root of `consistency/` we run:
```
PYTHONPATH=. python train.py --config_file /path/to/custom_config.yaml
```
with a custom config, or:
```
PYTHONPATH=. python train.py
```
to use the default settings in `config.py`.

Evaluation is automatically done while training and results are pushed to Weights and Biases dashboards
as specified via `WANDB.ORG` and `WANDB.PROJECT_NAME` config keys in the config.py / custom config file.

wandb API key must be provided in a file called `secrets.py` in the root of `consistency/`, the format
of this file is given as an example in `secrets_example.py`.
