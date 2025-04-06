#!/bin/bash

eval "$(micromamba shell hook --shell bash)"

# Create the environment
micromamba create -n tapas-upgrade python=3.10.15 -c conda-forge

# Activate the environment
micromamba activate tapas-upgrade

# Install from specific folders (in order)
cd ~/Desktop/colin-pelletier-develop/privacy/libs/tapas-main
pip install .

cd ~/Desktop/colin-pelletier-develop/privacy/libs/private-pgm-master
pip install .

cd ~/Desktop/colin-pelletier-develop/privacy/libs/reprosyn-main
pip install .

# Install general dependencies
pip install scikit-learn==1.5.2
pip install ctgan==0.10.2
pip install click

cd ~/Desktop/colin-pelletier-develop/privacy/libs/synthcity-main
pip install .

pip install --upgrade torch torchvision
pip install --upgrade pandas==1.5.3

cd ~/Desktop/colin-pelletier-develop/privacy/libs/querysnout-main/src/optimized_qbs
pip install .

pip install --upgrade setuptools
pip install hydra-core --upgrade
pip install hydra_colorlog --upgrade
pip install optuna optuna-dashboard
pip install hydra-optuna-sweeper --upgrade
pip install --upgrade optuna 
pip install sdmetrics

echo "Installation completed successfully."
