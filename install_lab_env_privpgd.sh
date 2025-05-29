#!/bin/bash

# This is the script for the environment that also works with Privpgd. Sometimes the dependencies cant get resolved,
# try manual installation of the packages in the same order in that case.
pip cache purge

ENV_NAME="lab_env_privpgd"

# Delete existing environment if it exists
if micromamba env list | grep -q "$ENV_NAME"; then
    echo "⚠️ Environment $ENV_NAME already exists. Removing..."
    micromamba remove -y -n "$ENV_NAME" --all
fi

# Create fresh environment
echo "🚀 Creating environment $ENV_NAME..."
micromamba create -y -n "$ENV_NAME" python=3.10.15 -c conda-forge

mamba_run() {
    micromamba run -n "$ENV_NAME" "$@"
}

# Absolute path to repo root
REPO_ROOT=$(pwd)


cd "$REPO_ROOT/privacy/libs/tapas-main"
mamba_run pip install .

cd "$REPO_ROOT/privacy/libs/private-pgm-master"
mamba_run pip install .

cd "$REPO_ROOT/privacy/libs/reprosyn-main"
mamba_run pip install .

mamba_run pip install scikit-learn==1.5.2 ctgan==0.10.2 click
mamba_run pip install fastcore==1.5.29


cd "$REPO_ROOT/privacy/libs/synthcity-main"
mamba_run pip install .

mamba_run pip install --upgrade torch torchvision
mamba_run pip install --upgrade pandas==1.5.3


cd "$REPO_ROOT/privacy/libs/querysnout-main/src/optimized_qbs"
mamba_run pip install .

mamba_run pip install --upgrade setuptools
mamba_run pip install hydra-core --upgrade
mamba_run pip install hydra_colorlog --upgrade
mamba_run pip install optuna optuna-dashboard
mamba_run pip install hydra-optuna-sweeper --upgrade
mamba_run pip install --upgrade optuna
mamba_run pip install sdmetrics

mamba_run pip install sdv
mamba_run pip install

mamba_run mamba install fastcore=1.5.29
mamba_run pip install gower
mamba_run pip install tf-keras
mamba_run pip install pgmpy==0.1.19
mamba_run pip install "xgboost<1.7"

#for sinthcity
mamba run pip install "transformers==4.41.0" "accelerate==1.7.0"

#for privpgd
cd "$REPO_ROOT/privacy/libs/private-pgd"
mamba_run pip install .

mamba_run pip install fastcore==1.5.29
mamba_run pip install --upgrade torch torchvision

echo "✅ Fresh installation of $ENV_NAME completed!"
