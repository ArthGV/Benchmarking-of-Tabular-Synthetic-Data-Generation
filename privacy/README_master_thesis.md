# Privacy-utility evaluation benchmark

## Intro
This code is based on 4 main libraries:
* [Hydra](https://hydra.cc/): to manage the experiments. All config files in the `./configs/` directory are defined based on hydra, allowing inheritance and composition between configs.
* [Optuna](https://optuna.org/): to perform hyperparameter optimization. In particular, we use the [hydra optuna sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/)
* [Synthcity](https://synthcity.readthedocs.io/en/latest/): provides access to a wide range of synthetic tabular data generators and evaluation metrics
* [TAPAS](https://tapas-privacy.readthedocs.io/en/latest/index.html): threat modeling and privacy risk estimations.


## Important scripts
The pipeline presented in the thesis is not implemented as a single script.   
* Classifier optimization: use the script `./run_classifier_hpo.py`. Provide the corresponding config file from `configs/classifier_hpo` to execute the script.
* Generator optimization: see examples in  `./scripts/run_generator_hpo.py`.
* Utility evalution: the uti1lity evaluation is done in the `./notebooks/evaluation_[dataset]_[model].ipynb` notebooks.
* Privacy evalaution: see examples in `./scripts/run_privacy_evaluation.sh`

Further analysis are realized in the `./notebooks/` directory. The notebooks are self-explanatory, and each notebook contains a small description.

## Repository structure
* `./configs/` hydra configuration files. 
* `data/`: contains all data files as csv. To extract the Adult dataset as a csv, please check the README inside this folder.
* `experiments/`: contains the results of hyperparameters optimization and privacy evaluation pipelines.
* `generated/`: contains the processed pipelines outputs, allowing easier analysis.
* `libs/`: contains the updated privacy libraries and synthcity. 
* `notebooks/`: contains all scripts for analysis: data exploratinon of the datasets, evaluation of individual datasets quality, privacy and utiltiy, hyperparameter optization results and generators comparison for the final results preentation. It also contains the code to find vulnerable records.
* `scripts/`: the script files contains example commands to run an experiment, with or without hydra multirun.
* `tools/`: self-defined library. It contains the functions to run hyperparameter optimization, synthetic data evaluation and the extensions to the TAPAS framework.


## Installation
Please follow the instructions below in the exact same order they are presented:
* micromamba create -n tapas-upgrade python=3.10.15 -c conda-forge
* micromamba activate tapas-upgrade
* pip install . #from tapas-main
* pip install . # from private-pgm-master
* pip install . #from reprosyn-main
* pip install scikit-learn==1.5.2
* pip install ctgan==0.10.2
* pip install click
* pip install . # from synthctiy-main
* pip install --upgrade torch torchvision
* pip install --upgrade pandas==1.5.3
* cd querysnout-main/src/optimized_qbs
* pip install .
* pip install --upgrade setuptools
* pip install hydra-core --upgrade
* pip install hydra_colorlog --upgrade
* pip install optuna optuna-dashboard
* pip install hydra-optuna-sweeper --upgrade
* pip install --upgrade optuna 
* pip install sdmetrics