### This framework was created during the ETH ZÜRICH Data Science Lab of Spring 2025.

### Contributors:
- Artheme Gauthier-Villars
- Zoe Meier
- Eduard von Bothmer

### Challenge Giver:
- XXX / Swisscom

### Academic Supervisors:
- XXX / ETH ZÜRICH

# Defender-Attacker Framework for Evaluating Privacy Risks in Synthetic Data Sharing

This repository implements the experiments and methodology on evaluating privacy risks using a defender/attacker framework in synthetic data sharing. The focus is on quantifying information leakage through Membership Inference Attacks (MIAs).

# Overview

We propose a formalized framework to evaluate the privacy risks associated with releasing synthetic datasets. The core idea is to model the interaction between:
	•	Defender: Owns a private dataset D and generates synthetic data D_s using a known generative model.
	•	Attacker: Aims to infer whether a specific target record x_T belongs to the private dataset D, given auxiliary public data and knowledge of the generator architecture.

We define two main attack settings:
	•	Prior Attack: Uses only the publicly available Auxiliary dataset 𝒟̃ to infer membership. This represents a baseline risk without any data release.
	•	MIA Attack (Membership Inference Attack): Uses the released synthetic data D_s, auxiliary dataset 𝒟̃, and generator metadata to estimate the membership status of x_T.

# Repository Structure

# Installation
#TODO
micromamba create -n tapas-upgrade python=3.10.15 -c conda-forge
micromamba activate tapas-upgrade
pip install . #from tapas-main
pip install . # from private-pgm-master
pip install . #from reprosyn-main
pip install scikit-learn==1.5.2
pip install ctgan==0.10.2
pip install click
pip install . # from synthctiy-main
pip install --upgrade torch torchvision
pip install --upgrade pandas==1.5.3
cd querysnout-main/src/optimized_qbs
pip install .
pip install --upgrade setuptools
pip install hydra-core --upgrade
pip install hydra_colorlog --upgrade
pip install optuna optuna-dashboard
pip install hydra-optuna-sweeper --upgrade
pip install --upgrade optuna
pip install sdmetrics
