### This framework was created during the ETH ZÜRICH Data Science Lab of Spring 2025.

### Contributors:
- Artheme Gauthier-Villars
- Zoé Meier
- Eduard von Bothmer

### Challenge Giver:
- Dr. David Froelicher / Swisscom

### Academic Supervisors:
- Prof. Fanny YangMartinez / ETH AI Center, ETH ZÜRICH
- Dr. Javier Abad Martinez / ETH AI Center, ETH ZÜRICH

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

```
└── 📁Benchmarking-of-Tabular-Synthetic-Data-Generation
    └── 📁privacy
        └── __init__.py
        └── 📁data "Folder for input data"
        └── 📁libs "Folder for dependencies installations"
        └── pyproject.toml
        └── requirements.txt
        └── 📁results "Folder for storing results"
        └── 📁utils "Folder containing the pipeline code"
            └── __init__.py
            └── basegenerators.py
            └── baseline_attack.py
            └── benchmark_metric.py
            └── benchmark_pipeline.py
            └── 📁customgenerators
                └── __init__.py
                └── sdvgenerator.py
                └── synthcitygenerator.py
            └── my_attack.py
            └── my_feature.py
            └── plotting.py
            └── privpgdgenerator.py
            └── 📁tapas
        └── run_pipeline.ipynb "Notebook to run the pipeline"
    └── __init__.py
    └── .gitignore
    └── Benchmarking-of-Tabular-Synthetic-Data-Generation.code-workspace
    └── install_lab_env_privpgd.sh
    └── install_lab_env.sh
    └── LICENSE
    └── README.md
    └── requirements.txt
```
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->


# Installation
There's two installation scripts, one for the stable base setup covering all the packages for Synthcity, SDV and Reprosyn generators (install_lab_env.sh) and one additionally for privpgd (install_lab_env_privpgd.sh), that sometimes has issues with resolving dependencies depending on the system you're running it from. Unless you're running privpgd generators, please use the install_lab_env.sh script to set up your environment. In case the script doesnt work, please try manual installation of the steps in the exact same order with either micromanba or conda.
