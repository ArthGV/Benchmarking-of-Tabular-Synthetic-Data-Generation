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