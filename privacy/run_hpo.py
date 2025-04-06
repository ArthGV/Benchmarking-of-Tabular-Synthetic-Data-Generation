# stdlib
import os
os.environ['OMP_PATH'] = '/opt/homebrew/Cellar/libomp/19.1.3/include' # run XBGoost on Mac M1
import warnings
import time
import random
import timeit
import sys
sys.path.append('libs/MIA-synthetic-main') # required for using CQBS query extractor

# essentials
import pandas as pd
import numpy as np
import torch

# script config
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="./configs/generators_hpo")
def run_hpo(cfg):
    start = timeit.default_timer()
    #
    #   Setup
    #
    print(OmegaConf.to_yaml(cfg))
    
    # set cuda using environment variable before importing synthcity
    os.environ["SYNTHCITY_DEVICE"] = str(cfg.device)
    from synthcity.utils.constants import DEVICE
    # important to import after setting device!
    from tools.hpo.optimizer import GeneratorHPOPipeline

    # reproducibility
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    if str(cfg.device) == 'cpu':
        warnings.warn("Running on CPU.")
    else:
        print(f"Running on device: {DEVICE}")

    #
    #   Evaluation
    #
    pipeline = GeneratorHPOPipeline(cfg)
    score_large = pipeline.evaluate_generator(is_small=False)[0]
    score_small, std_small = pipeline.evaluate_generator(is_small=True)
    score_authenticity = pipeline.evaluate_authenticity()
    
    score_difference = score_large-score_small

    # export all results that are not part of the objective

    logger = pd.DataFrame([{
        'score_large': score_large, 
        'score_small': score_small,
        'std_small': std_small,
        'score_authenticity': score_authenticity, 
        'score_difference': score_difference}])
    exp_dir = HydraConfig.get().runtime.output_dir
    print("Exporting results to:", exp_dir)
    logger.to_csv(f"{exp_dir}/log.csv", index=False)

    stop = timeit.default_timer()
    print(f"******* Time elapsed in seconds: {stop - start}")

    if cfg.hpo_objective=="utility_large":
        print("Returning score large")
        return score_large
    elif cfg.hpo_objective==["utility_large", "authenticity"]:
        print("Returning score large and score authenticity")
        # return (1-cfg.privacy_weight)*score_large, cfg.privacy_weight*score_authenticity
        return score_large, score_authenticity
    elif cfg.hpo_objective==["utility_large", "dcr"]:
        score_dcr = pipeline.evaluate_dcr()
        return score_large, score_dcr


if __name__ == "__main__":
    run_hpo()