# stdlib
import os
os.environ['OMP_PATH'] = '/opt/homebrew/Cellar/libomp/19.1.3/include' # run XBGoost on Mac M1

# script config
import hydra
from tools.synthetic_evaluation.classification_optimizer import ClassificationOptimizer


@hydra.main(version_base=None, config_path="./configs/classifier_hpo")
def run_hpo(cfg):
    clf_opt = ClassificationOptimizer(cfg)
    clf_opt.fit()

    print("model fitted!")

    # we need to optimize hyperparameters based on the cv_score_mean
    cv_scores_mean, _ = clf_opt.evaluate_classifier()

    print("cv_score_mean:", cv_scores_mean)
    return cv_scores_mean


if __name__ == "__main__":
    run_hpo()