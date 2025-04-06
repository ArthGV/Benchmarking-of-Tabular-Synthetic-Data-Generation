"""
Script to run a threat models with:
* Data knowledge: auxiliary
* Generator knowledge: black-box
* Goal: membership inference
"""

# stdlib
import os
os.environ['OMP_PATH'] = '/opt/homebrew/Cellar/libomp/19.1.3/include' # run XBGoost on Mac M1
import warnings
import time
import random
import sys
sys.path.append('libs/MIA-synthetic-main') # required for using CQBS query extractor

# essentials
import numpy as np
from tqdm import tqdm
import torch

# script config
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

# privacy
from tapas.datasets import TabularDataset
from tapas.threat_models import AuxiliaryDataKnowledge, BlackBoxKnowledge, TargetedMIA
from tapas.report import ROCReport, MIAttackReport, EffectiveEpsilonReport
from tapas.generators import ReprosynGenerator

# custom
from tools.utils import (aux_test_split, 
                         export_elapsed_time, 
                         get_class_object_from_fqdn,
                         dump_pickle)


@hydra.main(version_base=None, config_path="./configs/privacy")
def run_pipeline(cfg: DictConfig) -> None:
    #
    #   Setup
    #
    get_relative_path = lambda x: os.path.join(HydraConfig.get().runtime.output_dir, x)
    total_time_start = time.time()
    np_rng = np.random.default_rng(cfg.random_seed)

    # set cuda using environment variable before importing synthcity
    os.environ["SYNTHCITY_DEVICE"] = str(cfg.device)
    from synthcity.utils.constants import DEVICE
    from tools.tapas.generators import SynthcityGenerator

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    if str(cfg.device) == 'cpu':
        warnings.warn("Running on CPU.")
    else:
        print(f"Running on device: {DEVICE}")


    #
    #   Load data
    #
    print("Loading data...")
    dataset = TabularDataset.read(cfg.data.path, label=cfg.data.label)
    dataset = instantiate(cfg.data.tapas_data_processor).process_tapas_tabulardataset(dataset)
    target_record = dataset.get_records([cfg.threat_model.target_record_indice])
    dataset.drop_records([cfg.threat_model.target_record_indice], in_place=True)

    #
    #   Define the threat model
    #
    print("Loading the threat model...")
    try:
        # TODO not supported yet.
        if cfg.threat_model.reload: 
            # load form 
            threat_model = TargetedMIA.load(cfg.threat_model.reload_path)
            print(f"Threat model successfully loaded from {cfg.threat_model.reload_path}.")
        else:
            raise Exception("Don't use cache")
    except Exception:
        print(f"Threat model not found, creating one.")

        #
        #   Data knowledge
        #
        tm_data = cfg.threat_model.data_knowledge
        aux_data, test_data = aux_test_split(dataset, tm_data.aux_data_size, tm_data.test_data_size, np_rng)
        
        data_knowledge = AuxiliaryDataKnowledge(
            dataset,
            aux_data=aux_data,
            test_data=test_data,
            num_training_records=tm_data.real_datasets_size
        )

        #
        #   Generator knowledge
        #
        tm_gen = cfg.threat_model.generator_knowledge
        if 'ReprosynGenerator' in tm_gen.generator._target_:
            # to instantiate a reprosyn object, we need to split the params so that we can recover the reprosyn class
            ReprosynClass = get_class_object_from_fqdn(tm_gen.generator.reprosyn_class)
            gen_params = {k:v for k,v in tm_gen.generator.items() if k != 'reprosyn_class' and k!= '_target_'}
            # the reprosyn objects don't use a "random_sate" but a seed
            gen_params['seed'] = gen_params['random_state']
            gen_params.pop('random_state')
            generator = ReprosynGenerator(ReprosynClass, **gen_params)
        else:
            if tm_gen.generator.label == "Raw" or tm_gen.generator.plugin_name != "ddpm":
                generator = instantiate(tm_gen.generator)
            else:
                # model_params: dict = dict(n_layers_hidden=3, n_units_hidden=256, dropout=0.0)
                gen_params = {k:v for k,v in tm_gen.generator.items() if k != 'reprosyn_class' and k!= '_target_'}

                gen_params['model_params'] = {
                    'n_layers_hidden': gen_params['n_layers_hidden'],
                    'n_units_hidden': gen_params['n_units_hidden'],
                    'dropout': gen_params['dropout']
                }

                # pop the keys
                gen_params.pop('n_layers_hidden')
                gen_params.pop('n_units_hidden')
                gen_params.pop('dropout')
                print("Generator params", gen_params)

                generator = SynthcityGenerator(**gen_params)

        sdg_knowledge = BlackBoxKnowledge(
            generator=generator,
            num_synthetic_records=tm_gen.synth_datasets_size
        )

        #
        #   Attacker goal
        #
        tm_goal = cfg.threat_model.attacker_goal
        threat_model = TargetedMIA(
            attacker_knowledge_data=data_knowledge,
            attacker_knowledge_generator=sdg_knowledge,
            target_record=target_record, 
            iterator_tracker=tqdm,
            generate_pairs=tm_goal.generate_pairs,
            replace_target=tm_goal.replace_target
        )

    #
    #   Generate synthetic datasets
    #
    threat_model_path_str = str(get_relative_path(cfg.threat_model.paths.threat_model_path)) # != reloaded threat model path

    # We create the synthetic datasets and cache them.
    # This is the most time-consuming step.
    atk = cfg.attack
    print("Generate training datasets...")
    training_ds_gen_time_start = time.time()
    threat_model._generate_samples(num_samples=atk.num_training_samples, training=True)
    training_ds_gen_time_elapsed = time.time() - training_ds_gen_time_start
    threat_model.save(threat_model_path_str)

    print("Generate testing datasets...")
    testing_ds_gen_time_start = time.time() 
    threat_model._generate_samples(num_samples=atk.num_testing_samples, training=False)
    testing_ds_gen_time_elapsed = time.time() - testing_ds_gen_time_start
    threat_model.save(threat_model_path_str)
    
    #
    #   Run the attack
    #    
    if 'AchillesHeelAttack' in atk.attack_object._target_ or 'QueryBasedAttackFast' in atk.attack_object._target_:
        # here, we need to instanciate the attack "by hand" if it requires the target record, as 
        # hydra dict only accept primitive types.
        attack_class = get_class_object_from_fqdn(atk.attack_object._target_)
        attack_params = {k:v for k, v in atk.attack_object.items() if k != '_target_'}
        attack_params['target_record'] = target_record
        attack_params['model'] = instantiate(attack_params['model'])
        attack = attack_class(**attack_params)
    else:
        attack = instantiate(atk.attack_object)

    print("Training meta-classifier...")
    meta_classifier_training_time_start = time.time()
    attack.train(threat_model, num_samples=atk.num_training_samples)
    meta_classifier_training_time_elapsed = time.time() - meta_classifier_training_time_start

    print("Testing meta-classifier...")
    meta_classifier_testing_time_start = time.time()
    attack_summary = threat_model.test(attack, num_samples=atk.num_testing_samples)
    meta_classifier_testing_time_elapsed = time.time() - meta_classifier_testing_time_start
    
    threat_model.save(threat_model_path_str)

    #
    #   Export evaluation reports
    #
    print("Exporting metrics...")
    attack_summary.get_metrics().to_csv(get_relative_path(atk.paths.metrics))

    dump_pickle(attack_summary, get_relative_path(atk.paths.summary))
    
    reports = atk.reports
    if 'mia' in reports:
        mia_report = MIAttackReport([attack_summary], metrics=reports.mia.metrics, num_bootstrap=reports.mia.num_bootstrap)
        mia_report.publish(get_relative_path(atk.paths.mia_report))
    
    if 'roc' in reports:
        roc_report = ROCReport([attack_summary], eff_epsilon=reports.roc.eff_epsilon)
        roc_report.publish(get_relative_path(atk.paths.roc_report))

    if 'eff_eps' in reports:
        raise NotImplementedError("Effective Epsilon reports are not implemented yet!")

    total_time_elapsed = time.time() - total_time_start

    export_elapsed_time(training_ds_gen_time_elapsed, testing_ds_gen_time_elapsed, 
                        meta_classifier_training_time_elapsed,meta_classifier_testing_time_elapsed, 
                        total_time_elapsed, get_relative_path(atk.paths.elapsed_time))


if __name__ == '__main__':
    run_pipeline()