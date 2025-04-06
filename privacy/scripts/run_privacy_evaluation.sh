#!/bin/bash

#
# TabDDPM: adults in two parts, each half on a different gpu
#
python run_auxiliary_blackbox_mia_attack.py --config-name replicate_achilles_heel_tabddpm_adult threat_model.target_record_indice=39435,36838,6066,1678,27566 device=cuda:0 -m
python run_auxiliary_blackbox_mia_attack.py --config-name replicate_achilles_heel_tabddpm_adult threat_model.target_record_indice=43082,16249,14731,5918,35988 device=cuda:1 -m

#
# List of random records
#

# adult
# 4036,34999,29601,19846,19581,38827,3886,31536,9110,4258
python run_auxiliary_blackbox_mia_attack.py --config-name replicate_achilles_heel_tabddpm_adult threat_model.target_record_indice=4036,34999,29601,19846,19581,38827,3886,31536,9110,4258 device=cuda:0 -m
python run_auxiliary_blackbox_mia_attack.py --config-name replicate_achilles_heel_privbayes_adult threat_model.target_record_indice=4036,34999,29601,19846,19581,38827,3886,31536,9110,4258 -m

# census
# 50849,440954,372936,250047,246706,489178,48966,397319,114785,53656
python run_auxiliary_blackbox_mia_attack.py --config-name replicate_achilles_heel_tabddpm_census threat_model.target_record_indice=50849,440954,372936,250047,246706,489178,48966,397319,114785,53656 device=cuda:1 -m
python run_auxiliary_blackbox_mia_attack.py --config-name replicate_achilles_heel_baynet_census threat_model.target_record_indice=50849,440954,372936,250047,246706,489178,48966,397319,114785,53656 -m