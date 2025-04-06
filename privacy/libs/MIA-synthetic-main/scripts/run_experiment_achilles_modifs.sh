#!/bin/bash

python Achilles_modifs.py --path_to_data='./data/census_header.csv' \
                --path_to_metadata='./data/2011 Census Microdata Teaching Discretized.json' \
                --target_record_id=44444 \
                --output_dir='./results_experiments' \
                --name_generator='BAYNET' \
                --n_aux=50000 \
                --n_test=25000 \
                --n_original=1000 \
                --n_synthetic=1000 \
                --n_pos_train=10 \
                --n_pos_test=10 \
                --seed=123456789 \
                --generate_datasets=False \
                --compute_utility=False