# stdlib
import random

# tapas
import tapas

# SDMetrics
from sdmetrics.reports.single_table import QualityReport

# third-party
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def convert_metadata_to_sdm_format(metadata: list[dict]) -> dict:
    sdm_metadata = {'columns': {}}

    for col in metadata:
        if col['type'] == 'finite':
            sdm_metadata['columns'][col['name']] = {'sdtype': 'categorical'}
        elif col['type'] == 'Integer':
            sdm_metadata['columns'][col['name']] = {'sdtype': 'numerical'}
        elif col['type'] == 'Float':
            sdm_metadata['columns'][col['name']] = {'sdtype': 'numerical', 'compute_representation': 'Float'}

    return sdm_metadata