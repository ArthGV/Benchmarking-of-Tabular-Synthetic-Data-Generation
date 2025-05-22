"""
Extensions of our base generator class (BenchmarkGenerator) for private-pgd (Particle Gradient Descent) methods
from (https://github.com/jaabmar/private-pgd/tree/master)
"""
import os
import inspect
import re
import pandas as pd
import numpy as np
from tapas.datasets.dataset import TabularDataset
from utils.basegenerators import BenchmarkGenerator

from pathlib import Path
import sys
project_root = Path().resolve()
pgd_path = project_root / "libs/private-pgd/src"
sys.path.append(str(pgd_path))
from inference.dataset import Dataset
from inference.domain import Domain

pgd_path = project_root / "libs/private-pgd/data"
sys.path.append(str(pgd_path))
from data_handler import DataHandler
from mechanisms.kway import KWay

pgd_path = project_root / "libs/private-pgd/examples"
sys.path.append(str(pgd_path))
from utils_examples import flatten_dict
from mechanisms.utils_mechanisms import generate_all_kway_workload


DEFAULT_EPSILON = 2.5
DEFAULT_DELTA = 1e-5


class PrivpgdGenerator(BenchmarkGenerator):
    """A wrapper for privpgd generators for our custom BenchmarkGenerator class"""
    def __init__(self, generation_engine, mechanism, label=None, num_bins= 35, deg_workload = 2, **kwargs):
        self.tabular_dataset = None
        self.data = None
        self.transformed_dataset = None
        self.all_k_way_workload = None
        self.generation_engine = generation_engine
        self.mechanism = mechanism
        self.num_bins = num_bins
        self.generator_kwargs = kwargs
        self.deg_workload = deg_workload
        self.trained = False
        self._label = label or str(sdv_class)
        super().__init__(self.label, **{**{"num_bins": self.num_bins},**{"deg_workload": self.deg_workload}, **self.generator_kwargs})



    def fit(self, dataset):
        """Fitting does nothing, as we don't yet know the output size."""
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'

        self.tabular_dataset = dataset
        datahandler = DataHandler(dataset.data.copy().reset_index(drop=True))
        df_processed, domain, self.inverse_mapping = datahandler.forward(k=self.num_bins)

        self.data = Dataset(df=df_processed, domain=Domain.fromdict(domain))
        self.all_k_way_workload = generate_all_kway_workload(data=self.data, degree=self.deg_workload)

        self.trained = True

    @staticmethod
    def backward_map(inverse_mapping, df_transformed):
        df_original = pd.DataFrame()

        for column in df_transformed.columns:
            if isinstance(inverse_mapping[column], pd.Index) or isinstance(inverse_mapping[column], np.ndarray):
                df_original[column] = inverse_mapping[column][df_transformed[column]].values
            else:
                bin_medians = inverse_mapping[column]
                df_original[column] = df_transformed[column].map(
                    lambda x: bin_medians[x]
                    if (len(bin_medians) > x >= 0)
                    else np.nan
                )

        return df_original


    def generate(self, num_samples):
        """Instantiate a SDV model, run it, and return output."""
        assert self.trained, "No dataset provided to generator."

        self.generator_kwargs["n_particles"] = num_samples

        gen_engine_instance = self.generation_engine(
            domain=self.data.domain,
            hp=self.generator_kwargs)

        mechanism = KWay(
            epsilon=self.generator_kwargs.get("epsilon", DEFAULT_EPSILON),
            delta=self.generator_kwargs.get("delta", DEFAULT_DELTA),
            degree=self.deg_workload,
            bounded=True)


        synth, loss = mechanism.run(
            data=self.data,
            workload=self.all_k_way_workload,
            engine=gen_engine_instance,
        )

        return TabularDataset(self.backward_map(self.inverse_mapping, synth.df),self.tabular_dataset.description)




    @property
    def label(self):
        """Cherry on top."""
        return self._label
