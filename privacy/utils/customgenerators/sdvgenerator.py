"""
Extensions of our base generator class (BenchmarkGenerator) for sdv generators
"""
import os
import inspect
import re
from tapas.datasets.dataset import TabularDataset
from utils.basegenerators import BenchmarkGenerator
from sdv.metadata import Metadata


class SDVGenerator(BenchmarkGenerator):
    """A wrapper for sdv generators for our custom BenchmarkGenerator class"""
    def __init__(self, sdv_class, label=None, **kwargs):
        self.dataset = None
        self.sdv_class = sdv_class
        self.generator_kwargs = kwargs
        self.trained = False
        self._label = label or str(sdv_class)
        self.sdv_metadata = None
        self.sdv_data = None
        super().__init__(self.label, **self.generator_kwargs)



    def fit(self, dataset):
        """Fitting does nothing, as we don't yet know the output size."""
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'

        self.dataset = dataset
        self.sdv_data = dataset.data
        self.set_sdv_metadata(dataset.description.schema)
        self.trained = True


    def set_sdv_metadata(self, dataset_schema):
        # transform tapas schema into sdv metadata
        col_dic = {}
        for col in dataset_schema:

            name = col['name']
            dtype = col['type']
            representation = col['representation']
            col_dtype_dic = {}

            if representation == "Integer" or representation == "integer":
                col_dtype_dic["sdtype"] = "numerical"
                col_dtype_dic["computer_representation"] = "Int32"
            elif representation == "Number" or representation == "number":
                col_dtype_dic["sdtype"] = "numerical"
                col_dtype_dic["computer_representation"] = "Float"
            elif dtype == "finite":
                col_dtype_dic["sdtype"] = "categorical"
            col_dic[name] = col_dtype_dic
        self.sdv_metadata = Metadata.load_from_dict({"columns": col_dic}, single_table_name=self.dataset.label)



    def generate(self, num_samples):
        """Instantiate a SDV model, run it, and return output."""
        assert self.trained, "No dataset provided to generator."
        synthesizer = self.sdv_class(
            metadata=self.sdv_metadata,
            **self.generator_kwargs
        )

        synthesizer.fit(
            data=self.sdv_data
        )

        synthetic_data = synthesizer.sample(
            num_rows= num_samples
        )

        return TabularDataset(synthetic_data, self.dataset.description)

    @property
    def label(self):
        """Cherry on top."""
        return self._label