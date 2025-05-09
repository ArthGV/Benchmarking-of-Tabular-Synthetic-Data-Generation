"""
Extensions of our base generator class (BenchmarkGenerator) for sdv generators
"""
import os
import inspect
import re
from tapas.datasets.dataset import TabularDataset
from .basegenerators import BenchmarkGenerator


class SDVGenerator(BenchmarkGenerator):
#TODO!!!!!!!

    def __init__(self, sdv_class, label=None, **kwargs):
        self.sdv_class = sdv_class
        self.generator_kwargs = kwargs
        self.trained = False
        self._label = label or str(sdv_class)

    def fit(self, dataset):
        """Fitting does nothing, as we don't yet know the output size."""

        #TO DO: transform into df & infer metadata 
        '''
        ....
        self.dataset = dataset
        self.metadata = metadata
        self.trained = True
        '''

    def generate(self, num_samples):
        #TO DO; 
        """Instantiate a SDV model, run it, and return output."""
        assert self.trained, "No dataset provided to generator."
        model = self.sdv_class(
            metadata=self.reprosyn_metadata,
            **self.generator_kwargs
        )
        model.fit(
            data=self.dataset
        )

        synthetic_data = model.sample(
            num_rows=self.num_samples
        )

       
        #TO DO: bring into right format......
        output[self.categorical_columns] = output[self.categorical_columns].astype('category').astype(str)
        return TabularDataset(output, self.dataset.description)

    @property
    def label(self):
        """Cherry on top."""
        return self._label