"""
Extensions of our base generator class (BenchmarkGenerator) for synthcity plug ins
"""

import os
import re
from tapas.datasets.dataset import TabularDataset
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from .basegenerators import BenchmarkGenerator


class SynthcityGenerator(BenchmarkGenerator):
    """
    A wrapper for synthcity plugins for our custom BenchmarkGenerator
    """
    def __init__(self, plugin_name, label=None, random_state=42, **kwargs):
        """
        Parameters
        ----------
        plugin_name: str
            The synthcity plugin name
        label: str (default: plugin_name)
            The TAPAS generator label
        kwargs: dict
            The model hyperparameters
        """
        self.plugin_name = plugin_name
        self.generator_kwargs = kwargs
        self.trained = False
        self.random_state = random_state
        self._label = plugin_name if label is None else label
        self.merged_generator_kwargs = {**{"random_state":self.random_state}, **self.generator_kwargs}
        super().__init__(self._label, **self.merged_generator_kwargs)

    def fit(self, dataset):
        """Fitting does nothing, as we don't yet know the output size."""
        df_data = dataset.data

        self.trained = True
        # TODO add metadata properly to support different synthcity plugins
        '''
        if self.plugin_name == 'ddpm' or self.plugin_name == 'ctgan':
            self.generator_kwargs['metadata'] = dataset.description.schema
        '''
        if self.plugin_name == 'ddpm':
            self.generator_kwargs['metadata'] = dataset.description.schema
        # TODO zoe: add metadata for other plugins ?

        # create a synthcity dataloader form the dataset
        self.data_loader = GenericDataLoader(dataset.data, random_state=self.random_state)

    def generate(self, num_samples):
        """Instantiate a reprosyn model, run it, and return output."""
        assert self.trained, "No dataset provided to generator."

        # TODO metadata will likely not need to be provided like that, but simply using the dataset metadata
        model = Plugins().get(self.plugin_name, random_state=self.random_state, **self.generator_kwargs)
        model.fit(self.data_loader)
        output = model.generate(count=num_samples, random_state=self.random_state).dataframe()

        if self.plugin_name == 'ddpm':
            synthcity_device = os.environ.get("SYNTHCITY_DEVICE", "cpu")
            with open(f"./loss_{synthcity_device}.txt", "a") as file:
                file.write(str(model.model.loss_history['loss'].to_dict()) + '\n')

        return TabularDataset(output, self.tabular_dataset.description)

    @property
    def label(self):
        """Cherry on top."""
        return self._label