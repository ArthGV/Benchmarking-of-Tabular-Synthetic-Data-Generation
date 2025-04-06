"""
Extension of the TAPAS library to incorporate generators from the synthcity library.
"""
import os
from tapas.datasets.dataset import TabularDataset
from tapas.generators import Generator, Raw

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


class RawGenerator(Raw):
    """
    Simple wrapper to make the Raw generator work with required config hyperparameters.
    """
    def __init__(self, label, random_state):
        super().__init__()
        self.num_calls = 0
        self.prev_ds = None
    
    def __call__(self, dataset, num_samples, random_state = None):
        self.num_calls+=1
        if len(dataset.data)==1000:
            self.prev_ds = dataset
        else:
            dataset = self.prev_ds

        self.fit(dataset)
        return self.generate(num_samples, random_state = random_state)


class SynthcityGenerator(Generator):
    """
    A wrapper for synthticity plugins.
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

    def fit(self, dataset):
        """Fitting does nothing, as we don't yet know the output size."""
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'
        self.tabular_dataset = dataset
        self.trained = True
        # TODO add metadata properly to support different synthcity plugins
        if self.plugin_name == 'ddpm' or self.plugin_name == 'ctgan':
            self.generator_kwargs['metadata'] = dataset.description.schema
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
                file.write(str(model.model.loss_history['loss'].to_dict())+'\n')

        return TabularDataset(output, self.tabular_dataset.description)

    @property
    def label(self):
        """Cherry on top."""
        return self._label