"""
Extension of the TAPAS library to incorporate generators from other libraries and provide the functionality to read out parameters
"""
import os
import inspect
import re
from abc import abstractmethod
from tapas.datasets.dataset import TabularDataset
from tapas.generators import Generator, Raw
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

class BenchmarkGenerator(Generator):
    def __init__(self, label, **args):
        super().__init__()
        self._label_ = label
        self.params = args
        
    @abstractmethod
    def fit(self,dataset, **kwargs):
        pass

    @abstractmethod
    def generate(self, num_samples, random_state=None):
        pass

    @staticmethod
    def _format_hyperparameters_for_filename(hyperparams):
        parts = []
        for key, value in sorted(hyperparams.items()):
            if isinstance(value, type):
                val_str = value.__name__
            elif hasattr(value, '__name__'):
                val_str = value.__name__
            else:
                val_str = str(value)
            val_str = re.sub(r'[^a-zA-Z0-9_.-]', '', val_str)
            parts.append(f"{key}={val_str}")
        return "_".join(parts)


    def __repr__(self):
        hyperparameters = self._format_hyperparameters_for_filename(self.params)
        # if hyperparameters empty return only the label
        if not hyperparameters:
            return self._label_
        return f"{self._label_}_{hyperparameters}"

    @property
    def parameters(self):
        return self.params



class ReprosynGenerator(BenchmarkGenerator):
    """A wrapper for reprosyn objects for our custom BenchmarkGenerator class"""

    def __init__(self, reprosyn_class, label=None, **kwargs):
        self.reprosyn_class = reprosyn_class
        self.generator_kwargs = kwargs
        self.reprosyn_metadata = []
        self.trained = False
        self._label = label or str(reprosyn_class)
        super().__init__(self.label, **self.generator_kwargs)

    def fit(self, dataset):
        """Fitting does nothing, as we don't yet know the output size."""
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'
        self.dataset = dataset
        self.categorical_columns = dataset.description.one_hot_cols # used to convert numpy strings to python strings
        self.set_reprosyn_metadata(dataset.description.schema)
        self.trained = True

    def set_reprosyn_metadata(self, dataset_schema):
        """Convert a dataset schema to reprosyn compatible metadata.

        This step is required as reprosyn expects a different schema than the one
        used in TAPAS, with less complex type definitions. Using invalid reprosyn types
        will results in the columns being ignored.
        """
        self.reprosyn_metadata = dataset_schema.copy()
        for col_description in self.reprosyn_metadata:
            # Reprosyn does not take multiple types into account.
            if 'countable' in col_description['type']:
                col_description['type'] = 'Integer'
                col_description['representation'] = 'number'
            elif col_description['type'] == 'real':
                col_description['type'] = 'Float'

    def generate(self, num_samples):
        """Instantiate a reprosyn model, run it, and return output."""
        assert self.trained, "No dataset provided to generator."
        model = self.reprosyn_class(
            dataset=self.dataset.data,
            metadata=self.reprosyn_metadata,
            size=num_samples,
            **self.generator_kwargs,
        )
        model.run()
        output = model.output.copy()
        output[self.categorical_columns] = output[self.categorical_columns].astype('category').astype(str)
        return TabularDataset(output, self.dataset.description)

    @property
    def label(self):
        """Cherry on top."""
        return self._label


class Raw(BenchmarkGenerator):
    """Wrapper for BenchmarkGenerator thatsimply samples from the real data."""
    def __init__(self):
        super().__init__("Raw")

    def fit(self, dataset):
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples=None, random_state=None):
        if self.trained:
            if num_samples is None:
                return self.dataset
            return self.dataset.sample(num_samples, random_state=random_state)
        else:
            raise RuntimeError("No dataset provided to generator")

    def __call__(self, dataset, num_samples, random_state=None):
        self.fit(dataset)
        return self.generate(num_samples, random_state=random_state)

    @property
    def label(self):
        return "Raw"




class SynthcityGenerator(BenchmarkGenerator):
    """
    A wrapper for synthticity plugins for custom BenchmarkGenerator.
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
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'
        self.tabular_dataset = dataset
        self.trained = True
        # TODO add metadata properly to support different synthcity plugins
        '''
        if self.plugin_name == 'ddpm' or self.plugin_name == 'ctgan':
            self.generator_kwargs['metadata'] = dataset.description.schema
        '''
        if self.plugin_name == 'ddpm':
            self.generator_kwargs['metadata'] = dataset.description.schema
        # zoe: add metadata for other plugins ?

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

# -----------------------------------------------------------------------------------------------------------------#
# MASTER THESIS STUFF
# -----------------------------------------------------------------------------------------------------------------#
'''

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
        """
        if self.plugin_name == 'ddpm' or self.plugin_name == 'ctgan':
            self.generator_kwargs['metadata'] = dataset.description.schema
        """
        if self.plugin_name == 'ddpm':
            self.generator_kwargs['metadata'] = dataset.description.schema
        # zoe: add metadata for other plugins ? 
        

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


'''

class SDVGenerator(Generator):
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