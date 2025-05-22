"""
Base class for Generators used in Benchmark, extending the Tapas Library to incorporate generator from other libraries
and providing functionality to read out parameters & formatted filename w. parameters for easy hyperparameter tuning of
generators
"""

import os
import re
from abc import ABC, abstractmethod
from tapas.generators import Generator
from tapas.datasets.dataset import TabularDataset

class BenchmarkGenerator(Generator):
    def __init__(self, label, **args):
        super().__init__()
        self._label_ = label
        self.params = args

    @abstractmethod
    def fit(self,dataset,**kwargs):
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

    @staticmethod
    def _format_hyperparameters_for_legend(hyperparams):
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
        return f"({', '.join(parts)})"

    def __repr__(self):
        hyperparameters = self._format_hyperparameters_for_filename(self.params)
        # if hyperparameters empty return only the label
        if not hyperparameters:
            return self._label_
        return f"{self._label_}_{hyperparameters}"

    def get_label_for_legend(self):
        #pretty formatting of generator label and parameters for plotting legends
        hyperparameters = self._format_hyperparameters_for_legend(self.params)
        if not hyperparameters:
            return self._label_
        return f"{self._label_} {hyperparameters}"

    @property
    def parameters(self):
        return self.params




class Raw(BenchmarkGenerator):
    """Wrapper for BenchmarkGenerator that simply samples from the real data."""
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
