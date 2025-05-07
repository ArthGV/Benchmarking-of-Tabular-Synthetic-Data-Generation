"""
Extension of the TAPAS library to incorporate generators from other libraries and provide the functionality to read out parameters
"""
import os
import inspect
import re
from tapas.datasets.dataset import TabularDataset
from tapas.generators import Generator, Raw
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

import inspect
import re

import inspect
import re

class BenchmarkGenerator:
    def __init__(self, generator_class, method_kwargs, param_kwargs, label=None):
        self.generator_class = generator_class
        self.method_kwargs = method_kwargs.copy()
        self.param_kwargs = param_kwargs.copy()
        self.generator_instance = None
        self.trained = False
        self._label = label

        # Merge kwargs for instantiation
        combined_kwargs = {**self.method_kwargs, **self.param_kwargs}
        init_sig = inspect.signature(generator_class.__init__)
        init_args = self._filter_kwargs(init_sig, combined_kwargs)


        # Depending on your generator class you might need to change the initialization
        self.generator_instance = generator_class(**init_args)

        # Enforce fit & generate methods
        for method_name in ["fit", "generate"]:
            method = getattr(self.generator_instance, method_name, None)
            if not callable(method):
                raise AttributeError(
                    f"The class `{generator_class.__name__}` must implement a callable `{method_name}()` method.\n"
                    f"If it doesn't, you can subclass `BenchmarkGenerator` and override the `{method_name}()` method "
                    f"to provide custom logic."
                )

    #-----------------------------------------------------------------------------------------------------------------#
    # fit() & generate()  can/should be overridden for new generator classes
    #
    # e.g:
    #    class MyBenchmarkGenerator(BenchmarkGenerator):
    #        def fit(self, dataset):
    #            # baseclass implementation can be overwritten if needed
    #            super().fit(dataset)
    #
    #        def generate(self, num_samples):
    #            # baseclass implementation can be overwritten if needed
    #           return super().generate(num_samples)
    # -----------------------------------------------------------------------------------------------------------------#

    def fit(self, dataset, **kwargs):
        self.generator_instance.fit(dataset, **kwargs)
        self.trained = True

    def generate(self, num_samples):
        if not self.trained:
            raise RuntimeError("Generator must be fitted before generating.")

        generate_sig = inspect.signature(self.generator_instance.generate)
        args = {**self.param_kwargs, 'num_samples': num_samples}
        gen_args = self._filter_kwargs(generate_sig, args)

        return self.generator_instance.generate(**gen_args)

        # The call method should map a dataset to a synthetic dataset.
        # Subclasses can implement __call__ as a function or a coroutine. If coroutine is
        # used then concurrency should be enabled in the threat model.



    # -----------------------------------------------------------------------------------------------------------------#
    # these can be left as is (but can also be overwritten if needed)
    # -----------------------------------------------------------------------------------------------------------------#
    def __call__(self, dataset, num_samples, **kwargs):
        # The call method should map a dataset to a synthetic dataset.
        self.fit(dataset, **kwargs)
        return self.generate(num_samples)

    @staticmethod
    def _filter_kwargs(signature, kwargs):
        valid_params = signature.parameters
        return {
            k: v for k, v in kwargs.items()
            if k in valid_params or any(p.kind == p.VAR_KEYWORD for p in valid_params.values())
        }

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

    def _get_label_from_method(self):
        for val in self.method_kwargs.values():
            if isinstance(val, str):
            # if the value is a string, use it directly
                return val
            # else use the name of the generator method
            if isinstance(val, type):
                return val.__name__
        return self.generator_class.__name__ #use name of generator class if no method provided (e.g. Raw)

    def get_filename_label(self):
        hyperparameters = self._format_hyperparameters_for_filename(self.param_kwargs)
        # if hyperparameters empty return only the label
        if not hyperparameters:
            return self.label
        return f"{self.label}_{hyperparameters}"

    @property
    def parameters(self):
        return self.param_kwargs

    @property
    def label(self):
        # if label is set and not an empty string, return it
        if self._label not in (None, ""):
            return self._label

        # if label not manually set: infer from method or generator class
        else:
            class_path_str = self._get_label_from_method()
            if class_path_str.startswith("<class '") and class_path_str.endswith("'>"):
                class_path_str = class_path_str[len("<class '"):-len("'>")]
            self._label = class_path_str.split('.')[-1]  # Use the last part of the class name

        return self._label



# -----------------------------------------------------------------------------------------------------------------#
# MASTER THESIS STUFF
# -----------------------------------------------------------------------------------------------------------------#

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
                file.write(str(model.model.loss_history['loss'].to_dict())+'\n')

        return TabularDataset(output, self.tabular_dataset.description)

    @property
    def label(self):
        """Cherry on top."""
        return self._label



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