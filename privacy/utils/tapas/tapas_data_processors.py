"""
Contains utility classes to process datasets after their tapas import.
For instance, after importing the 2011 UK data census in TAPAS, the last column
is an integer even though it's defined as a category in the json description file.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseTapasDataProcessor(ABC):
    """
    Base class that defines the interface of data processors.
    """

    @staticmethod
    @abstractmethod
    def process_tapas_tabulardataset(dataset):
        """
        Parameters  
        ----------
        dataset: tapas.datasets.dataset.TabularDataset
            The dataset to process
        """
        pass

class CensusDataProcessor(BaseTapasDataProcessor):
    """
    Specific class to correct the 2011 UK data census after its import in TAPAS.
    """

    @staticmethod
    def process_tapas_tabulardataset(dataset):
        """
        Parameters  
        ----------
        dataset: tapas.datasets.dataset.TabularDataset
            The dataset to process
        """
        dataset.data["Approximated Social Grade"] = dataset.data["Approximated Social Grade"].astype(str)
        return dataset
    

class AdultDataProcessor(BaseTapasDataProcessor):
    """
    Specific class to correct the 2011 UK data census after its import in TAPAS.
    """

    @staticmethod
    def process_tapas_tabulardataset(dataset):
        """
        Parameters  
        ----------
        dataset: tapas.datasets.dataset.TabularDataset
            The dataset to process
        """
        dataset.data = dataset.data.replace('?', np.nan).dropna()
        return dataset