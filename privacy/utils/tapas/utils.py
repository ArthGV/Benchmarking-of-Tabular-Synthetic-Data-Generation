"""
Utility functions to work with the TAPAS library.
"""
from tapas.datasets import TabularDataset
from tools.utils import load_pickle, dump_pickle

def export_threat_model_memory_to_dataframes_list(paths, threat_model):
    """
        Export a tapas.threat_models.TargetedMIA internal datasets
        into simple pandas datasets.
    """
    train_datasets = list(map(lambda x: x.data, threat_model._memory[True][0]))
    train_labels = threat_model._memory[True][1]
    del threat_model._memory[True][0]
    dump_pickle(train_datasets, paths['train_datasets'])
    dump_pickle(train_labels, paths['train_labels'])
    del threat_model._memory[True]

    test_datasets = list(map(lambda x: x.data, threat_model._memory[False][0]))
    test_labels = threat_model._memory[False][1]
    del threat_model._memory[False][0]
    dump_pickle(test_datasets, paths['test_datasets'])
    dump_pickle(test_labels, paths['test_labels'])
    del threat_model._memory[False]

def empty_threat_model_memory(threat_model):
    del threat_model._memory

def load_dataframes_list_to_threat_model_memory(paths, threat_model, metadata):
    # Note: alter the threat model object
    # TODO get the metadata
    raise NotImplemented("This function is not implemented yet")
    train_datasets = load_pickle(paths['train_datasets'])
    train_labels = load_pickle(paths['train_labels'])
    test_datasets = load_pickle(paths['train_datasets'])
    test_labels = load_pickle(paths['train_datasets'])


def get_categorical_and_numerical_features(data_description, target_column=None):
    categorical_features = [col for col in data_description.one_hot_cols if col != target_column]
    numerical_features = [col for col in data_description.columns if col not in categorical_features and col != target_column]
    return categorical_features, numerical_features