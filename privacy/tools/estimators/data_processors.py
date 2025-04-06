"""
This module contains the data processors for the estimators.
"""
from tapas.datasets import TabularDataset, DataDescription

class DataPreprocessor():
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.fitted = False
        # self.label_col = label_column

    def fit(self, dataset: TabularDataset):
        data_description = dataset.description
        self.features_cols = [col for col in data_description.columns if col != self.target_column]
        self.categorical_data = [col for col in data_description.one_hot_cols if col != self.target_column]
        self.continuous_data = [col for col in self.features_cols if col not in self.categorical_data]

    def process(self, dataset: TabularDataset):        
        if not self.fitted:
            self.fit(dataset)
            self.fitted = True

        X = dataset.data[self.features_cols].copy()
        y = dataset.data[self.target_column].copy()

        return X, y


class AdultDataPreprocessor(DataPreprocessor):
    def __init__(self, target_column: str):
        super().__init__(target_column)

    def process(self, dataset: TabularDataset):
        X, y = super().process(dataset)
        X[self.categorical_data] = X[self.categorical_data].astype('category')
        X[self.categorical_data] = X[self.categorical_data].apply(lambda x: x.cat.codes)
        y = y.astype('category').cat.codes


        # reuse the code in "utility.py"
        import warnings
        warnings.warn("Not implemented yet.")

        return X, y
