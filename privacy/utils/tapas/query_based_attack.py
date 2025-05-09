"""
Adapt the implementation of Achille's Heel to use CQBS instead of the python implementation
for query-extraction.
"""

# stdlib
import math
import itertools
import warnings
from copy import deepcopy

# third party
import pandas as pd
import numpy as np
from tqdm import tqdm # TODO change that to allow the possibility to use a silent iterator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

# MIA-synthetic
from src.optimized_qbs import qbs

from tapas.attacks import ShadowModellingAttack
from tapas.attacks.set_classifiers import FeatureBasedSetClassifier, SetFeature
from tapas.datasets import TabularRecord


class RandomTargetedQueryFeatureFast(SetFeature):
    """
    Features that computes random targeted queries that include the user.
    This implementation uses the CQBS C implementation to speed up the process of
    the RandomTargetedQueryFeature python class. 

    These queries take the form SUM_{x in D} AND_{a in S} I{x_s = target_s},
    counting all records that match the target record in all attributes in S.
    The set S is sampled randomly from all subsets of attributes of a fixed
    length (order).

    This is equivalent to using entries of the order-way contingency
    table that count the target user as features. In particular, for 
    order=1, this is a subset of HistSetFeature().
    """
    def __init__(self, target: TabularRecord, low_order: int, high_order: int, num_queries: int, 
                 cat_condition_options: tuple = (1,), cont_condition_options: tuple = (-3,),
                 random_state: int = 42):
        """
        Parameters
        ----------
        """
        self.target_record = target.data
        self.num_columns = self.target_record.shape[1]

        if low_order < 1:
            raise ValueError("The lowest order must be at least 1!")
        if high_order > self.num_columns:
            warnings.warn("The highest order is larger than the number of columns. The combinations with order > num_columns will be ignored.")
            high_order = self.num_columns
            
        self.low_order = low_order
        self.high_order = high_order
        self.num_queries = num_queries
        self.np_rng = np.random.default_rng(random_state)

        data_description = target.description
        self.categorical_columns = data_description.one_hot_cols
        self.continuous_columns = [col for col in data_description.columns if col not in self.categorical_columns]
        self.categorical_indices = [data_description.columns.index(col) for col in self.categorical_columns]
        self.continous_indices = [data_description.columns.index(col) for col in self.continuous_columns]

        self.set_categorical_encoder(data_description)
        # for C QBS we need int for categorical.
        self.target_record_int = self.convert_categorical_columns_to_int(self.target_record)

        self.set_queries(cat_condition_options, cont_condition_options)


    def set_categorical_encoder(self, data_description):
        """
        Initialize the categorical encoder for the categorical columns.
        It's an OrdinalEncoder object but we use it only for query definition. It
        will not impact the non-ordering of the categorical columns.
        """
        self.categorical_encoder = OrdinalEncoder(
            categories=[col['representation'] for col in data_description.schema if col['name'] in self.categorical_columns],
            dtype=int)
        self.categorical_encoder.fit(self.target_record[self.categorical_columns])


    def convert_categorical_columns_to_int(self, dataset):
        """
        Convert the dataset to int for categorical columns, using the label encoders.

        Parameters
        ----------
        dataset: pd.DataFrame
            The dataset to convert.

        Returns
        -------
        pd.DataFrame
            The converted dataset, containing only int values for the categorical columns.
        """
        dataset_int = dataset.copy()
        dataset_int[self.categorical_columns] = self.categorical_encoder.transform(dataset_int[self.categorical_columns])
        return dataset_int
    
        
    def get_n_queries_per_order(self):
        """
        Generate a list of tuples (order, number) specifying the number of combinations
        to gather for each order from order_low to order_high, limited by the maximum
        possible combinations for each order given num_columns. The combinations are sampled
        from the 2^num_columns total combinations possible.

        For instance, calling get_query_features(1, 1, 20, 17) would return [(1, 17)] 
        as there can only be 17 order 1 combinations in this setting.

        Parameters
        ----------
            order_low: int 
                Minimum order of combination.
            order_high: int 
                Maximum order of combination.
            num_meta_features: int
                Total number of combinations to gather across all orders.
            num_columns: int
                Total number of columns available for combinations. It typically corresponds to the number of features in the dataset.
            random_state: int
                Seed for reproducibility. TODO change that

        Returns
        -------
            combination_list: list of tuples (order, number)
                Specifies the number of combinations for each order.
        """    
        # Generate possible orders and calculate max possible combinations per order
        possible_orders = list(range(self.low_order, self.high_order + 1))
        max_combinations = [min(math.comb(self.num_columns, order), self.num_queries) for order in possible_orders]
                
        # Generate random allocation of `number_total` combinations within the max limits
        proportions = self.np_rng.dirichlet(np.ones(len(possible_orders)))
        counts = np.round(proportions * self.num_queries).astype(int)
        
        # Adjust counts to ensure they do not exceed max_combinations
        counts = np.minimum(counts, max_combinations)
        
        # Adjust counts to match `number_total` precisely, adding counts one by one
        # for a random index that needs to be adjusted, i.e. for which the current count is smaller than its max
        discrepancy = self.num_queries - counts.sum()
        while discrepancy != 0:
            # Find indices of counts that can be adjusted
            adjustable_indices = [i for i in range(len(counts)) if (counts[i] < max_combinations[i] if discrepancy > 0 else counts[i] > 0)]
            
            # Randomly pick an index to adjust
            if adjustable_indices:
                idx = self.np_rng.choice(adjustable_indices)
                counts[idx] += 1 if discrepancy > 0 else -1
                discrepancy = self.num_queries - counts.sum()
            else:
                break  # If no adjustments can be made without violating max limits, simply return the current state
        
        # Create the list of (order, count) tuples    
        order_counts = [(order, count) for order, count in zip(possible_orders, counts) if count > 0]
        
        return order_counts
        
    
    def set_queries(self, cat_condition_options, cont_condition_options):
        '''
        Generate queries given the orders, categorical and continuous indices, number of columns, number of queries.
        
        The generated queries are permutations of the conditions on the categorical and continuous indices. The returned array
        contains multiple queries, each with a different condition on the categorical and continuous indices.
        Each query will correspond to a feature in the final feature matrix.

        Condition options:
            0  ->  no condition on this attribute;
            1  ->  ==
            -1  ->  !=
            2  ->  >
            3  ->  >=
            -2  ->  <
            -3  ->  <=
        '''
        self.queries = []
        n_queries_per_order = self.get_n_queries_per_order()
        
        for (order, n_queries) in n_queries_per_order:
            all_order_combinations = list(itertools.combinations(range(self.num_columns), order))

            # select only a subset of the combinations
            selected_indices = np.random.choice(
                len(all_order_combinations), replace=False, size=(n_queries,)
            )            
            order_combinations = [all_order_combinations[idx] for idx in selected_indices]

            for columns_indices in order_combinations:
                indices_combinations = []
                for i, col_index in enumerate(columns_indices):
                    if col_index in self.categorical_indices:
                        index_options = cat_condition_options
                    else:
                        index_options = cont_condition_options
                        
                    if i == 0:
                        for index_option in index_options:
                            base_tup = np.array([0] * self.num_columns)
                            base_tup[col_index] = index_option
                            indices_combinations.append(base_tup)
                    else:
                        for j, index_option in enumerate(index_options):
                            if j == 0:
                                for base_tup in indices_combinations:
                                    base_tup[col_index] = index_option
                            else:
                                indices_combinations_c = deepcopy(indices_combinations)
                                for base_tup in indices_combinations_c:
                                    base_tup[col_index] = index_option
                                indices_combinations += indices_combinations_c
                for combo in indices_combinations:
                    self.queries.append(tuple(combo))

        # make sure that there are no duplicates
        print("is duplicates?", len(set(self.queries)) != len(self.queries))


    def feature_extractor_queries_CQBS(self, synthetic_df: pd.DataFrame, target_record_int: pd.DataFrame):
        target_record_int = target_record_int.astype(int)
        synthetic_df = synthetic_df.astype(int)
        qbs_data = qbs.SimpleQBS(synthetic_df.itertuples(index=False, name=None))
        target_values = [tuple(target_record_int.values[0])]

        # get features by batch-quering using the queries and qbs
        features = qbs_data.query(target_values * len(self.queries), self.queries)
        return features


    def extract(self, datasets):
        """Compute queries on each dataset.
        
        Parameters
        ----------
        datasets: list of pd.DataFrame
            List of datasets to extract features from.

        Returns
        -------
        np.array
            The features extracted from the datasets.
        """
        all_features = []
        for dataset in tqdm(datasets, desc="Extracting query-based features"):
            # for C QBS we need int for categorical.
            dataset_int = self.convert_categorical_columns_to_int(dataset.data)
            features = self.feature_extractor_queries_CQBS(dataset_int, self.target_record_int)
            all_features.append(features)

        features = np.array(all_features)
        print("Features shape", features.shape)
        return features


    @property
    def label(self):
        return f"RandomQueriesFast(low={self.low_order}, high={self.high_order}, n={self.num_queries})"


class QueryBasedAttackFast(ShadowModellingAttack):
    """
    The query-based attack as defined in [1] (chapter 5.5)

    """

    def __init__(
        self, target_record, low_order, high_order, num_meta_features,
        model=RandomForestClassifier(n_estimators=100, max_depth=10, verbose=0, random_state=42),
        label="QueryBasedFast", random_state=42
    ):
        """
        All orders generated are within [low_order, high_order] included.

        Parameters
        ----------
        target_record: tapas.datasets.dataset.TabularDataset
            The record on which query matching is computed.
        order_low: int 
            Minimum order of combination.
        order_high: int 
            Maximum order of combination.
        num_meta_features: int
            Total number of combinations to gather across all orders.
        num_columns: int
            Total number of columns available for combinations. It typically corresponds to the number of features in the dataset.
        model: sklearn.base.ClassifierMixin (default RandomForestClassifier with 100 estimators and max depths 10)
            Binary classifier respectingt the scikit-learn interface.
        label: str (default "Achille's Heel")
            An optional label to refer to the attack in reports.
        seed: int (default 42)
            A seed for random queries selection reproducibility. TODO change that
        """
        features = RandomTargetedQueryFeatureFast(target_record, low_order=low_order, high_order=high_order, 
                        num_queries=num_meta_features, random_state=random_state)

        super().__init__(
            FeatureBasedSetClassifier(
                features,
                model
            ),
            label=label,
        )