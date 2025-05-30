import os
import gc
import yaml
import numpy as np
import pandas as pd
from typing import Literal
from datetime import datetime
import time
import pickle
import random

import tapas
from tapas.datasets.data_description import DataDescription
from utils.plotting import single_plot, double_plot, plot_generators_ranks, breaking_time_plot, plot_radar_comparison
from utils.benchmark_metric import BenchmarkMetric
from utils.baseline_attack import get_baseline_score
from utils.tracker import TqdmTracker

from utils.basegenerators import BenchmarkGenerator
import matplotlib.pyplot as plt

import torch
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import gower
from scipy.stats import randint, uniform
from sklearn.metrics import f1_score
from sklearn.preprocessing import minmax_scale


class BenchmarkPipeline():

    STORED_DATA_FOLDER: str = 'data/generated_datasets/'
    RESULTS_FOLDER: str = 'results/'
    RUN_FOLDER: str = RESULTS_FOLDER
    RESULTS_PLOT_FOLDER: str =  RUN_FOLDER + 'plots/'


    def __init__(self, data, attack, generators: list[BenchmarkGenerator], target_record = None, data_size=15000, size_of_datasets: int | None = None):
        
        self.data = data
        self.attack = attack
        self.generators = generators
        self.data_size = data_size

        if target_record:
            #if target record is provided, it must not be in data
            self.target_record = target_record
        else:
            ind = [int(random.random() * len(data.data))]
            self.target_record = self.data.get_records(ind)
            self.data.drop_records(ind, in_place=True)


        if self.data_size > len(data.data):
            self.data_size = len(data.data)
            print("Dataset contains less samples than specified or default data_size, reduced data_size")
        else:
            #Take only subset of data
            self.data = data.sample(n_samples=self.data_size)


        self.attacker_data, self.defender_data = self.data.create_subsets(n = 2, sample_size= int(len(self.data) / 2))

        self.size_of_datasets =  size_of_datasets if size_of_datasets else int(len(self.defender_data) / 5)

        self.data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
            test_data= self.defender_data,
            aux_data= self.attacker_data,
            num_training_records= self.size_of_datasets
        )
        self.generator_knowledge = [
            tapas.threat_models.BlackBoxKnowledge(
                g,
                num_synthetic_records = self.size_of_datasets,
            )
            for g in self.generators
        ]

        self.threat_models = [
            tapas.threat_models.TargetedMIA(
                    attacker_knowledge_data=self.data_knowledge,
                    target_record=self.target_record,
                    attacker_knowledge_generator=g_k,
                    generate_pairs=True,
                    replace_target=True,
                    iterator_tracker=TqdmTracker
            )
            for g_k in self.generator_knowledge
        ]

    def run(self, 
            complexity_range: list[int], 
            run_per_range: int, 
            number_of_tests: int,
            p: float = 0.5,
            plot_style: Literal['single', 'double', 'all'] | None = 'all',
            benchmarking_metric: BenchmarkMetric | None = None, 
            classification_target_col: str = 'target',
            classification_num_samples: int = 1000,
            classification_test_size: float = 0.2,
            classification_classifier = None,
            classification_cv: int = 5,
            classification_n_bootstrap: int = 100,
            classification_random_state: int = 42,
            classification_optimize_hyperparams: bool = True,
            classification_preprocess_data: bool = True,
            classification_cat_features: list[str] = None,
            dcr_metric: str = 'euclidean',
            dcr_num_samples: int = 100,
            generate_data = True,
            store_results = True,
            ):
        
        """Run the Pipeline

        Args:
            complexity_range (_type_): Range of number of shadow models to test the attacks
            run_per_range (_type_): Number of attack per complexity point
            number_of_tests (_type_): Number of test per attack
            p: maximum average overlap fraction between two datasets taken from the shadow pool, cannot be 0
            plot_style (Literal['single', 'double'] | None, optional): indicate if the user wants to get attack plots for every generators, and their styles.
            benchmarking_metric (BenchmarkMetric | None, optional): class containing the metric to compute the final benchmark rank, None to skip that part
            imported_data_paths (list[str  |  None] | None, optional): Put path of the data at the position corresponding to the generator to load data instead of
            generating it, or None to generate the data. Put the all argument to None to generate data for all generators.
            store_generated_datasets_paths (list[str  |  None] | None, optional): Put path where you want to store generated data at the position corresponding to the generator, or None not to store them. Put the all argument to None to store nothing.
            store_results_path (str | None, optional): Path to store the results of the attack. If None, the results are not stored.
        """
        now = datetime.now()
        formatted_now = now.strftime("%Y_%m_%d_%H_%M")
        self.RUN_FOLDER = self.RUN_FOLDER + formatted_now + '/'
        self.RESULTS_PLOT_FOLDER: str =  self.RUN_FOLDER + 'plots/'
        if store_results:
            os.makedirs(self.RUN_FOLDER, exist_ok=True)
            os.makedirs(self.RESULTS_PLOT_FOLDER, exist_ok=True)
        if generate_data:
            os.makedirs(self.STORED_DATA_FOLDER, exist_ok=True)

        self.baseline_score = get_baseline_score(self.attacker_data, self.target_record, num_bins=25) 

        if store_results:
            metadata = {
                'dataset' : self.data.description.label,
                'generators' : [repr(g) for g in self.generators],
                'benchmarking_metric' : repr(benchmarking_metric),
                'complexity_range' : complexity_range,
                'run_per_range' : run_per_range,
                'number_of_tests' : number_of_tests,
                'p' : p,
                'size_of_datasets': self.size_of_datasets,
                'baseline_score' : round(float(self.baseline_score[0]), 3)
            }
            with open(self.RUN_FOLDER + 'metadata.yaml', "w") as file:
                yaml.dump(metadata, file, default_flow_style=False)   
        
        attack_results = []
        for i in range(len(self.generators)):

            M_0, S_0, M_1, S_1, gen_time = self._attack_one_generator(i, complexity_range, run_per_range, number_of_tests, p, generate_data)
            attack_results.append({
                                    'M_0': M_0,
                                    'S_0': S_0,
                                    'M_1': M_1,
                                    'S_1': S_1,
                                    'gen_time' : gen_time
                                 })
            if plot_style in ['single', 'all']:
                fig, axes = single_plot(repr(self.generators[i]), np.array(complexity_range), (1 - np.array(M_0) + np.array(M_1)) / 2, np.array(S_0) + np.array(S_1), self.baseline_score)
                plt.show()
                if store_results:
                    fig.savefig(self.RESULTS_PLOT_FOLDER + repr(self.generators[i]) + "_single" + ".png", dpi=300)
            if plot_style in ['double', 'all']:
                fig, axes = double_plot(repr(self.generators[i]), np.array(complexity_range), np.array(M_0), np.array(S_0), np.array(M_1), np.array(S_1), self.baseline_score)
                plt.show()
                if store_results:
                    fig.savefig(self.RESULTS_PLOT_FOLDER + repr(self.generators[i]) + "_double" + ".png", dpi=300)

        if benchmarking_metric:
            print('----[Benchmark ranks]----')
            generators_metrics = []
            baseline_score = self.baseline_score[0]
            for i in range(len(self.generators)):
                # compute generator ranks
                data_mean = (1 - np.array(attack_results[i]['M_0']) + np.array(attack_results[i]['M_1'])) / 2
                data_std = np.array(attack_results[i]['S_0']) + np.array(attack_results[i]['S_1'])
                benchmark_rank = benchmarking_metric.compute_rank(complexity_range, data_mean, data_std, baseline_score)
                benchmark_metric = benchmarking_metric.compute_metric(complexity_range, data_mean, data_std, baseline_score)

                print(f'{self.generators[i]} : {benchmark_rank}, {benchmark_metric}')
                # compute generator breaking time
                breaking_complexity = self._find_breaking_point(complexity_range, attack_results[i]['M_0'], attack_results[i]['M_1'], baseline_score)
                if breaking_complexity < 0:
                    breaking_time = -1
                    print(f'{self.generators[i]} was not broken by the attack')
                else:
                    breaking_time = breaking_complexity * attack_results[i]['gen_time']
                generators_metrics.append({"Model": repr(self.generators[i]), "Speed": attack_results[i]['gen_time'], "Benchmark_rank":benchmark_rank, "Benchmark_score":benchmark_metric, "Breaking_Time": breaking_time})

            if store_results:
                results_path = self.RUN_FOLDER + 'generators_metrics.csv'
                df = pd.DataFrame(generators_metrics)
                df.to_csv(
                    results_path,
                    mode='w',
                    index=False,
                    header=True
                )
                print(f'Generators Metrics saved to {results_path}')

                results_path = self.RUN_FOLDER + 'attack_results.yaml'
                attack_results_dict = {}
                for i, gen in enumerate(self.generators):
                    attack_results_dict[repr(gen)] = attack_results[i]
                with open(results_path, "w") as file:
                    yaml.dump(attack_results_dict, file, default_flow_style=False)
                print(f'Attack results saved to {results_path}')



            plot_generators_ranks(generators_metrics, store_results=store_results, result_plot_folder=self.RESULTS_PLOT_FOLDER)
            breaking_time_plot(generators_metrics, store_results=store_results, result_plot_folder=self.RESULTS_PLOT_FOLDER)
            
            self._ml_utility(target_col=classification_target_col, num_samples=classification_num_samples, 
                            test_size=classification_test_size, classifier=classification_classifier, cv=classification_cv,
                            random_state=classification_random_state, 
                            optimize_hyperparams=classification_optimize_hyperparams, preprocess_data=classification_preprocess_data, store_results=store_results)
            self._average_dcr(num_samples = dcr_num_samples, metric=dcr_metric,store_results=store_results)

            results = self._data_results(generators_metrics, repr(benchmarking_metric))
            plot_radar_comparison(results, metrics=['Speed', repr(benchmarking_metric), 'Breaking_Time', 'DCR', 'Utility'], store_results=store_results, result_plot_folder=self.RESULTS_PLOT_FOLDER)


    def _attack_one_generator(self, generator_ind: int,complexity_range: list[int], run_per_range: int, number_of_tests: int, p: float, generate_data: bool):
        print('Attacking :', repr(self.generators[generator_ind]))
        number_of_generated_shadow_datasets = int(complexity_range[-1] * (1/p))
        path_data = self.STORED_DATA_FOLDER + repr(self.generators[generator_ind]) + ".pkl"
        if generate_data:
            print("Generating Shadow Data Pool")
            shadow_data_pool, gen_time = self._generate_shadow_datasets(self.threat_models[generator_ind], number_of_generated_shadow_datasets, path_data)
        else:
            print('Import datasets from', path_data)
            shadow_data_pool, gen_time = self._import_shadow_datasets(path_data)
        print("Generating Test Data")
        test_datasets, truth_labels = self.threat_models[generator_ind]._generate_samples(number_of_tests, False, True)
        M_0 = []
        S_0 = []
        M_1 = []
        S_1 = []
        for complexity in complexity_range:
            print('COMPLEXITY: ', complexity)
            P_0 = []
            P_1 = []
            for _ in range(run_per_range):
                sampled_shadow_data = self._sample_shadow_dataset(shadow_data_pool, complexity)
                random.shuffle(sampled_shadow_data)
                sampled_datasets = [dataset[0] for dataset in sampled_shadow_data]
                sampled_labels = [dataset[1] for dataset in sampled_shadow_data]

                self.attack.classifier.fit(sampled_datasets, sampled_labels)
                self.attack.trained = True
                
                pred_labels = self.attack.attack(test_datasets)
                P_0.extend([p for i, p in enumerate(pred_labels) if truth_labels[i] == False])
                P_1.extend([p for i, p in enumerate(pred_labels) if truth_labels[i] == True])
            M_0.append(float(np.mean(P_0)))
            S_0.append(float(np.std(P_0)))
            M_1.append(float(np.mean(P_1)))
            S_1.append(float(np.std(P_1)))
        return M_0, S_0, M_1, S_1, gen_time
    
    def _find_breaking_point(self, complexity_range, mean_0, mean_1, baseline_score):
        for i in range(len(mean_0)):
            if all(x > baseline_score for x in mean_1[i:]) and all(x < baseline_score for x in mean_0[i:]):
                if i == 0:
                    return complexity_range[i]
                #breaking point
                c_b = complexity_range[i]
                M_b_0 = mean_0[i]
                M_b_1 = mean_1[i]
                #previous point
                c_p = complexity_range[i - 1]
                M_p_0 = mean_0[i - 1]
                M_p_1 = mean_1[i - 1]
                #inverse interpolations
                slope_0 = (M_b_0 - M_p_0 + 1e-6) / (c_b - c_p) #1e-6 for numerical stability
                slope_1 = (M_b_1 - M_p_1 + 1e-6) / (c_b - c_p)
                if_0 = lambda y : c_p + ((y - M_p_0) / slope_0)
                if_1 = lambda y : c_p + ((y - M_p_1) / slope_1)
                #find breaking point
                breaking_0 = if_0(baseline_score)
                breaking_1 = if_1(baseline_score)
                if breaking_0 > c_b or breaking_0 < c_p:
                    breaking_0 = -1
                if breaking_1 > c_b or breaking_1 < c_p:
                    breaking_1 = -1
                breaking_point = max(breaking_0, breaking_1)
                return breaking_point
        return -1  # return -1 if no such point exists
    
    def _sample_shadow_dataset(self, shadow_dataset_pool: list[tapas.datasets.dataset.TabularDataset], number_of_shadow_models: int):
        """
        Returns:
            list[tapas.datasets.dataset.TabularDataset, bool]: subset of shadow_dataset_pool of lenght number_of_shadow_models such that there is the same number of dataset with/without target.
        """
        assert number_of_shadow_models % 2 == 0, 'number_of_shadow_models should be a even integer'
        shadow_datasets = random.sample(shadow_dataset_pool[0], int(number_of_shadow_models / 2))
        shadow_datasets.extend(random.sample(shadow_dataset_pool[1], int(number_of_shadow_models / 2)))
        return shadow_datasets
    
    def _generate_shadow_datasets(self, threat_model, number_of_train_datasets: int, path_to_store_generated_datasets: str):
        start_time = time.time()
        shadow_datasets, shadow_labels = threat_model.generate_training_samples(number_of_train_datasets, ignore_memory=True)
        generation_time = (time.time() - start_time) * 1000000 # in microseconds
        normalized_gen_time = generation_time / (self.size_of_datasets * number_of_train_datasets)
        shadow_data = list(zip(shadow_datasets, shadow_labels))


        with open(path_to_store_generated_datasets, "wb") as f:
            data_to_save = {
                "shadow_data": shadow_data,
                "gen_time": normalized_gen_time
            }
            pickle.dump(data_to_save, f)
            print(f'Generated datasest of {threat_model.atk_know_gen.generator} stored at :{path_to_store_generated_datasets}')

        shadow_data_0 = [sd for sd in shadow_data if not sd[1]]
        shadow_data_1 = [sd for sd in shadow_data if sd[1]]
        shadow_data_pool = [shadow_data_0, shadow_data_1]
        return shadow_data_pool, normalized_gen_time
        
    def _import_shadow_datasets(self, path:str):
        with open(path, "rb") as d:
            loaded_data = pickle.load(d)
            shadow_data = loaded_data["shadow_data"]
            gen_time = loaded_data["gen_time"]

        shadow_data_0 = [sd for sd in shadow_data if not sd[1]]
        shadow_data_1 = [sd for sd in shadow_data if sd[1]]
        shadow_data_pool = [shadow_data_0, shadow_data_1]
        return shadow_data_pool, gen_time   
    
    def _ml_utility(
        self, 
        target_col='target', 
        num_samples=1000,
        test_size=0.2,
        classifier=None, 
        cv=5,
        random_state=42, 
        optimize_hyperparams=True,
        preprocess_data=True,
        store_results=True,
        ):

        cat_features = self.data.description.one_hot_cols
        cat_features = [col for col in cat_features if col != target_col]
        description = DataDescription(self.data.description.schema)
        dataset = self.data.data # get the data from the dataset as a pandas dataframe
        dataset = dataset.sample(n= round(num_samples/(1-test_size))+2)


        y = dataset[target_col]
        X = dataset.drop(columns=[target_col])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # combine X_train and y_train into a single dataframe
        train_combined = pd.concat([X_train, y_train], axis=1)
        # transform train_combined into tapas.datasets.dataset.TabularDataset
        train_combined_tab = tapas.datasets.dataset.TabularDataset(train_combined,description)
        num_samples = num_samples if num_samples else len(train_combined)

        # evaluate the dataset using the ml_utility function with original data
        print('Evaluating original data')
        result = self._evaluate_ml_pipeline(
                X_train, y_train, X_test, y_test, classifier, 
                cv, random_state,optimize_hyperparams,
                preprocess_data, cat_features)
        rows = []
        for i in range(len(self.generators)):
            print('Evaluating :', repr(self.generators[i]))
            generator = self.generators[i]
            generated_data = generator(train_combined_tab, num_samples)

            # evaluate the dataset using the ml_utility function with generated data
            y_gen = generated_data.data[target_col]
            X_gen = generated_data.data.drop(columns=[target_col])

            print('Evaluating generated data')
            result_gen = self._evaluate_ml_pipeline(
                X_gen, y_gen, X_test, y_test, classifier,
                  cv, random_state,optimize_hyperparams,
                  preprocess_data,categorical_cols=cat_features)
            row_gen = {"Model": repr(generator), "test_score_original": result['test_score'], "test_score_generated": result_gen['test_score'], "utility_score": min(1., float(result_gen['test_score'] / result['test_score']))}
            # print('Results for generated data:', result_gen)
            rows.append(row_gen)

            #clear memory
            del generated_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # save the results to a csv file
        if store_results:
            results_path= self.RUN_FOLDER + "results_ml_utility.csv"
            df = pd.DataFrame(rows)
            df.to_csv(
                results_path,
                mode='w',
                index=False,
                header=True
            )
            print(f'Results saved to {results_path}')

    def _evaluate_ml_pipeline(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        classifier=None,
        cv: int = 5,
        random_state: int = 42,
        optimize_hyperparams: bool = True,
        preprocess_data : bool = True,
        categorical_cols: list[str] = None
        ) -> dict:
        """
        Simple ML utility evaluation pipeline using XGBoost:
        1. Splits the dataset into train and test.
        2. Performs k-fold cross-validation on the train set.
        3. Fits the classifier on the entire train set.
        4. Evaluates on the test set.

        Parameters:
            X_train, y_train : training data
            X_test, y_test : test data
            classifier : estimator object, default=None
                If None, XGBClassifier will be used
            cv : int, default=5
                Number of cross-validation folds
            random_state : int, default=42
                Random seed for reproducibility
            optimize_hyperparams : bool, default=True
                Whether to perform hyperparameter optimization
            model_path : str, default='trained_model.pkl'
                Path to save the trained model
            preprocess_data : bool, default=True
                Whether to automatically preprocess data (encode categoricals and normalize numerics)
        """
        # Default classifier: XGBoost
        if preprocess_data:
            # Convert to pandas DataFrame if not already
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)
            
            # numerical columns are the ones not in categorical_cols
            numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ],
                remainder='passthrough'
            )
            
            # Process features
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Encode target variable if it's categorical
            if isinstance(y_train, (list, pd.Series, np.ndarray)) and (isinstance(y_train.iloc[0], str) or isinstance(y_train.iloc[0], bool)):
                # print("Encoding categorical target variable...")
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_test_encoded = label_encoder.transform(y_test)
            else:
                y_train_encoded = y_train
                y_test_encoded = y_test
                
            # Use the processed data
            X_train, X_test = X_train_processed, X_test_processed
            y_train, y_test = y_train_encoded, y_test_encoded
        
        # Default classifier: XGBoost
        if classifier is None:
            classifier = XGBClassifier(eval_metric='logloss',
                                    random_state=random_state)
        
        # Hyperparameter optimization
        if optimize_hyperparams and isinstance(classifier, XGBClassifier):    
            # Define the hyperparameter search space
            param_dist = {
                'n_estimators':        randint(50, 500),
                'max_depth':           randint(3, 12),
                'learning_rate':       uniform(0.01, 0.3),
                'subsample':           uniform(0.5, 0.5),
                'min_child_weight':    uniform(0, 10),
                'gamma':               uniform(0, 5),
                'reg_alpha':           uniform(0, 1),
                'reg_lambda':          uniform(0, 1),
            }
            
            # Randomized search with cross-validation
            random_search = RandomizedSearchCV(
                classifier,
                param_distributions=param_dist,
                n_iter=20,  # Number of parameter settings to try
                cv=cv,
                verbose=1,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            )
            
            # Fit randomized search
            random_search.fit(X_train, y_train)
            
            # Get the best classifier
            classifier = random_search.best_estimator_

        # Fit on full training set
        classifier.fit(X_train, y_train)

        # Test-set evaluation with f1 score
        y_pred = classifier.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='binary')

        return {
            'test_score': f1,
            'model': classifier#,
            # 'model_path': model_path
        }

    def _binary_mask_list(self, df: pd.DataFrame, selected_cols: list[str]) -> list[int]:
        """
        Given a DataFrame `df` and a list of column names `selected_cols`,
        return a binary list of length df.shape[1] where each position is
        1 if that column is in selected_cols, else 0.
        """
        return [1 if col in selected_cols else 0 for col in df.columns]

    def _compute_dcr(self, real_data, synth_data, metric='euclidean', cat_features=None):
        """
        Compute the Distance to Closest Record (DCR) for each synthetic sample.
        Supports numeric-only (euclidean, manhattan, etc.) or mixed data via Gower.

        Parameters
        ----------
        real_data : array-like or pandas.DataFrame of shape (n_real, n_features)
        synth_data : array-like or pandas.DataFrame of shape (n_synth, n_features)
        metric : str, default='euclidean'
            Numeric metrics supported by sklearn neighbors, or 'gower' for mixed data.
        cat_features : list of column names (optional)
            When using Gower, specify any numeric-coded categorical columns here.

        Returns
        -------
        dcr_values : numpy.ndarray of shape (n_synth,)
            Minimum distance from each synthetic sample to its nearest real sample.
        """
        if metric == 'gower':
            if cat_features is not None:
                # Explicit categorical list
                cat_features_bin = self._binary_mask_list(real_data, cat_features)
                D = gower.gower_matrix(real_data, synth_data, cat_features=cat_features_bin)
            return np.min(D, axis=0)

        # Numeric-only path
        real_arr = np.asarray(real_data)
        synth_arr = np.asarray(synth_data)
        if real_arr.ndim != 2 or synth_arr.ndim != 2:
            raise ValueError("real_data and synth_data must be 2D arrays for numeric metrics.")
        if real_arr.shape[1] != synth_arr.shape[1]:
            raise ValueError(
                f"Feature mismatch: real has {real_arr.shape[1]} features, synth has {synth_arr.shape[1]}"
            )
        nn = NearestNeighbors(n_neighbors=1, metric=metric)
        nn.fit(real_arr)
        dist, _ = nn.kneighbors(synth_arr, return_distance=True)
        return dist.ravel()

    def _data_results(self, generators_metrics, metric_name):
        results_complexity_break = pd.DataFrame(generators_metrics)
        results_dcr = pd.read_csv(self.RUN_FOLDER + "results_dcr.csv")
        results_utility = pd.read_csv(self.RUN_FOLDER + "results_ml_utility.csv")

        # normalize the results in complexity_break
        max_speed = 1.05 * results_complexity_break["Speed"].max()
        min_speed = 0.95 * results_complexity_break["Speed"].min()
        results_complexity_break["Speed"] = (results_complexity_break["Speed"] - max_speed) / (min_speed - max_speed)

        results_complexity_break["Benchmark_score"] = 1 - results_complexity_break["Benchmark_score"]

        mask = results_complexity_break['Breaking_Time'] != -1
        bt_min = 1.05*results_complexity_break.loc[mask, 'Breaking_Time'].min()
        bt_max = 0.95*results_complexity_break.loc[mask, 'Breaking_Time'].max()
        results_complexity_break.loc[mask, 'Breaking_Time'] =  (results_complexity_break.loc[mask, 'Breaking_Time'] - bt_min) / (bt_max - bt_min)
        results_complexity_break.loc[~mask, 'Breaking_Time'] = 1.05

        # normalize the results in dcr
        results_dcr["dcr_mean"] = minmax_scale(results_dcr["dcr_mean"])

        # normalize the results in ml_utility
        results_utility = results_utility[["Model", "utility_score"]]

        results = pd.merge(results_complexity_break, results_dcr, on="Model")
        results = pd.merge(results, results_utility, on="Model")
        results = results.rename(columns={"Benchmark_score": metric_name, "dcr_mean": "DCR", "utility_score": "Utility"})
        return results
    
    def _average_dcr(self, num_samples = 100, metric='euclidean',store_results=True, results_path="results_dcr.csv"):
        """
        Compute average DCR across synthetic samples.
        """
        cat_features = self.data.description.one_hot_cols
        #use only a subset of the data to train the generators
        train_data = self.data
        train_data = train_data.sample(n_samples=1000)
            
        rows = []
        for i in range(len(self.generators)):
            print('Evaluating :', repr(self.generators[i]))
            generator = self.generators[i]
            generated_data = generator(train_data, num_samples)
            dcr_vals = self._compute_dcr(self.defender_data.data, generated_data.data, metric, cat_features)
            row = {"Model": repr(generator), "dcr_mean": np.mean(dcr_vals), "dcr_std": np.std(dcr_vals)}
            rows.append(row)
        
        # save the results to a csv file
        if store_results:
            results_path= self.RUN_FOLDER + "results_dcr.csv"
            df = pd.DataFrame(rows)
            df.to_csv(
                results_path,
                mode='w',
                index=False,
                header=True
            )
        
