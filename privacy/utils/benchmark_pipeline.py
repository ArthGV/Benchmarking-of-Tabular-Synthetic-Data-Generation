import os
import tapas
import time
import numpy as np
from typing import Literal
import random
from utils.plotting import single_plot, double_plot, plot_generators_ranks
from utils.benchmark_metric import BenchmarkMetric
from tools.baseline_attack import get_baseline_score
import pickle
from tools.tapas.generators import BenchmarkGenerator
import matplotlib.pyplot as plt


class BenchmarkPipeline():
    def __init__(self, data, attack, generators: list[BenchmarkGenerator], target_record = None, size_of_datasets: int | None = None):
        
        self.data = data
        self.attack = attack
        self.generators = generators

        if target_record:
            #if target record is provided, it must not be in data
            self.target_record = target_record
        else:
            ind = [int(random.random() * len(data.data))]
            self.target_record = self.data.get_records(ind)
            self.data.drop_records(ind, in_place=True)

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
                    replace_target=True
            )
            for g_k in self.generator_knowledge
        ]

    def run(self, complexity_range: list[int], run_per_range, number_of_tests, plot_style: Literal['single', 'double'] | None = 'double', benchmarking_metric: BenchmarkMetric | None = None, imported_data_paths: list[str | None] | None = None, store_generated_datasets_paths: list[str | None] | None = None, store_results_path: str | None = None):
        """Run the Pipeline

        Args:
            complexity_range (_type_): Range of number of shadow models to test the attacks
            run_per_range (_type_): Number of attack per complexity point
            number_of_tests (_type_): Number of test per attack
            plot_style (Literal['single', 'double'] | None, optional): indicate if the user wants to get attack plots for every generators, and their styles.
            benchmarking_metric (BenchmarkMetric | None, optional): class containing the metric to compute the final benchmark rank, None to skip that part
            imported_data_paths (list[str  |  None] | None, optional): Put path of the data at the position corresponding to the generator to load data instead of
            generating it, or None to generate the data. Put the all argument to None to generate data for all generators.
            store_generated_datasets_paths (list[str  |  None] | None, optional): Put path where you want to store generated data at the position corresponding to the generator, or None not to store them. Put the all argument to None to store nothing.
            store_results_path (str | None, optional): Path to store the results of the attack. If None, the results are not stored.
        """

        self.baseline_score = get_baseline_score(self.defender_data, self.target_record, 25)
        attack_results = []
        for i in range(len(self.generators)):
            imported_data_path = imported_data_paths[i] if (imported_data_paths is not None) else None
            store_generated_datasets_path = store_generated_datasets_paths[i] if (store_generated_datasets_paths is not None) else None
            M_0, S_0, M_1, S_1, gen_time = self.attack_one_generator(i, complexity_range, run_per_range, number_of_tests, imported_data_path, store_generated_datasets_path)
            attack_results.append({
                                    'M_0': M_0,
                                    'S_0': S_0,
                                    'M_1': M_1,
                                    'S_1': S_1,
                                    'gen_time' : gen_time
                                 })
            if plot_style == 'single':
                fig, axes = single_plot(np.array(complexity_range), (1 - np.array(M_0) + np.array(M_1)) / 2, np.array(S_0) + np.array(S_1), self.baseline_score)
                plt.show()
                if store_results_path is not None:
                    fig.savefig(os.path.join(store_results_path, f'{i.get_filename_label()}.png'), dpi=300)
            elif plot_style == 'double':
                fig, axes = double_plot(np.array(complexity_range), np.array(M_0), np.array(S_0), np.array(M_1), np.array(S_1), self.baseline_score)
                plt.show()
                if store_results_path is not None:
                    fig.savefig(os.path.join(store_results_path, f'{i.get_filename_label()}.png'), dpi=300)
        if benchmarking_metric:
            print('----[Benchmark ranks]----')
            generators_final_data = []
            for i in range(len(self.generators)):
                data_mean = (1 - np.array(M_0) + np.array(M_1)) / 2
                data_std = np.array(attack_results[i]['S_0']) + np.array(attack_results[i]['S_1'])
                benchmark_rank = benchmarking_metric.compute_rank(complexity_range, data_mean, data_std, self.baseline_score[0])
                benchmark_metric = benchmarking_metric.compute_metric(complexity_range, data_mean, data_std, self.baseline_score[0])
                print(f'{self.generators[i]} : {benchmark_rank}, {benchmark_metric}')
                generators_final_data.append({"Model": self.generators[i], "Speed": attack_results[i]['gen_time'], "Final_Score": benchmark_rank})
            print('----[Printing]----')
            print(generators_final_data)
            fig, axes = plot_generators_ranks(generators_final_data)
            plt.show()
            if store_results_path is not None:
                fig.savefig(os.path.join(store_results_path, f'{i.get_filename_label()}.png'), dpi=300)

    def attack_one_generator(self, generator_ind: int,complexity_range: list[int], run_per_range: int, number_of_tests: int, imported_data_path: str | None = None, path_to_store_generated_datasets: str | None = None):
        print('Attacking :', self.generators[generator_ind])
        number_of_generated_shadow_datasets = complexity_range[-1]
        if imported_data_path: #load data
            print('Import datasets from', imported_data_path)
            shadow_data_pool, gen_time = self._import_shadow_datasets(imported_data_path)
        else: #generate data
            print('Generate datasets')
            shadow_data_pool, gen_time = self._generate_shadow_datasets(self.threat_models[generator_ind], number_of_generated_shadow_datasets, path_to_store_generated_datasets)

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
            M_0.append(np.mean(P_0))
            S_0.append(np.std(P_0))
            M_1.append(np.mean(P_1))
            S_1.append(np.std(P_1))
        return M_0, S_0, M_1, S_1, gen_time
    
    def _sample_shadow_dataset(self, shadow_dataset_pool: list[tapas.datasets.dataset.TabularDataset], number_of_shadow_models: int):
        """
        Returns:
            list[tapas.datasets.dataset.TabularDataset, bool]: subset of shadow_dataset_pool of lenght number_of_shadow_models such that there is the same number of dataset with/without target.
        """
        assert number_of_shadow_models % 2 == 0, 'number_of_shadow_models should be a even integer'
        shadow_datasets = random.sample(shadow_dataset_pool[0], int(number_of_shadow_models / 2))
        shadow_datasets.extend(random.sample(shadow_dataset_pool[1], int(number_of_shadow_models / 2)))
        return shadow_datasets
    
    def _generate_shadow_datasets(self, threat_model, number_of_train_datasets: int, path_to_store_generated_datasets: str | None = None):
        start_time = time.time()
        shadow_datasets, shadow_labels = threat_model.generate_training_samples(number_of_train_datasets, ignore_memory=True)
        generation_time = time.time() - start_time
        normalized_gen_time = generation_time / (self.size_of_datasets * number_of_train_datasets)
        shadow_data = list(zip(shadow_datasets, shadow_labels))

        if path_to_store_generated_datasets:
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