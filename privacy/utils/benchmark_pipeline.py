import tapas
import numpy as np
from typing import Literal
import random
from utils.plotting import single_plot, double_plot
from tools.baseline_attack import get_baseline_score
import pickle


class BenchmarkPipeline():
    def __init__(self, data, attack, generators: list[tapas.generators.Generator], target_record = None):
        
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

        i = int(len(self.defender_data) / 5)

        self.data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
            test_data= self.defender_data,
            aux_data= self.attacker_data,
            num_training_records= i
        )
        self.generator_knowledge = [
            tapas.threat_models.BlackBoxKnowledge(
                g,
                num_synthetic_records = i,
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

    def run(self, complexity_range: list[int], run_per_range, number_of_tests, plot_style: Literal['single', 'double'] = 'double', imported_data_paths: list[str | None] | None = None, store_generated_datasets_paths: list[str | None] | None = None):
        """Run the Pipeline

        Args:
            complexity_range (_type_): Range of number of shadow models to test the attacks
            run_per_range (_type_): Number of attack per complexity point
            number_of_tests (_type_): Number of test per attack
            plot_style (Literal['single', 'double'], optional): _description_. Defaults to 'double'.
            imported_data_paths (list[str  |  None] | None, optional): Put path of the data at the position corresponding to the generator to load data instead of
            generating it, or None to generate the data. Put the all argument to None to generate data for all generators.
            store_generated_datasets_paths (list[str  |  None] | None, optional): Put path where you want to store generated data at the position corresponding to the generator, or None not to store them. Put the all argument to None to store nothing.
        """
        
        self.baseline_score = get_baseline_score(self.defender_data, self.target_record, 25)
        for i in range(len(self.generators)):
            imported_data_path = imported_data_paths[i] if (imported_data_paths is not None) else None
            store_generated_datasets_path = store_generated_datasets_paths[i] if (store_generated_datasets_paths is not None) else None
            M_0, S_0, M_1,S_1 = self.benchmark_one_generator(i, complexity_range, run_per_range, number_of_tests, imported_data_path, store_generated_datasets_path)
            if plot_style == 'single':
                single_plot(np.array(complexity_range), 1 - np.array(M_0) + np.array(M_0), np.array(S_0) + np.array(S_1), self.baseline_score)
            elif plot_style == 'double':
                double_plot(np.array(complexity_range), np.array(M_0), np.array(S_0), np.array(M_1), np.array(S_1), self.baseline_score)

    def benchmark_one_generator(self, generator_ind: int,complexity_range: list[int], run_per_range: int, number_of_tests: int):
        print('--Generator :', self.generators[generator_ind],'--')
        print('Generate datasets')
        number_of_generated_shadow_datasets = complexity_range[-1]
        if imported_data_path: #load data
            print('Import datasets from', imported_data_path)
            shadow_data_pool = self._import_shadow_datasets(self.threat_models[generator_ind], imported_data_path)
        else: #generate data
            print('Generate datasets')
            shadow_data_pool = self._generate_shadow_datasets(self.threat_models[generator_ind], number_of_generated_shadow_datasets, path_to_store_generated_datasets)

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
                # print('AuROC :', roc_auc_score(truth_labels, attacker.attack_score(test_datasets)))
            M_0.append(np.mean(P_0))
            S_0.append(np.std(P_0))
            M_1.append(np.mean(P_1))
            S_1.append(np.std(P_1))
        return M_0, S_0, M_1, S_1
    
    def _sample_shadow_dataset(self, shadow_dataset_pool: list[tapas.datasets.dataset.TabularDataset], number_of_shadow_models: int):
        """_summary_

        Args:
            shadow_dataset_pool (_type_): _description_
            number_of_shadow_models (int): _description_

        Returns:
            list[tapas.datasets.dataset.TabularDataset, bool]: subset of shadow_dataset_pool of lenght number_of_shadow_models such that there is the same number of dataset with/without target.
        """
        assert number_of_shadow_models % 2 == 0, 'number_of_shadow_models should be a even integer'
        shadow_datasets = random.sample(shadow_dataset_pool[0], int(number_of_shadow_models / 2))
        shadow_datasets.extend(random.sample(shadow_dataset_pool[1], int(number_of_shadow_models / 2)))
        return shadow_datasets
    
    def _generate_shadow_datasets(self, threat_model, number_of_train_datasets: int, path_to_store_generated_datasets: str | None = None):
        shadow_datasets, shadow_labels = threat_model.generate_training_samples(number_of_train_datasets, ignore_memory=True)
        shadow_data = list(zip(shadow_datasets, shadow_labels))

        if path_to_store_generated_datasets:
            print("Path:", path_to_store_generated_datasets)
            with open(path_to_store_generated_datasets, "wb") as f:
                pickle.dump(shadow_data, f)

        shadow_data_0 = [sd for sd in shadow_data if not sd[1]]
        shadow_data_1 = [sd for sd in shadow_data if sd[1]]
        shadow_data_pool = [shadow_data_0, shadow_data_1]
        return shadow_data_pool
    
    def _import_shadow_datasets(self, threat_model, path:str):
        with open(path, "rb") as d:
            shadow_data = pickle.load(d)

        shadow_data_0 = [sd for sd in shadow_data if not sd[1]]
        shadow_data_1 = [sd for sd in shadow_data if sd[1]]
        shadow_data_pool = [shadow_data_0, shadow_data_1]
        return shadow_data_pool