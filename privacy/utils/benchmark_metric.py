from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import auc

class BenchmarkMetric(ABC):

    @abstractmethod
    def compute_metric(self, complexity: list[int], data_mean: list[float], data_std: list[float], baseline_score: float):
        pass

    @abstractmethod
    def compute_rank_from_metric(self, metric: float):
        pass

    @abstractmethod
    def compute_rank(self, complexity: list[int], data_mean: list[float], data_std: list[float], baseline_score: float):
        pass

class AUCMetric(BenchmarkMetric):

    def __init__(self, rank_range: list[float]):
        self.rank_range = rank_range

    def compute_metric(self, complexity: list[int], data_mean: list[float], data_std: list[float], baseline_score: float):
        data_mean = np.array(data_mean)
        complexity = np.array(complexity)
        auc_score = auc(complexity, data_mean)
        return auc_score
        
    def compute_rank_from_metric(self, metric: float):
        idx = np.searchsorted(self.rank_range, metric, side='right') - 1
        rank = chr(ord('A') + idx)
        return rank

    def compute_rank(self, complexity: list[int], data_mean: list[float], data_std: list[float], baseline_score: float):
        metric = self.compute_metric(complexity, data_mean, data_std, baseline_score)
        return self.compute_rank_from_metric(metric)