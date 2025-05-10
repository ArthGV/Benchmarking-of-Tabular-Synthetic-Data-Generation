from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from sklearn.metrics import auc

class BenchmarkMetric(ABC):

    @abstractmethod
    def compute_metric(self, complexity: list[int], data_mean: list[float], data_std: list[float], baseline_score: float):
        pass

    @abstractmethod
    def compute_rank_from_metric(self, metric: float) -> Literal['A', 'B', 'C', 'D', 'E', 'F', 'U']:
        """
        Output the generator rank, A being the best, F the worst and U unassigned if the metric/rank computation can fail.
        """
        pass

    def compute_rank(self, complexity: list[int], data_mean: list[float], data_std: list[float], baseline_score: float) -> Literal['A', 'B', 'C', 'D', 'E', 'F', 'U']:
        metric = self.compute_metric(complexity, data_mean, data_std, baseline_score)
        return self.compute_rank_from_metric(metric)

#Example of a BenchmarkMetric implementation using Area Under the Curve (AUC)
class AUCMetric(BenchmarkMetric):

    def __init__(self, rank_range: list[float] = [0, 0.17, 0.33, 0.5, 0.67, 0.83]):
        self.rank_range = rank_range

    def compute_metric(self, complexity: list[int], data_mean: list[float], data_std: list[float], baseline_score: float):
        data_mean = np.array(data_mean)
        complexity = np.array(complexity)
        auc_score = auc(complexity, data_mean) / (complexity[-1] - complexity[0])
        return auc_score
        
    def compute_rank_from_metric(self, metric: float) -> Literal['A', 'B', 'C', 'D', 'E', 'F', 'U']:
        idx = np.searchsorted(self.rank_range, metric, side='right') - 1
        rank = chr(ord('A') + idx)
        return rank