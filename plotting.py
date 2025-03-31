import numpy as np
import matplotlib.pyplot as plt

def plot(mean: np.array, complexity: np.array, var: np.array = None):
    plt.scatter(complexity, mean, c='red')
    if var is not None:
        lower_bound = mean - var
        upper_bound = mean + var
        plt.scatter(complexity, mean, c='red')
        plt.fill_between(complexity, lower_bound, upper_bound, alpha=.3, color='red')