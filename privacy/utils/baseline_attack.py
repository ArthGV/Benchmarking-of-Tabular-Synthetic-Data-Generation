import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KernelDensity
import math


# Categorical probabilities with Laplace smoothing
def estimate_categorical_probs(train, categories, k=1):
    cat_probs = {}
    for col in categories:
        counts = Counter(train[col])
        # Get unique values only from the train dataset
        unique_vals = list(set(train[col].unique()))
        total = sum(counts.values())
        probs = {
            val: (counts[val] + k) / (total + k * len(unique_vals))
            for val in unique_vals
        }
        cat_probs[col] = probs
    return cat_probs


# KDE for numerical columns
def estimate_kdes(train, num_cols):
    kde_models = {}
    for col in num_cols:

        kde = KernelDensity(kernel="gaussian", bandwidth=2.0)
        # Reshape column to be 2D for fitting KDE
        kde.fit(train[[col]].values.reshape(-1, 1))
        kde_models[col] = kde
    return kde_models


def estimate_kdes_approx(train, num_cols, bins=50):
    kde_models = {}
    for col in num_cols:
        # Use histograms to approximate KDE
        counts, bin_edges = np.histogram(train[col], bins=bins, density=True)
        kde_models[col] = (counts, bin_edges)
    return kde_models


    
def compute_log_likelihood(row, cat_probs, kde_models):
    log_prob = 0.0

    # Categorical
    for col, probs in cat_probs.items():
        val = row[col]
        p = probs.get(val, 1e-8)  # unseen fallback
        log_prob += math.log(p)

    # Numerical
    for col, (counts, bin_edges) in kde_models.items():
        val = row[col]
        
        # Find the bin where the value falls
        bin_idx = np.digitize(val, bin_edges) - 1  # Bin index (subtract 1 since digitize returns 1-based index)
        
        # Avoid out-of-bound indices
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= len(counts):
            bin_idx = len(counts) - 1
        
        # Approximate the density by dividing the count in that bin by the total number of samples
        p = counts[bin_idx] / sum(counts)
        
        # Small fallback for values that might not fall into the expected bins
        p = max(p, 1e-8)  # prevent log(0)
        
        log_prob += math.log(p)

    return log_prob


# scaled likelihoods (score between 0 and 1)
def get_score(point_likelihood, train_ll):
    return np.mean(train_ll <= point_likelihood)




def get_baseline_score(dataset, data_point, num_bins, approx=True ):
    
    categorical_features = dataset.description.one_hot_cols
    numerical_features = [col for col in dataset.description.columns if col not in categorical_features]
    
    df_train = dataset.data
    df_test = data_point.data
    
    # Estimate distributions
    print("estimating categorical probs")
    cat_probs = estimate_categorical_probs(df_train, categorical_features)
    
    if approx: 
        print("estimating approx kdes(histogram)")
        kde_models = estimate_kdes_approx(df_train, numerical_features, num_bins)
    else: 
        print("estimating kdes")
        kde_models = estimate_kdes(df_train, numerical_features)
    
    # Compute log-likelihoods
    train_ll = df_train.apply(lambda row: compute_log_likelihood(row, cat_probs, kde_models), axis=1)
    test_ll = df_test.apply(lambda row: compute_log_likelihood(row, cat_probs, kde_models), axis=1)
    
    scaled_scores = [get_score(ll, train_ll) for ll in test_ll]
    print(scaled_scores)
    return scaled_scores