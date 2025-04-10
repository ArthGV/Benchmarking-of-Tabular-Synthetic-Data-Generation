import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Categorical probabilities with Laplace smoothing
def estimate_categorical_probs(train, categories, k=1):
    cat_probs = {}
    for col in categories:
        counts = Counter(train[col])
        unique_vals = list(set(train[col].unique()).union(set(test_data[col].unique())))
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
        kde.fit(train[[col]])
        kde_models[col] = kde
    return kde_models


    
def compute_log_likelihood(row, cat_probs, kde_models):
    log_prob = 0.0

    # Categorical
    for col, probs in cat_probs.items():
        val = row[col]
        p = probs.get(val, 1e-8)  # unseen fallback
        log_prob += math.log(p)

    # Numerical
    for col, kde in kde_models.items():
        val = pd.DataFrame([[row[col]]], columns=[col])
        log_prob += kde.score_samples(val)[0]

    return log_prob



# scaled likelihoods (score between 0 and 1)
def get_score(train_ll, point_likelihood):
    return np.mean(train_ll <= point_likelihood)




def get_baseline_score(dataset, data_point):
    
    df_train = dataset.data
    df_test = data_point.data
    
    categorical_features = df_train.description.one_hot_cols
    numerical_features = [col for col in df_train.description.columns if col not in categorical_features]
    
    # Estimate distributions
    cat_probs = estimate_categorical_probs(df_train, categorical_features)
    kde_models = estimate_kdes(df_train, numerical_features)
    
    # Compute log-likelihoods
    train_ll = df_train.apply(lambda row: compute_log_likelihood(row, cat_probs, kde_models), axis=1)
    test_ll = df_test.apply(lambda row: compute_log_likelihood(row, cat_probs, kde_models), axis=1)
    
    scaled_scores = [get_score(ll, train_ll) for ll in test_ll]
    return scaled_scores
    


    
    
    
  


    