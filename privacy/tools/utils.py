# stdlib
import os
import pickle
from importlib import import_module

# third party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#
# Eda utils
#
def load_census(file_path):
    """
    Load the census dataset from a csv file, and convert the columns to categorical.

    Parameters
    ----------
    file_path : str
        The path to the csv file, without header.

    Returns
    -------
    pd.DataFrame
        The census dataset.
    """
    census =  pd.read_csv(file_path, sep=',',
        names=['region', 'residence_type', 'family_composition', 'population_base',
                'sex', 'age', 'marital_status', 'student', 'country_of_birth', 
                'health', 'ethnic_group', 'religion', 'economic_activity', 'occupation',
                'industry', 'hours_worked_per_week', 'approximated_social_grade'])

    for col in census.columns:
        census[col] = census[col].astype('category')

    return census


def load_adults(file_path, is_contain_na=True):
    """
    Load the adults dataset from a csv file, and convert the columns to categorical.

    Parameters
    ----------
    file_path : str
        The path to the csv file, without header.

    Returns
    -------
    pd.DataFrame
        The adults dataset.
    """
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'sex', 'native-country', 'income']
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                         'hours-per-week']
    
    adults = pd.read_csv(file_path, sep=',')

    if not is_contain_na:
        adults = adults.replace('?', np.nan).dropna()

    for cat_col in categorical_columns:
        adults[cat_col] = adults[cat_col].astype('category')

    return adults, numerical_columns, categorical_columns


#
#   Experiment utils
#

def aux_test_split(data, n_aux, n_test, np_rng) -> tuple:
    rows_idx = np.arange(len(data.data))
    idx_list = np_rng.choice(rows_idx, n_aux+n_test)
    aux_data = data.get_records(idx_list[:n_aux])
    test_data = data.get_records(idx_list[n_aux:])

    return aux_data, test_data

def tapas_train_test_split(dataset, train_size, test_size=None, target_column=None, random_state=42) -> tuple:
    labels = None if target_column is None else LabelEncoder().fit_transform(dataset.data[target_column])
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), train_size=train_size, 
                                            test_size=test_size,
                                             stratify=labels, 
                                             random_state=random_state)

    return dataset.get_records(train_idx), dataset.get_records(test_idx)


def get_hh_mm_ss(elapsed_time_seconds) -> str:
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def export_elapsed_time(training_ds_gen_time_elapsed, testing_ds_gen_time_elapsed, 
                            meta_classifier_training_time_elapsed, meta_classifier_testing_time_elapsed,
                            total_time_elapsed, file_path):
    lines = [
        f"Elapsed time generating training datasets {get_hh_mm_ss(training_ds_gen_time_elapsed)} [hh:mm:ss]\n",
        f"Elapsed time generating testing datasets {get_hh_mm_ss(testing_ds_gen_time_elapsed)} [hh:mm:ss]\n",
        f"Elapsed time training meta-classifier {get_hh_mm_ss(meta_classifier_training_time_elapsed)} [hh:mm:ss]\n",
        f"Elapsed time testing meta-classifier {get_hh_mm_ss(meta_classifier_testing_time_elapsed)} [hh:mm:ss]\n",
        f"Elapsed time total {get_hh_mm_ss(total_time_elapsed)} [hh:mm:ss]",
    ]

    with open(file_path, "w") as file:
        file.writelines(lines)


def get_class_object_from_fqdn(class_fqdn):
    parts = class_fqdn.rsplit('.', 1)
    return getattr(import_module(parts[0]), parts[1]) # parts[0]: package, parts[1]: class


def dump_pickle(object, file_path):
    with open(file_path, 'wb') as file: 
        pickle.dump(object, file)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


#
#   Results utils
#

def get_privacy_summaries(experiments_dir, selected_datasets, selected_generators):
    dataframes = []

    # Walk through the directory structure
    for subdir, _, files in os.walk(experiments_dir):
        for file in files:
            if file == "summary_metrics.csv":
                # Get the full path of the file
                file_path = os.path.join(subdir, file)
                
                # Split the path into components
                parts = file_path.split(os.sep)
                
                # Extract relevant hierarchy based on the folder structure
                # Assuming root_dir = "results"
                dataset = parts[3]         # Example: "dataset"
                threat_model = parts[4]    # Example: "threat_model"
                attack = parts[5]          # Example: "attack"
                generator = parts[6]       # Example: "generator"
                timestamp = parts[8]       # Example: "timestamp"
                
                if dataset in selected_datasets and generator in selected_generators:
                    # Load the CSV file into a DataFrame
                    df = pd.read_csv(file_path, index_col=0)
                    
                    # Add metadata columns
                    df['dataset'] = dataset
                    df['threat_model'] = threat_model
                    df['attack'] = attack
                    df['generator'] = generator
                    df['timestamp'] = timestamp
                    
                    # Append to the list
                    dataframes.append(df)

    # Concatenate all dataframes into one
    return pd.concat(dataframes, ignore_index=True)


def plot_privacy_metrics(summaries_df, title="Privacy Metrics", metrics=['accuracy', 'true_positive_rate', 
                                                'false_positive_rate', 'mia_advantage',
                                                'privacy_gain', 'auc']):
    n_rows = len(metrics) // 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24//n_cols, 12//n_rows))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # plot seaborn boxplot
        ax.boxplot(summaries_df[metric].values)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_yticks(np.arange(0, 1.1, 0.1))

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
        

def get_privacy_attack_summaries(experiments_dir, selected_datasets, selected_generators):
    summaries = []

    # Walk through the directory structure
    for subdir, _, files in os.walk(experiments_dir):
        for file in files:
            if file == "attack_summary.pkl":
                # Get the full path of the file
                file_path = os.path.join(subdir, file)
                
                # Split the path into components
                parts = file_path.split(os.sep)
                
                # Extract relevant hierarchy based on the folder structure
                # Assuming root_dir = "results"
                dataset = parts[3]         # Example: "dataset"
                generator = parts[6]       # Example: "generator"
                
                if dataset in selected_datasets and generator in selected_generators:
                    summaries.append(load_pickle(file_path))

    # Concatenate all dataframes into one
    return summaries