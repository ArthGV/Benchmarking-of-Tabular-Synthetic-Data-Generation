"""
Code adapted form the Achille's Heel repository.
"""
import numpy as np

if __name__ == '__main__':
    # stdlib
    import sys
    import argparse

    # third party
    from tqdm import tqdm

    # Achille's Heel
    sys.path.append('.')
    sys.path.append('./libs/MIA-synthetic-main')
    from src.data_prep import read_data, read_metadata, discretize_dataset, normalize_cont_cols
    from src.feature_extractors import fit_ohe, apply_ohe
    from src.distance import compute_distances
    from tools.utils import dump_pickle

    # get the corresponding data split
    parser = argparse.ArgumentParser(description="Select a subset of the dataset based on split.")
    parser.add_argument('--split', type=int, default=-1, 
                        help="An integer representing the subset of the dataset to select.")
    parser.add_argument('--drop_duplicates', type=bool, default=False, 
                        help="Whether to drop duplicates or not.")
    split = parser.parse_args().split
    drop_duplicates = parser.parse_args().drop_duplicates

    if split==1:
        start_index = 0
        end_index = 168511 # not contained
    elif split==2:
        start_index = 168511
        end_index = 337023 # not contained
    else:
        import warnings
        # warn that it will likely lead to a memory error
        warnings.warn("Using the full dataset, this take a long time. It will likely lead to a memory error.")
        start_index = 0
        end_index = 337023

    print(f"Selected split: {split}, index from {start_index} to {end_index}")

    # Load data
    PATH_TO_METADATA = './data/ah_repo/2011 Census Microdata Teaching Discretized.json'
    PATH_TO_DATA = './data/census_header.csv'

    meta_data_og, categorical_cols, continuous_cols = read_metadata(PATH_TO_METADATA)
    df = read_data(PATH_TO_DATA, categorical_cols, continuous_cols)
    if drop_duplicates:
        df = df.drop_duplicates()
        print("Duplicated removed. New shape:", df.shape)
    df = discretize_dataset(df, categorical_cols)
    df = normalize_cont_cols(df, meta_data_og, df_aux = df, types = ('Float', 'Integer'))
    

    # preprocessing
    ohe, ohe_column_names = fit_ohe(df, categorical_cols, meta_data_og)
    df_ohe = apply_ohe(df.copy(), ohe, categorical_cols, ohe_column_names, continuous_cols)
    all_columns = list(df_ohe.columns)
    ohe_cat_indices = [all_columns.index(col) for col in ohe_column_names]
    continous_indices = [all_columns.index(col) for col in continuous_cols]

    # setup distance computations
    ALL_DISTANCES = dict()
    N_TO_SAVE = 5
    DISTANCE_METHOD = 'cosine'

    df_ohe_values = df_ohe.values    
    
    if drop_duplicates:
        targets = df_ohe.iloc[start_index:end_index].index # duplicated have already been removed
        assert len(targets) == end_index
    else:
        targets = df_ohe.drop_duplicates()

    # for each record, we compute its distance with all datapoints
    SAVE_FREQUENCY = 5000
    i_prev = 0
    for i, target_id in tqdm(enumerate(targets), total=len(targets)):
        target_record = df_ohe.loc[target_id].values
        distances_target = compute_distances(record=target_record, values=df_ohe_values, 
                                            ohe_cat_indices=ohe_cat_indices, continous_indices=continous_indices,
                                            n_cat_cols=len(categorical_cols), n_cont_cols=len(continuous_cols),
                                            method = DISTANCE_METHOD)

        # let's sort it already
        if i % SAVE_FREQUENCY==0 and i!=0:
            dump_pickle(ALL_DISTANCES, f"generated/most_vulnerable_records_no_duplicate/distances_{i_prev+start_index}_to_{i+start_index}.pkl")
            i_prev = i
            ALL_DISTANCES = dict()

        ALL_DISTANCES[target_id] = np.sort(distances_target)[:N_TO_SAVE]

    dump_pickle(ALL_DISTANCES, f"../generated/most_vulnerable_records_no_duplicate/distances_{i_prev+start_index}_to_{end_index-1}.pkl")