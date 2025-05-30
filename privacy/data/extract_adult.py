"""
This script is used to convert the adult folder from the UCI repository (https://archive.ics.uci.edu/dataset/2/adult) to a csv file, directly
usable with the "Achilles' heels: Vulnerable Record Identification in Synthetic Data" repository code.
The processing steps are quite simple:
* Load the adult.data and adult.test files
* Concatenate them
* Remove the rows containing '?' values
* Save the resulting dataframe to a csv file

The columns are named according to the metadata from this repository, so the csv can be directly used with this repository code.
"""

if __name__ == '__main__':
    import warnings
    import argparse
    from pathlib import Path

    import pandas as pd
    import numpy as np

    # ignore c engine warnings due to ', ' separator
    warnings.filterwarnings('ignore')

    # parse input folder and output path
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to the adult folder from the UCI repository', default='data/adult')
    parser.add_argument('--output_path', type=str, help='Path to the output csv file', default='data/adult.csv')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # load the adult data
    col_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income']    
    
    adult_train = pd.read_csv(input_path/'adult.data', sep=', ', names=col_names)
    adult_test = pd.read_csv(input_path/'adult.test', skiprows=1, sep=', ', names=col_names)
    adult_test['income'] = adult_test['income'].str.removesuffix('.')
    adult_full = pd.concat((adult_train, adult_test), axis=0)

    adult_full.to_csv(output_path, sep=',', index=False)