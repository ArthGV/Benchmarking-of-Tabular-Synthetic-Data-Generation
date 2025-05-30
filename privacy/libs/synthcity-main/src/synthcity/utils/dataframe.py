# third party
import pandas as pd


def constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Find constant value columns in a pandas dataframe.
    """
    return discrete_columns(dataframe, 1)


def discrete_columns(
    dataframe: pd.DataFrame, max_classes: int = 10, return_counts: bool = False
) -> list:
    """
    Find columns containing discrete values in a pandas dataframe.
    """
    return [
        (col, cnt) if return_counts else col
        for col, vals in dataframe.items()
        for cnt in [vals.nunique()]
        if cnt <= max_classes
    ]


def discrete_columns_metadata(
    dataframe: pd.DataFrame, max_classes: int = 10, return_counts: bool = False,
    metadata: dict = None
) -> list:
    """
    Find columns containing discrete values in a pandas dataframe.

    Ignore the max_classes parameter and use the metadata to determine the number of classes, if provided.
    """
    if metadata is None:
        return discrete_columns(dataframe, max_classes, return_counts)
    else:
        # otherwise, extract categorical columns from metadata
        cat_cols = [col_metadata['name'] for col_metadata in metadata if col_metadata['type'] == 'finite']
        if return_counts:
            cat_counts = [len(col_metadata['representation']) for col_metadata in metadata if col_metadata['type'] == 'finite']
            # return [(col_name, dataframe[col_name].nunique()) for col_name in cat_cols]
            return [(col_name, cat_counts) for col_name, cat_counts in zip(cat_cols, cat_counts)]
        else:
            return cat_cols