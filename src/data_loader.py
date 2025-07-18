import pandas as pd

def load_data(filepath):
    """
    Loads data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded dataframe.
    """
    return pd.read_csv(filepath)
